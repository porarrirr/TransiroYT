import asyncio
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from transiroyt import download_progress, model_downloads, youtube_file


def test_model_byte_progress_includes_cached_files_and_resumed_bytes(tmp_path, monkeypatch):
    import huggingface_hub as hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    monkeypatch.setitem(model_downloads.SPECS, "test", ("repo", "revision", ("cached", "new")))
    reports = []
    monkeypatch.setattr(model_downloads, "write_progress", lambda path, value: reports.append(dict(value)))
    clock = itertools.count(1, 0.3)
    monkeypatch.setattr(model_downloads.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(hub, "hf_hub_url", lambda repo, name, **kwargs: name)
    monkeypatch.setattr(hub, "get_hf_file_metadata", lambda name: SimpleNamespace(size=120))
    (tmp_path / "cached").write_bytes(b"a" * 40)

    def download(repo, name, *, local_files_only=False, tqdm_class=None, **kwargs):
        if local_files_only:
            if name == "cached":
                return tmp_path / name
            raise LocalEntryNotFoundError("missing")
        with tqdm_class(total=120, initial=20) as progress:
            progress.update(40)
            progress.display()
            progress.update(60)
            progress.display()
        (tmp_path / name).write_bytes(b"b" * 120)
        return tmp_path / name

    monkeypatch.setattr(hub, "hf_hub_download", download)
    model_downloads.fetch("test", tmp_path, tmp_path / "progress.json")
    assert any(r["downloaded"] == 60 and r["total"] == 160 for r in reports)
    assert any(r["downloaded"] == 100 and r["total"] == 160 for r in reports)
    assert reports[-1] == {"resource": "test", "state": "complete", "filename": None, "downloaded": 160, "total": 160}


def test_video_progress_combines_video_and_audio_and_handles_unknown_total(tmp_path):
    reporter = youtube_file.VideoDownloadProgress(tmp_path / "progress.json")
    reporter.formats({"requested_formats": [{"format_id": "video", "filesize": 100},
                                             {"format_id": "audio", "filesize": 20}]})
    reporter.hook({"status": "finished", "info_dict": {"format_id": "video"},
                   "downloaded_bytes": 100, "total_bytes": 100})
    reporter.hook({"status": "finished", "info_dict": {"format_id": "audio"},
                   "downloaded_bytes": 20, "total_bytes": 20})
    assert json.loads(reporter.path.read_text())["downloaded"] == 120
    assert json.loads(reporter.path.read_text())["total"] == 120
    reporter = youtube_file.VideoDownloadProgress(tmp_path / "unknown.json")
    reporter.formats({"format_id": "video"})
    reporter.hook({"status": "downloading", "info_dict": {"format_id": "video"}, "downloaded_bytes": 50})
    reporter.report("downloading", force=True)
    assert json.loads(reporter.path.read_text())["total"] is None
    assert json.loads(reporter.path.read_text())["downloaded"] == 50


async def test_progress_is_delivered_before_worker_finishes_and_cancel_stops_worker(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    stopped = asyncio.Event()
    seen = []

    async def worker(*args, **kwargs):
        assert kwargs["process_group"] is True
        download_progress.write_progress(path, {"resource": "gemma", "downloaded": 123, "total": 456})
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(download_progress, "run_process", worker)
    task = asyncio.create_task(download_progress.run_download(["worker"], path, seen.append))
    for _ in range(50):
        if seen:
            break
        await asyncio.sleep(.01)
    assert len(seen) == 1
    assert seen[0]["downloaded"] == 123 and seen[0]["total"] == 456
    assert seen[0]["updated_at"] > 0
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()


async def test_model_progress_reaches_job_status(tmp_path, monkeypatch):
    from transiroyt import file_jobs
    from transiroyt.config import Settings
    from transiroyt.dubbing_options import FileDubbingOptions

    entered = asyncio.Event()

    class Dubber:
        def __init__(self, settings, options, progress, *, download_progress, mobile):
            self.callback = download_progress
            self.progress = progress
        async def run(self):
            self.progress("translation", .25, "Gemmaで翻訳しています")
            self.callback({"resource": "gemma", "state": "downloading", "downloaded": 20, "total": 100})
            self.callback({"resource": "coreml", "state": "complete", "downloaded": 0, "total": None})
            self.progress("speakers", 1, "参照音声を確認しました")
            entered.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(file_jobs, "FileDubber", Dubber)
    jobs = file_jobs.FileDubbingJobs(Settings(data=tmp_path))
    jobs.start(FileDubbingOptions(Path("video"), Path("srt"), tmp_path / "out.mp4"))
    await asyncio.wait_for(entered.wait(), 1)
    status = jobs.public()
    assert status["stage"] == "translation" and status["fraction"] == .25
    # Translation waits for the Gemma download, so its row says so; other models stay in their rows.
    assert status["message"] == "Gemmaのモデルを取得しています（20%）"
    assert status["activities"]["translation"]["message"] == status["message"]
    assert status["activities"]["speakers"]["fraction"] == 1
    assert status["downloads"]["gemma"]["downloaded"] == 20
    await jobs.shutdown()


async def test_model_failure_is_reported_as_job_error(tmp_path):
    from transiroyt.config import Settings
    from transiroyt.dubbing_options import FileDubbingOptions
    from transiroyt.file_jobs import FileDubbingJobs
    from transiroyt.gemma_audio import GemmaUnavailable

    class Dubber:
        def __init__(self, options, progress):
            self.progress = progress

        async def run(self):
            self.progress("translation", 0, "Gemmaで翻訳しています")
            raise GemmaUnavailable("Gemma翻訳に失敗したため停止しました")

    jobs = FileDubbingJobs(Settings(data=tmp_path), dubber_factory=Dubber)
    jobs.start(FileDubbingOptions(Path("video"), Path("srt"), tmp_path / "out.mp4"))
    await jobs.task
    assert jobs.public()["state"] == "error"
    assert jobs.public()["stage"] == "translation"
    assert jobs.public()["error"] == "Gemma翻訳に失敗したため停止しました"


async def test_late_translation_updates_its_activity_without_resetting_synthesis(tmp_path):
    from transiroyt.config import Settings
    from transiroyt.dubbing_options import FileDubbingOptions
    from transiroyt.file_jobs import FileDubbingJobs

    class Dubber:
        def __init__(self, options, progress):
            self.progress = progress

        async def run(self):
            self.progress("translation", .25, "翻訳中")
            self.progress("synthesis", .1, "音声を生成中")
            self.progress("translation", 1, "翻訳完了")
            self.progress("speakers", 1, "参照音声を確認しました")
            status = jobs.public()
            assert (status["stage"], status["fraction"], status["message"]) == (
                "synthesis", .1, "音声を生成中")
            assert status["activities"]["translation"]["fraction"] == 1
            assert status["activities"]["speakers"]["fraction"] == 1
            self.progress("synthesis", .5, "音声を半分生成しました")
            assert jobs.public()["fraction"] == .5
            self.progress("mix", 0, "合成中")
            self.progress("translation", 1, "翻訳完了")
            assert jobs.public()["stage"] == "mix"
            return {"summary": {"segments": 1, "ready": 1, "skipped": 0}, "speakers": {}}

    jobs = FileDubbingJobs(Settings(data=tmp_path), dubber_factory=Dubber)
    jobs.start(FileDubbingOptions(Path("video"), Path("srt"), tmp_path / "out.mp4"))
    await jobs.task
    assert jobs.public()["state"] == "done"


def test_model_files_download_concurrently_and_report_combined_bytes(tmp_path, monkeypatch):
    import threading

    import huggingface_hub as hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    monkeypatch.setitem(model_downloads.SPECS, "parallel", ("repo", "revision", ("a", "b")))
    barrier = threading.Barrier(2)
    monkeypatch.setattr(hub, "hf_hub_url", lambda repo, name, **kwargs: name)
    monkeypatch.setattr(hub, "get_hf_file_metadata", lambda name: SimpleNamespace(size=100))

    def download(repo, name, *, local_files_only=False, tqdm_class=None, **kwargs):
        if local_files_only:
            raise LocalEntryNotFoundError("missing")
        barrier.wait(timeout=2)  # Sequential fetching cannot pass this barrier.
        with tqdm_class(total=100) as progress:
            progress.update(100)
            progress.display()
        path = tmp_path / name
        path.write_bytes(b"a" * 100)
        return path

    monkeypatch.setattr(hub, "hf_hub_download", download)
    model_downloads.fetch("parallel", tmp_path, tmp_path / "progress.json")
    result = json.loads((tmp_path / "progress.json").read_text())
    assert result["downloaded"] == result["total"] == 200
    assert result["state"] == "complete"


@pytest.mark.parametrize("explicit, expected", [(None, "1"), ("0", "0")])
def test_high_performance_mode_is_set_before_hub_fetch_and_respects_override(monkeypatch, explicit, expected):
    import os

    if explicit is None:
        monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    else:
        monkeypatch.setenv("HF_XET_HIGH_PERFORMANCE", explicit)
    monkeypatch.setattr(model_downloads.sys, "argv", ["worker", "gemma", "cache", "progress"])
    seen = []
    monkeypatch.setattr(model_downloads, "fetch", lambda *args: seen.append(os.environ["HF_XET_HIGH_PERFORMANCE"]))
    assert model_downloads.main() == 0 and seen == [expected]


def test_recent_download_rate_excludes_resumed_bytes_and_expires_old_speed():
    rate = download_progress.DownloadRate()
    assert rate.measure(1000, 3000, transferred=0, now=0) == {"speed": None, "eta": None}
    assert rate.measure(1500, 3000, transferred=500, now=1) == {"speed": 500, "eta": 3}
    assert rate.measure(2000, 3000, transferred=1000, now=2) == {"speed": 500, "eta": 2}
    rate.measure(2000, 3000, transferred=1000, now=4)
    assert rate.measure(2000, 3000, transferred=1000, now=5) == {"speed": 0, "eta": None}
    assert rate.measure(10, None, now=6) == {"speed": None, "eta": None}
    assert rate.measure(20, None, now=7) == {"speed": 10, "eta": None}


def test_youtube_uses_transport_speed_for_resumed_downloads(tmp_path):
    reporter = youtube_file.VideoDownloadProgress(tmp_path / "resumed.json")
    reporter.formats({"format_id": "video", "filesize": 1000})
    reporter.hook({"status": "downloading", "info_dict": {"format_id": "video"},
                   "downloaded_bytes": 800, "total_bytes": 1000, "speed": 50})
    reporter.report("downloading", force=True)
    value = json.loads(reporter.path.read_text())
    assert value["speed"] == 50 and value["eta"] == 4


async def test_coreml_worker_byte_events_reach_download_status(tmp_path, monkeypatch):
    import sys

    from transiroyt import coreml_tts

    worker = tmp_path / "worker"
    worker.write_text(f"#!{sys.executable}\n" + '''import json, sys
for line in sys.stdin:
    for value in [
        {"event": "download", "completed": 0, "total": 2, "filename": "model"},
        {"event": "download", "downloaded": 800, "total_bytes": 1000, "transferred": 0, "filename": "model"},
        {"event": "download", "downloaded": 900, "total_bytes": 1000, "transferred": 100, "filename": "model"},
        {"event": "download", "completed": 2, "total": 2, "filename": "Complete"},
        {"ok": True},
    ]:
        print(json.dumps(value), flush=True)
''')
    worker.chmod(0o700)
    monkeypatch.setattr(coreml_tts, "coreml_binary", lambda settings: worker)
    reports = []
    tts = coreml_tts.CoreMLTTS(SimpleNamespace(data=tmp_path), download_progress=reports.append)
    try:
        assert await tts._request({"op": "prepare"}) == {"ok": True}
        assert reports[0]["state"] == "preparing"
        assert reports[2]["downloaded"] == 800 and reports[2]["speed"] is None
        assert reports[3]["downloaded"] == 900 and reports[3]["total"] == 1000
        assert reports[3]["speed"] > 0 and reports[3]["eta"] > 0
        assert reports[-2]["state"] == "preparing" and reports[-2]["completed_files"] == 2
        assert reports[-1]["state"] == "ready"
    finally:
        await tts.close()
