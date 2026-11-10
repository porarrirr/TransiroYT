import asyncio
import io
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from transiroyt import file_jobs, youtube_file
from transiroyt.api_models import StartFileDubbing
from transiroyt.config import Settings
from transiroyt.dubbing_options import FileDubbingOptions
from transiroyt.errors import FileDubbingError
from transiroyt.subtitle_files import parse_srt


@pytest.mark.parametrize("url", ["https://youtu.be/abcdefghijk?t=2", "https://www.youtube.com/watch?v=abcdefghijk&list=123",
                                 "https://m.youtube.com/shorts/abcdefghijk", "https://youtube.com/embed/abcdefghijk"])
def test_video_url(url):
    assert youtube_file.youtube_video_id(url) == "abcdefghijk"


@pytest.mark.parametrize("url", ["file:///tmp/video", "https://youtube.com.evil.com/watch?v=abcdefghijk",
                                 "https://youtube.com/playlist?list=abc", "https://youtu.be/abc",
                                 "https://user@youtube.com/watch?v=abcdefghijk", "https://youtube.com:443/watch?v=abcdefghijk"])
def test_invalid_url_rejected_by_api(url):
    with pytest.raises(ValidationError):
        StartFileDubbing(youtube_url=url, output="out.mp4")


def test_source_contract():
    assert StartFileDubbing(youtube_url="https://youtu.be/abcdefghijk", output="out.mp4").video is None
    assert StartFileDubbing(video="v.mp4", output="out.mp4").subtitles is None
    for fields in [{}, {"subtitles": "s.srt"}, {"youtube_url": "https://youtu.be/abcdefghijk", "subtitles": "s.srt"}]:
        with pytest.raises(ValidationError):
            StartFileDubbing(output="out.mp4", **fields)


def test_download_original_captions_before_video(tmp_path, monkeypatch):
    import yt_dlp

    calls = []

    class YDL:
        def __init__(self, options):
            assert options["skip_download"] is False and options["noplaylist"] is True
            assert options["merge_output_format"] == "mp4"
            assert options["cookiesfrombrowser"] == ("chrome",)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def extract_info(self, url, download):
            assert url == "https://www.youtube.com/watch?v=abcdefghijk" and not download
            return {"language": "en", "title": "Test", "channel": "Test Channel",
                    "thumbnail": "https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg", "subtitles": {}, "automatic_captions": {
                "ja": [{"ext": "json3", "url": "https://captions?tlang=ja"}],
                "en-orig": [{"ext": "json3", "url": "https://captions"}]}}
        def urlopen(self, url):
            assert url == "https://captions"
            calls.append("captions")
            return io.BytesIO(json.dumps({"events": [{"tStartMs": 1234, "dDurationMs": 2000,
                                                       "segs": [{"utf8": "Hello &amp; hi"}]}]}).encode())
        def process_info(self, info):
            calls.append("video")
            (tmp_path / "source.mp4").write_bytes(b"video")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", YDL)
    result = youtube_file.download_once(Settings(cookies_browser="chrome"), "abcdefghijk", tmp_path, "auto")
    assert calls == ["captions", "video"] and result["language"] == "en"
    assert result["uploader"] == "Test Channel"
    assert result["thumbnail_url"] == "https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg"
    cue, = parse_srt((tmp_path / "source.srt").read_text())
    assert (cue.start, cue.end, cue.text) == (1.234, 3.234, "Hello & hi")


async def test_youtube_api_download_then_dub_and_cancel(tmp_path, monkeypatch):
    from transiroyt.app import create_app

    entered, release = asyncio.Event(), asyncio.Event()
    seen = []

    async def download(settings, url, directory, language, *, quality="best", progress=None):
        assert quality == "720"
        entered.set()
        await release.wait()
        directory.mkdir(parents=True)
        (directory / "source.mp4").write_bytes(b"video")
        (directory / "source.srt").write_text("1\n00:00:00,000 --> 00:00:02,000\nHello\n")
        return {"language": "en", "subtitles": True}

    class Dubber:
        def __init__(self, options, progress):
            options.validate()
            seen.append(options)
        async def run(self):
            return {"summary": {"ready": 1, "segments": 1, "skipped": 0}, "speakers": {}}

    monkeypatch.setattr(file_jobs, "download_youtube", download)
    app = create_app(Settings(data=tmp_path / "data", token="t"), file_dubber_factory=Dubber)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1",
                                    headers={"Authorization": "Bearer t"}) as client:
            body = {"youtube_url": "https://youtu.be/abcdefghijk", "output": str(tmp_path / "out.mp4"),
                    "original_volume": 0.4, "dub_volume": 1.3, "speakers": 2, "youtube_quality": "720"}
            reply = await client.post("/api/file-dubbing", json=body)
            assert reply.status_code == 202
            await asyncio.wait_for(entered.wait(), 2)
            assert (await client.get("/api/file-dubbing")).json()["job"]["stage"] == "download"
            assert (await client.post("/api/file-dubbing", json=body)).status_code == 409
            release.set()
            for _ in range(20):
                job = (await client.get("/api/file-dubbing")).json()["job"]
                if job["state"] != "running":
                    break
                await asyncio.sleep(0.01)
            assert job["state"] == "done"
            assert (seen[0].source_language, seen[0].original_volume, seen[0].dub_volume, seen[0].speakers) == ("en", 0.4, 1.3, 2)
            entered.clear()
            release.clear()
            job = (await client.post("/api/file-dubbing", json=body)).json()["job"]
            await asyncio.wait_for(entered.wait(), 2)
            cancelled = (await client.delete(f"/api/file-dubbing/{job['id']}")).json()["job"]
            assert cancelled["state"] == "cancelled" and len(seen) == 1


async def test_download_failure_does_not_start_dubber(tmp_path, monkeypatch):
    async def fail(*args, **kwargs):
        raise FileDubbingError("利用できる字幕がありません。")
    monkeypatch.setattr(file_jobs, "download_youtube", fail)
    jobs = file_jobs.FileDubbingJobs(Settings(data=tmp_path), lambda *_: pytest.fail("must not dub"))
    jobs.start(FileDubbingOptions(Path(""), Path(""), tmp_path / "out.mp4"), youtube_url="https://youtu.be/abcdefghijk")
    await jobs.task
    assert jobs.public()["state"] == "error" and "字幕" in jobs.public()["error"]


async def test_youtube_without_captions_is_dubbed_from_transcribed_speech(tmp_path, monkeypatch):
    async def download(settings, url, directory, language, *, quality="best", progress=None):
        directory.mkdir(parents=True)
        (directory / "source.mp4").write_bytes(b"video")
        return {"language": None, "subtitles": False, "title": "Untitled talk", "uploader": "Channel",
                "thumbnail_url": "https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg"}

    seen = []

    class Dubber:
        def __init__(self, options, progress):
            options.validate()
            seen.append(options)
        async def run(self):
            return {"summary": {"ready": 1}, "speakers": {}}

    monkeypatch.setattr(file_jobs, "download_youtube", download)
    jobs = file_jobs.FileDubbingJobs(Settings(data=tmp_path), Dubber)
    jobs.start(FileDubbingOptions(Path(""), Path(""), tmp_path / "out.mp4"), youtube_url="https://youtu.be/abcdefghijk")
    await jobs.task
    options, = seen
    assert jobs.public()["state"] == "done"
    assert options.transcribe and options.source_language == "auto" and options.title == "Untitled talk"
    assert options.uploader == "Channel"
    assert options.thumbnail_url == "https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg"
    assert options.subtitles.name == "source.srt" and not options.subtitles.exists()


async def test_download_worker_imports_without_pythonpath_and_hides_errors(tmp_path, monkeypatch):
    import sys

    from transiroyt.process import ProcessError, run_process

    captured = []

    async def inspect_worker(*args, **kwargs):
        captured.extend(args)
        # Run the actual worker with an empty PYTHONPATH. Malformed ID has no network access.
        probe = list(args)
        probe[probe.index("abcdefghijk")] = "invalid"
        with pytest.raises(ProcessError) as error:
            await run_process(*probe, timeout=10)
        assert "ModuleNotFoundError" not in str(error.value)
        assert "Traceback" not in str(error.value)
        assert kwargs["process_group"] is True
        (tmp_path / "metadata.json").write_text('{"language":"en"}')

    monkeypatch.delenv("PYTHONPATH", raising=False)
    from transiroyt import download_progress
    monkeypatch.setattr(download_progress, "run_process", inspect_worker)
    await youtube_file.download_youtube(Settings(), "https://youtu.be/abcdefghijk", tmp_path)
    assert captured[0] == sys.executable


def test_video_without_original_captions_is_downloaded_for_transcription(tmp_path, monkeypatch):
    import yt_dlp

    class YDL:
        def __init__(self, options):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def extract_info(self, *args, **kwargs):
            # A machine translation is never taken as the original transcript.
            return {"language": "en", "title": "No captions", "subtitles": {}, "automatic_captions": {
                "ja": [{"ext": "json3", "url": "https://captions?tlang=ja"}]}}
        def urlopen(self, url):
            pytest.fail("no caption track to fetch")
        def process_info(self, info):
            (tmp_path / "source.mp4").write_bytes(b"video")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", YDL)
    result = youtube_file.download_once(Settings(), "abcdefghijk", tmp_path, "auto")
    assert result == {"language": None, "subtitles": False, "title": "No captions",
                      "uploader": None, "thumbnail_url": None}
    assert not (tmp_path / "source.srt").exists()


@pytest.mark.parametrize("info, message", [({"is_live": True}, "生配信")])
def test_unavailable_source_does_not_download_video(tmp_path, monkeypatch, info, message):
    import yt_dlp
    from transiroyt.captions import CaptionError

    class YDL:
        def __init__(self, options):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def extract_info(self, *args, **kwargs):
            return info
        def process_info(self, info):
            pytest.fail("must not download")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", YDL)
    with pytest.raises(CaptionError, match=message):
        youtube_file.download_once(Settings(), "abcdefghijk", tmp_path, "auto")


async def test_cancellation_kills_download_process_group(tmp_path):
    import os
    import subprocess
    import sys

    from transiroyt.process import run_process

    marker = tmp_path / "child.pid"
    code = "import subprocess,sys,time; from pathlib import Path; " \
           "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); " \
           "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(120)"
    task = asyncio.create_task(run_process(sys.executable, "-c", code, str(marker), process_group=True))
    for _ in range(100):
        if marker.exists():
            break
        await asyncio.sleep(0.01)
    assert marker.exists()
    child_pid = int(marker.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(100):
        status = subprocess.run(["ps", "-p", str(child_pid), "-o", "stat="], capture_output=True, text=True).stdout.strip()
        if not status or status.startswith("Z"):
            break
        await asyncio.sleep(0.01)
    else:
        os.kill(child_pid, 9)
        pytest.fail("download child survived cancellation")


def test_finder_app_detects_homebrew_llama_but_respects_explicit_path(monkeypatch):
    from transiroyt import gemma_audio

    monkeypatch.delenv("TRANSIROYT_LLAMA_SERVER", raising=False)
    monkeypatch.setattr(gemma_audio.shutil, "which", lambda path: path if path == "/opt/homebrew/bin/llama-server" else None)
    assert gemma_audio.llama_server_binary() == "/opt/homebrew/bin/llama-server"
    monkeypatch.setenv("TRANSIROYT_LLAMA_SERVER", "/missing/custom-server")
    with pytest.raises(gemma_audio.GemmaUnavailable):
        gemma_audio.llama_server_binary()


@pytest.mark.parametrize("quality", youtube_file.YOUTUBE_QUALITIES)
@pytest.mark.parametrize("extension", ["mp4", "webm"])
@pytest.mark.parametrize("combined", [False, True])
def test_quality_selects_available_video_under_ceiling(quality, extension, combined):
    import yt_dlp

    formats = [{"format_id": str(height), "url": f"https://example.com/{height}",
                "ext": extension, "height": height, "width": height * 16 // 9,
                "vcodec": "h264", "acodec": "aac" if combined else "none"}
               for height in [144, 360, 480, 720, 1080, 1440, 2160]]
    if not combined:
        formats.append({"format_id": "audio", "url": "https://example.com/audio", "ext": "m4a",
                        "vcodec": "none", "acodec": "aac"})
    # 480p is intentionally unavailable: the selected 480p ceiling must fall back to 360p.
    formats = [item for item in formats if item["format_id"] != "480"]
    with yt_dlp.YoutubeDL({"quiet": True, "format": youtube_file.video_format(quality)}) as ydl:
        info = ydl.process_ie_result({"id": "fixture", "title": "Fixture", "formats": formats}, download=False)
    expected = 2160 if quality == "best" else 360 if quality == "480" else int(quality)
    assert info["height"] == expected
    selected = info.get("requested_formats") or [info]
    assert any(item["vcodec"] != "none" for item in selected)
    assert any(item["acodec"] != "none" for item in selected)


@pytest.mark.parametrize("quality", ["auto", "9999", "720p", 720, None, "720/best"])
def test_invalid_quality_rejected(quality):
    with pytest.raises(ValidationError):
        StartFileDubbing(youtube_url="https://youtu.be/abcdefghijk", output="out.mp4", youtube_quality=quality)
    with pytest.raises(FileDubbingError, match="画質"):
        youtube_file.video_format(quality)


def test_consumer_youtube_requires_bundled_node_and_pins_ffmpeg(tmp_path):
    from transiroyt.captions import CaptionError, _youtube_options

    (tmp_path / "distribution.json").write_text("{}")
    settings = Settings(root=tmp_path)
    with pytest.raises(CaptionError, match="再インストール"):
        _youtube_options(settings)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/node").touch()
    options = _youtube_options(settings)
    assert options["js_runtimes"] == {"node": {"path": str(tmp_path / "bin/node")}}
    assert options["ffmpeg_location"] == str(tmp_path / "bin")


@pytest.mark.parametrize("wrapped,status", [(False, 429), (True, 429), (False, 403)])
def test_worker_reports_http_failures_without_signed_urls(tmp_path, monkeypatch, capsys, wrapped, status):
    import sys

    from yt_dlp.networking.common import Response
    from yt_dlp.networking.exceptions import HTTPError
    from yt_dlp.utils import DownloadError

    response = Response(io.BytesIO(), "https://captions.invalid/?signature=secret", {}, status=status)
    error = HTTPError(response)
    if wrapped:
        error = DownloadError(f"Unable to download video: {error}: https://media.invalid/?signature=secret")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(youtube_file, "download_once", fail)
    monkeypatch.setattr(sys, "argv", ["worker", "abcdefghijk", str(tmp_path), "auto", str(tmp_path), "", "360"])
    assert youtube_file.main() == 1
    message = capsys.readouterr().err.strip()
    assert message == (youtube_file.RATE_LIMIT_MESSAGE if status == 429 else youtube_file.GENERIC_MESSAGE)
    assert "secret" not in message and "https://" not in message
    assert not (tmp_path / "metadata.json").exists()
