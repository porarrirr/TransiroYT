import asyncio
import json
from pathlib import Path

import httpx
import pytest
from transiroyt.app import create_app
from transiroyt.config import Settings
from transiroyt.library import CACHE_DIRECTORIES, DubbingLibrary


def export(directory, name="sample", *, title="作成した動画", size=16):
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{name}.ja.mp4"
    output.write_bytes(b"v" * size)
    output.with_suffix(".dubbing.json").write_text(json.dumps({
        "version": "file-dubbing-v1", "output": str(output), "title": title, "duration": 123.5,
    }))
    return output


def test_library_imports_existing_exports_and_remembers_previous_destinations(tmp_path):
    library = DubbingLibrary(tmp_path / "data")
    first = export(tmp_path / "old", size=21)
    second = export(tmp_path / "new", size=32)
    library.remember(first.parent)
    library.remember(first.parent)
    library.remember(second.parent)
    snapshot = DubbingLibrary(library.data).snapshot()
    assert len(snapshot["entries"]) == 2
    assert snapshot["total_bytes"] == 53
    assert {entry["output"] for entry in snapshot["entries"]} == {str(first), str(second)}
    assert snapshot["warnings"] == []
    first.unlink()
    missing = next(item for item in library.snapshot()["entries"] if item["output"] == str(first))
    assert not missing["available"]
    assert missing["size_bytes"] == 0
    assert library.snapshot()["total_bytes"] == 32


def test_library_ignores_unrelated_reports_and_reports_malformed_json(tmp_path):
    output = export(tmp_path, title=None)
    library = DubbingLibrary(tmp_path / "data")
    library.remember(tmp_path)
    (tmp_path / "broken.dubbing.json").write_text("{")
    (tmp_path / "foreign.dubbing.json").write_text(json.dumps({"version": "file-dubbing-v1", "output": str(output)}))
    snapshot = library.snapshot()
    assert len(snapshot["entries"]) == 1
    assert snapshot["entries"][0]["title"] == "sample.ja"
    assert len(snapshot["warnings"]) == 1


def test_cache_clear_preserves_exports_models_settings_and_symlink_targets(tmp_path):
    data = tmp_path / "data"
    library = DubbingLibrary(data)
    output = export(tmp_path / "exports")
    library.remember(output.parent)
    for name in CACHE_DIRECTORIES:
        (data / name).mkdir()
        (data / name / "working.wav").write_bytes(b"w" * 11)
    for name in ["models/weights", "cache/torch/model", "token", "settings.json"]:
        path = data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"keep")
    source = tmp_path / "original.mp4"
    source.write_bytes(b"source")
    (data / "file-dubbing-downloads/source.mp4").symlink_to(source)
    (data / "file-dubbing/models").symlink_to(data / "models", target_is_directory=True)
    assert library.cache_status()["total_bytes"] == 44
    assert library.clear_cache()["total_bytes"] == 0
    assert output.is_file() and output.with_suffix(".dubbing.json").is_file()
    assert source.read_bytes() == b"source"
    for name in ["models/weights", "cache/torch/model", "token", "settings.json"]:
        assert (data / name).read_bytes() == b"keep"
    assert len(library.snapshot()["entries"]) == 1


def test_cache_clear_rejects_export_destination_inside_cache(tmp_path):
    library = DubbingLibrary(tmp_path)
    output = export(tmp_path / "file-dubbing/exports")
    library.remember(output.parent)
    with pytest.raises(ValueError, match="完成動画"):
        library.clear_cache()
    assert output.is_file()
    destination = tmp_path / "exports"
    destination.mkdir()
    output.rename(destination / output.name)
    output.with_suffix(".dubbing.json").rename(destination / output.with_suffix(".dubbing.json").name)
    assert library.clear_cache()["total_bytes"] == 0


async def test_library_and_cache_apis_require_auth_and_refresh_after_clear(tmp_path):
    data = tmp_path / "data"
    output = export(tmp_path / "exports with spaces")
    (data / "file-dubbing").mkdir(parents=True)
    (data / "file-dubbing/work").write_bytes(b"cache")
    app = create_app(Settings(data=data, token="t"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        for method, path in [("GET", "/api/library"), ("GET", "/api/cache"), ("DELETE", "/api/cache")]:
            assert (await client.request(method, path)).status_code == 401
        client.headers["Authorization"] = "Bearer t"
        response = await client.get("/api/library", params={"directory": str(output.parent)})
        assert response.json()["entries"][0]["output"] == str(output)
        assert (await client.get("/api/cache")).json()["total_bytes"] == 5
        assert (await client.delete("/api/cache")).json()["total_bytes"] == 0
        assert (await client.get("/api/library")).json()["total_bytes"] == 16


async def test_cache_clear_is_blocked_during_job_and_job_registers_export(tmp_path):
    started, finish = asyncio.Event(), asyncio.Event()

    class Dubber:
        def __init__(self, options, progress):
            self.options = options
            options.output = Path(options.output)

        async def run(self):
            started.set()
            await finish.wait()
            export(self.options.output.parent, self.options.output.stem.removesuffix(".ja"))
            return {"summary": {}, "speakers": {}}

    app = create_app(Settings(data=tmp_path / "data", token="t"), file_dubber_factory=Dubber)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1",
                                 headers={"Authorization": "Bearer t"}) as client:
        result = await client.post("/api/file-dubbing", json={"video": str(tmp_path / "source.mp4"),
                                   "subtitles": str(tmp_path / "source.srt"), "output": str(tmp_path / "exports/sample.ja.mp4")})
        assert result.status_code == 202
        await started.wait()
        assert (await client.delete("/api/cache")).status_code == 409
        finish.set()
        for _ in range(50):
            await asyncio.sleep(0)
            if (tmp_path / "data/library-directories.json").exists():
                break
        assert (await client.get("/api/library")).json()["entries"][0]["output"] == str(tmp_path / "exports/sample.ja.mp4")


async def test_start_is_blocked_while_cache_clear_runs(tmp_path, monkeypatch):
    import threading

    entered, finish = threading.Event(), threading.Event()
    original = DubbingLibrary.clear_cache

    def clear(library):
        entered.set()
        assert finish.wait(timeout=5)
        return original(library)

    monkeypatch.setattr(DubbingLibrary, "clear_cache", clear)
    app = create_app(Settings(data=tmp_path, token="t"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1",
                                 headers={"Authorization": "Bearer t"}) as client:
        clearing = asyncio.create_task(client.delete("/api/cache"))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            response = await client.post("/api/file-dubbing", json={"video": str(tmp_path / "source.mp4"),
                                         "subtitles": str(tmp_path / "source.srt"), "output": str(tmp_path / "output.mp4")})
            assert response.status_code == 409
            assert "キャッシュ" in response.json()["detail"]
        finally:
            finish.set()
            assert (await clearing).status_code == 200


def test_automatic_cache_expiry_and_size_limit(tmp_path, monkeypatch):
    import os

    import transiroyt.library as cache_module

    now = 2_000_000_000
    monkeypatch.setattr(cache_module.time, "time", lambda: now)
    monkeypatch.setattr(cache_module, "CACHE_MAX_BYTES", 30)
    library = DubbingLibrary(tmp_path)
    units = []
    for name, age in [("expired", cache_module.CACHE_MAX_AGE), ("old", 100), ("new", 10)]:
        directory = tmp_path / "file-dubbing" / name
        directory.mkdir(parents=True)
        audio = directory / "audio.wav"
        audio.write_bytes(b"a" * 20)
        os.utime(audio, (now - age, now - age))
        os.utime(directory, (now - age, now - age))
        units.append(directory)
    library.prune_cache()
    assert not units[0].exists() and not units[1].exists()
    assert (units[2] / "audio.wav").read_bytes() == b"a" * 20
    assert library.cache_status()["total_bytes"] == 20


def test_automatic_cache_keeps_exact_limit_and_uses_latest_nested_update(tmp_path, monkeypatch):
    import os

    import transiroyt.library as cache_module

    monkeypatch.setattr(cache_module, "CACHE_MAX_BYTES", 30)
    directory = tmp_path / "file-dubbing/work"
    directory.mkdir(parents=True)
    (directory / "fresh.wav").write_bytes(b"a" * 30)
    os.utime(directory, (1, 1))
    DubbingLibrary(tmp_path).prune_cache()
    assert (directory / "fresh.wav").stat().st_size == 30


def test_automatic_cache_removes_reference_pair_without_following_symlinks(tmp_path, monkeypatch):
    import transiroyt.library as cache_module

    monkeypatch.setattr(cache_module, "CACHE_MAX_BYTES", 0)
    root = tmp_path / "reference-conversion"
    root.mkdir()
    (root / "sample.wav").write_bytes(b"a")
    (root / "sample.json").write_text("{}")
    source = tmp_path / "models/weights"
    source.parent.mkdir()
    source.write_bytes(b"keep")
    (root / "linked").symlink_to(source.parent, target_is_directory=True)
    library = DubbingLibrary(tmp_path)
    library.prune_cache()
    assert not (root / "sample.wav").exists()
    assert not (root / "sample.json").exists()
    assert source.read_bytes() == b"keep"
    assert library.cache_status()["total_bytes"] == 0


def test_automatic_cache_preserves_exports_inside_cache(tmp_path, monkeypatch):
    import transiroyt.library as cache_module

    monkeypatch.setattr(cache_module, "CACHE_MAX_BYTES", 0)
    output = export(tmp_path / "file-dubbing/exports")
    library = DubbingLibrary(tmp_path)
    library.remember(output.parent)
    with pytest.raises(ValueError, match="完成動画"):
        library.prune_cache()
    assert output.is_file()


async def test_automatic_maintenance_skips_running_jobs_and_recovers_from_failure(tmp_path, monkeypatch):
    from transiroyt.file_jobs import FileDubbingJobs

    calls = []

    def prune(library):
        calls.append(library.data)
        raise OSError("disk unavailable")

    monkeypatch.setattr(DubbingLibrary, "prune_cache", prune)
    jobs = FileDubbingJobs(Settings(data=tmp_path))
    jobs.task = asyncio.create_task(asyncio.Event().wait())
    await jobs.maintain_cache()
    assert calls == []
    jobs.task.cancel()
    await asyncio.gather(jobs.task, return_exceptions=True)
    jobs.clearing_cache = True
    await jobs.maintain_cache()
    assert calls == []
    jobs.clearing_cache = False
    await jobs.maintain_cache()
    assert calls == [tmp_path]
    assert not jobs.clearing_cache


async def test_automatic_cache_runs_on_application_startup(tmp_path, monkeypatch):
    import os
    from unittest.mock import AsyncMock

    from transiroyt.mobile_workers import MobileWorkers

    monkeypatch.setattr(MobileWorkers, "resume", AsyncMock())
    monkeypatch.setattr(MobileWorkers, "close", AsyncMock())
    old = tmp_path / "file-dubbing/expired.wav"
    old.parent.mkdir()
    old.write_bytes(b"old")
    os.utime(old, (1, 1))
    app = create_app(Settings(data=tmp_path, token="t"))
    async with app.router.lifespan_context(app):
        assert not old.exists()


@pytest.mark.parametrize("outcome", ["done", "error", "cancelled"])
async def test_automatic_cache_runs_after_job_stops(tmp_path, outcome):
    import os

    from transiroyt.dubbing_options import FileDubbingOptions
    from transiroyt.errors import FileDubbingError
    from transiroyt.file_jobs import FileDubbingJobs

    started = asyncio.Event()
    stop = asyncio.Event()
    old = tmp_path / "file-dubbing/expired.wav"
    old.parent.mkdir()
    old.write_bytes(b"old")
    os.utime(old, (1, 1))

    class Dubber:
        async def run(self):
            started.set()
            await stop.wait()
            if outcome == "error":
                raise FileDubbingError("failed")
            return {"summary": {}, "speakers": {}}

    jobs = FileDubbingJobs(Settings(data=tmp_path), lambda *_: Dubber())
    job = jobs.start(FileDubbingOptions(video=tmp_path / "source.mp4", subtitles=tmp_path / "source.srt",
                                       output=tmp_path / "exports/sample.ja.mp4"))
    await started.wait()
    assert old.exists()
    if outcome == "cancelled":
        await jobs.cancel(job["id"])
    else:
        stop.set()
        await jobs.task
    assert jobs.public()["state"] == outcome
    assert not old.exists()
    assert not jobs.clearing_cache


async def test_job_stays_running_until_automatic_cleanup_finishes(tmp_path, monkeypatch):
    import threading

    from transiroyt.dubbing_options import FileDubbingOptions
    from transiroyt.errors import FileDubbingError
    from transiroyt.file_jobs import FileDubbingJobs

    entered, finish = threading.Event(), threading.Event()

    def prune(library):
        entered.set()
        assert finish.wait(timeout=5)

    class Dubber:
        async def run(self):
            return {"summary": {}, "speakers": {}}

    monkeypatch.setattr(DubbingLibrary, "prune_cache", prune)
    jobs = FileDubbingJobs(Settings(data=tmp_path), lambda *_: Dubber())
    options = FileDubbingOptions(video=tmp_path / "source.mp4", subtitles=tmp_path / "source.srt",
                                 output=tmp_path / "exports/sample.ja.mp4")
    jobs.start(options)
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert jobs.public()["state"] == "running"
        assert "キャッシュ" in jobs.public()["message"]
        with pytest.raises(FileDubbingError):
            jobs.start(options)
    finally:
        finish.set()
        await jobs.task
    assert jobs.public()["state"] == "done"


def test_library_metadata_survives_working_cache_deletion(tmp_path):
    library = DubbingLibrary(tmp_path / "data")
    output = export(tmp_path / "exports")
    path = output.with_suffix(".dubbing.json")
    report = json.loads(path.read_text())
    report.update(uploader="投稿者", thumbnail_url="https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg")
    path.write_text(json.dumps(report))
    library.remember(output.parent)
    library.clear_cache()
    entry, = library.snapshot()["entries"]
    assert entry["uploader"] == "投稿者"
    assert entry["thumbnail_url"] == report["thumbnail_url"]
