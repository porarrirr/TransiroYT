import shutil
import subprocess

import pytest
from transiroyt.embedded_subtitles import extract_subtitles
from transiroyt.errors import FileDubbingError
from transiroyt.subtitle_files import parse_srt


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
@pytest.mark.asyncio
async def test_extract_real_container_and_missing_track(tmp_path):
    srt = tmp_path / "source.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello world.\n")
    video = tmp_path / "source.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=32x32:duration=1",
                    "-i", str(srt), "-c:v", "mpeg4", "-c:s", "srt", str(video)], check=True)
    output = tmp_path / "extracted.srt"
    assert await extract_subtitles(video, output)
    assert parse_srt(output.read_text())[0].text == "Hello world."
    bare = tmp_path / "bare.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-an", "-sn", "-c:v", "copy", str(bare)], check=True)
    missing = tmp_path / "missing.srt"
    assert not await extract_subtitles(bare, missing)
    assert not missing.exists()


@pytest.mark.asyncio
async def test_ambiguous_tracks_require_external_srt(tmp_path, monkeypatch):
    async def probe(*args):
        return b'{"streams":[{"index":2,"codec_name":"subrip"},{"index":3,"codec_name":"subrip"}]}'
    monkeypatch.setattr("transiroyt.embedded_subtitles.run_process", probe)
    with pytest.raises(FileDubbingError, match="複数"):
        await extract_subtitles(tmp_path / "source.mkv", tmp_path / "out.srt")


@pytest.mark.asyncio
async def test_embedded_job_extracts_before_invoking_existing_dubber(tmp_path, monkeypatch):
    from transiroyt.config import Settings
    from transiroyt.dubbing_options import FileDubbingOptions
    from transiroyt.file_jobs import FileDubbingJobs

    video = tmp_path / "source.mp4"
    video.touch()
    async def extract(source, target):
        assert source == video
        target.parent.mkdir(parents=True)
        target.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello.\n")
        return True
    monkeypatch.setattr("transiroyt.embedded_subtitles.extract_subtitles", extract)
    class Dubber:
        def __init__(self, options, progress):
            options.validate()
            assert parse_srt(options.subtitles.read_text())[0].text == "Hello."
        async def run(self):
            return {"summary": {"ready": 1}, "speakers": {}}
    jobs = FileDubbingJobs(Settings(data=tmp_path / "data"), dubber_factory=Dubber)
    jobs.start(FileDubbingOptions(video, tmp_path / "unused.srt", tmp_path / "out.mp4"), embedded_subtitles=True)
    await jobs.task
    assert jobs.public()["state"] == "done"


@pytest.mark.asyncio
async def test_video_without_subtitles_is_transcribed_by_the_dubber(tmp_path, monkeypatch):
    from transiroyt.config import Settings
    from transiroyt.dubbing_options import FileDubbingOptions
    from transiroyt.file_jobs import FileDubbingJobs

    video = tmp_path / "source.mp4"
    video.touch()

    async def extract(source, target):
        return False

    monkeypatch.setattr("transiroyt.embedded_subtitles.extract_subtitles", extract)
    seen = []

    class Dubber:
        def __init__(self, options, progress):
            options.validate()
            seen.append(options)

        async def run(self):
            return {"summary": {"ready": 1}, "speakers": {}}

    jobs = FileDubbingJobs(Settings(data=tmp_path / "data"), dubber_factory=Dubber)
    jobs.start(FileDubbingOptions(video, tmp_path / "unused.srt", tmp_path / "out.mp4"), embedded_subtitles=True)
    await jobs.task
    options, = seen
    assert options.transcribe and options.subtitles.name == "transcribed.srt"
    assert jobs.public()["state"] == "done" and jobs.public()["transcribed"]
    assert jobs.public()["subtitles"] == str(options.subtitles)
