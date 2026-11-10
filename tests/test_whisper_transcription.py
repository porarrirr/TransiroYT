import sys
import wave

import pytest
from transiroyt.config import Settings
from transiroyt.whisper_transcription import (
    MODEL_FILES,
    MODEL_REPO,
    MODEL_REVISION,
    TranscriptOutputLimit,
    WhisperTranscriber,
    WhisperUnavailable,
)

# Answers from the request itself: "<pcm bytes> <max_tokens>"; the expected
# text "崩壊" makes the worker exit, "ループする" leaves the transcript unfinished.
WORKER = '''import base64, json, os, sys
from pathlib import Path
assert Path(os.environ["TRANSIROYT_WHISPER_MODEL"], "weights.safetensors").is_file()
starts = Path(__file__).resolve().parents[1] / "starts"
starts.write_text(str(int(starts.read_text() if starts.exists() else 0) + 1))
print(json.dumps({"ok": True, "ready": True}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if "windows" in request:
        print(json.dumps({"ok": True, "texts": [f"{request['language']} {a}-{b}" for a, b in request["windows"]],
                          "complete": [b - a < 29 for a, b in request["windows"]]}), flush=True)
        continue
    if "detect" in request:
        print(json.dumps({"ok": True, "language": "zh", "probabilities": {"zh": len(request["detect"])}}), flush=True)
        continue
    if "pauses" in request:
        print(json.dumps({"ok": True, "pauses": [[1.0, 1.0 + request["pauses"]]]}), flush=True)
        continue
    if request.get("timestamps"):
        words = [{"start": 0.5, "end": 0.6, "word": "x" * 80} for _ in range(1000)] if "long" in request["audio"] else []
        print(json.dumps({"ok": True, "language": request["language"] or "en", "segments": [
            {"start": 0.5, "end": 1.5, "text": Path(request["audio"]).name, "words": words}]}), flush=True)
        continue
    with (Path(__file__).resolve().parents[1] / "languages").open("a") as log:
        log.write(request["language"] + chr(10))
    if request["max_tokens"] == 32 + 2 * len("崩壊"):
        sys.exit(1)
    text = f"{len(base64.b64decode(request['pcm']))} {request['max_tokens']}"
    print(json.dumps({"ok": True, "text": text, "complete": request["max_tokens"] != 32 + 2 * len("ループする")}), flush=True)
'''


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSIROYT_TTS_PYTHON", sys.executable)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/whisper_worker.py").write_text(WORKER)
    return Settings(root=tmp_path, data=tmp_path)


def install_model(settings):
    repo = settings.data / "models/huggingface" / ("models--" + MODEL_REPO.replace("/", "--"))
    snapshot = repo / "snapshots" / MODEL_REVISION
    snapshot.mkdir(parents=True)
    for name in MODEL_FILES:
        (snapshot / name).write_text("{}")


def take(path, seconds=1.0):
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(round(48000 * seconds) * 2))
    return path


async def test_transcribes_16khz_takes_in_one_worker_and_restarts_after_a_crash(settings, tmp_path):
    install_model(settings)
    transcriber = WhisperTranscriber(settings)
    try:
        assert await transcriber.transcribe(take(tmp_path / "a.wav"), "テスト") == f"{16000 * 2} {32 + 2 * 3}"
        assert await transcriber.transcribe(take(tmp_path / "b.wav", 2), "あ") == f"{32000 * 2} {32 + 2}"
        assert (tmp_path / "starts").read_text() == "1"
        with pytest.raises(TranscriptOutputLimit):
            await transcriber.transcribe(take(tmp_path / "c.wav"), "ループする")
        with pytest.raises(RuntimeError, match="終了"):
            await transcriber.transcribe(take(tmp_path / "d.wav"), "崩壊")
        assert await transcriber.transcribe(take(tmp_path / "e.wav"), "あ")
        assert (tmp_path / "starts").read_text() == "2"
        # The decoder's 448 positions bound the request for very long lines.
        assert (await transcriber.transcribe(take(tmp_path / "f.wav"), "あ" * 400)).endswith(" 444")
    finally:
        await transcriber.close()
    assert transcriber.process is None


async def test_missing_model_is_not_downloaded_without_permission(settings, tmp_path):
    transcriber = WhisperTranscriber(settings)
    with pytest.raises(WhisperUnavailable, match="ダウンロードを許可"):
        await transcriber.transcribe(take(tmp_path / "a.wav"), "テスト")
    assert not (tmp_path / "starts").exists()
    await transcriber.close()


async def test_reference_wav_bytes_preserve_pcm_and_select_source_language(settings, tmp_path):
    install_model(settings)
    transcriber = WhisperTranscriber(settings)
    try:
        wav = take(tmp_path / "reference.wav")
        assert await transcriber.transcribe(wav.read_bytes(), "Hello there.", language="en-US") == "32000 56"
        assert await transcriber.transcribe(wav.read_bytes(), "你好世界", language="zh-Hant") == "32000 40"
        assert await transcriber.transcribe(wav, "日本語") == "32000 38"
        assert (tmp_path / "languages").read_text().splitlines() == ["en", "zh", "ja"]
        assert (tmp_path / "starts").read_text() == "1"
    finally:
        await transcriber.close()


async def test_timed_transcript_of_a_whole_programme(settings, tmp_path):
    install_model(settings)
    transcriber = WhisperTranscriber(settings)
    try:
        detected = await transcriber.transcribe_timed(tmp_path / "speech.wav", None, timeout=10)
        assert detected == {"language": "en", "segments": [
            {"start": 0.5, "end": 1.5, "text": "speech.wav", "words": []}]}
        assert (await transcriber.transcribe_timed(tmp_path / "speech.wav", "zh-Hant", timeout=10))["language"] == "zh"
        # A programme's word times are one reply line, far beyond asyncio's default 64 KiB line limit.
        long = await transcriber.transcribe_timed(tmp_path / "long-programme.wav", "en", timeout=10)
        assert len(long["segments"][0]["words"]) == 1000
        # Subtitle and take requests share the loaded worker.
        assert await transcriber.transcribe(take(tmp_path / "a.wav"), "あ")
        assert (tmp_path / "starts").read_text() == "1"
    finally:
        await transcriber.close()


async def test_programme_language_pauses_and_review_windows_share_the_worker(settings, tmp_path):
    install_model(settings)
    transcriber = WhisperTranscriber(settings)
    try:
        speech = tmp_path / "speech.wav"
        assert await transcriber.detect_language(speech, [(0, 30), (60, 90)]) == ("zh", {"zh": 2})
        assert await transcriber.pauses(speech, 0.2) == [[1.0, 1.2]]
        assert await transcriber.transcribe_windows(speech, [(0.0, 28.0), (30.0, 59.5)], "zh-Hant") == [
            ("zh 0.0-28.0", True), ("zh 30.0-59.5", False)]
        assert (tmp_path / "starts").read_text() == "1"
    finally:
        await transcriber.close()
