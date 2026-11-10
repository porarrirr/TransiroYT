import asyncio
import json
import sys
import wave
from dataclasses import replace
from pathlib import Path

import pytest
from transiroyt import reference_conversion
from transiroyt.config import Settings
from transiroyt.errors import FileDubbingError
from transiroyt.reference_conversion import (
    CONVERSION_TEXT,
    REFERENCE_CONVERSION_VERSION,
    ReferenceConverter,
    converter_python,
    fetch_model,
    model_dir,
    model_ready,
)

# The fake worker answers from the request: "<reference stem>" picks the behaviour.
# "crash" exits without a reply, "hang" never replies, "fail" reports an error;
# every successful take writes a 0.2 s 22.05 kHz WAV, like IndexTTS does.
WORKER = '''import json, os, sys, time, wave
from pathlib import Path

def reply(payload):
    print(json.dumps(payload), flush=True)

root = Path(__file__).resolve().parents[1]
starts = root / "starts"
starts.write_text(str(int(starts.read_text() if starts.exists() else 0) + 1))
if os.environ.get("FAKE_INDEXTTS_STARTUP_SLEEP"):
    time.sleep(float(os.environ["FAKE_INDEXTTS_STARTUP_SLEEP"]))
reply({"ready": True, "device": "mps"})
for line in sys.stdin:
    request = json.loads(line)
    with (root / "requests").open("a") as log:
        log.write(json.dumps(request) + chr(10))
    mode = Path(request["reference"]).stem
    if mode == "crash":
        sys.exit(1)
    if mode == "hang":
        time.sleep(60)
    if mode == "fail":
        reply({"ok": False, "error": "テスト失敗"})
        continue
    with wave.open(request["output"], "wb") as audio:
        audio.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(4410 * 2))
    reply({"ok": True, "seconds": 0.25})
'''


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSIROYT_INDEXTTS_PYTHON", sys.executable)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/indextts_worker.py").write_text(WORKER)
    return Settings(root=tmp_path, data=tmp_path)


def install_model(settings, monkeypatch):
    """The pinned files exist at their expected paths; sizes are 0 so they stay tiny."""
    files = tuple(replace(spec, size=0) for spec in reference_conversion.MODEL_FILES)
    monkeypatch.setattr(reference_conversion, "MODEL_FILES", files)
    checkpoints = model_dir(settings) / "checkpoints"
    for spec in files:
        path = checkpoints / spec.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def reference_file(tmp_path, name):
    path = tmp_path / f"{name}.wav"
    path.write_bytes(b"RIFF")
    return path


def reports_of(calls):
    return [call[0] for call in calls]


def test_the_fixed_text_and_version_are_the_contracts():
    assert REFERENCE_CONVERSION_VERSION == "indextts25-ja-87mora-v1"
    assert CONVERSION_TEXT == (
        "昨日は久しぶりに家族と一緒に近所の公園へ出かけて、満開の桜を眺めながらゆっくりと散歩を楽しみ、"
        "そのあとは駅前のカフェで温かいコーヒーを飲みました。")
    assert model_dir(Settings(root=Path("/app"), data=Path("/data"))) == Path("/data/models/indextts-2.5")


def test_model_ready_checks_every_pinned_file_at_its_expected_size(settings, monkeypatch):
    files = tuple(replace(spec, size=1) for spec in reference_conversion.MODEL_FILES)
    monkeypatch.setattr(reference_conversion, "MODEL_FILES", files)
    checkpoints = model_dir(settings) / "checkpoints"
    assert not model_ready(settings)
    for spec in files:
        path = checkpoints / spec.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    assert model_ready(settings)
    # The flat auxiliary files are part of the model; a wrong size is not ready.
    (checkpoints / "hf_cache/semantic_codec_model.safetensors").write_bytes(b"xx")
    assert not model_ready(settings)
    (checkpoints / "hf_cache/semantic_codec_model.safetensors").write_bytes(b"x")
    (checkpoints / "gpt.pth").unlink()
    assert not model_ready(settings)


def test_converter_python_uses_the_env_then_the_dev_venv_then_the_bundle(settings, monkeypatch):
    configured = settings.root / "custom/bin/python"
    configured.parent.mkdir(parents=True)
    configured.write_text("")
    monkeypatch.setenv("TRANSIROYT_INDEXTTS_PYTHON", str(configured))
    assert converter_python(settings) == configured
    dev = settings.root / "third_party/index-tts/.venv/bin/python"
    dev.parent.mkdir(parents=True)
    dev.write_text("")
    assert converter_python(settings) == configured
    monkeypatch.setenv("TRANSIROYT_INDEXTTS_PYTHON", str(settings.root / "missing/python"))
    assert converter_python(settings) == dev
    bundle = settings.root / "python-indextts/bin/python3.11"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("")
    dev.unlink()
    assert converter_python(settings) == bundle
    bundle.unlink()
    with pytest.raises(FileDubbingError, match="再インストール"):
        converter_python(settings)


async def test_fetching_skips_files_already_at_their_size(settings, monkeypatch):
    spec = reference_conversion.ModelFile("owner/repo", "rev", "weights.bin", 4, "", "weights.bin")
    monkeypatch.setattr(reference_conversion, "MODEL_FILES", (spec,))
    target = model_dir(settings) / "checkpoints/weights.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"abcd")

    def fail(*args, **kwargs):
        raise AssertionError("a complete file must not be downloaded")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)
    calls = []
    await fetch_model(settings, lambda *call: calls.append(call))
    assert reports_of(calls) == ["checking", "complete"]
    assert calls[-1] == ("complete", 4, 4, None)
    assert target.read_bytes() == b"abcd"


async def test_fetching_downloads_missing_files_and_renames_the_auxiliary_ones(settings, monkeypatch):
    files = (
        reference_conversion.ModelFile("owner/a", "r1", "gpt.pth", 4, "", "gpt.pth"),
        reference_conversion.ModelFile("owner/b", "r2", "semantic_codec/model.safetensors", 5,
                                       "hf_cache", "hf_cache/semantic_codec_model.safetensors"),
    )
    monkeypatch.setattr(reference_conversion, "MODEL_FILES", files)
    checkpoints = model_dir(settings) / "checkpoints"
    calls = []

    def fake_download(repo, filename, *, revision, local_dir, tqdm_class):
        calls.append((repo, filename, revision, Path(local_dir)))
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * (4 if filename == "gpt.pth" else 5))
        return str(path)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    reports = []
    await fetch_model(settings, lambda *call: reports.append(call))
    assert calls == [("owner/a", "gpt.pth", "r1", checkpoints),
                     ("owner/b", "semantic_codec/model.safetensors", "r2", checkpoints / "hf_cache")]
    assert (checkpoints / "gpt.pth").read_bytes() == b"xxxx"
    assert (checkpoints / "hf_cache/semantic_codec_model.safetensors").read_bytes() == b"xxxxx"
    assert reports[0][0] == "checking"
    assert ("downloading", 0, 9, "gpt.pth") in reports
    assert reports[-1] == ("complete", 9, 9, None)


async def test_fetching_reuses_a_file_left_at_its_repository_path(settings, monkeypatch):
    spec = reference_conversion.ModelFile("owner/b", "r2", "semantic_codec/model.safetensors", 5,
                                          "hf_cache", "hf_cache/semantic_codec_model.safetensors")
    monkeypatch.setattr(reference_conversion, "MODEL_FILES", (spec,))
    source = model_dir(settings) / "checkpoints/hf_cache/semantic_codec/model.safetensors"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"xxxxx")

    def fail(*args, **kwargs):
        raise AssertionError("an interrupted download must be reused, not repeated")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)
    reports = []
    await fetch_model(settings, lambda *call: reports.append(call))
    assert (model_dir(settings) / "checkpoints/hf_cache/semantic_codec_model.safetensors").read_bytes() == b"xxxxx"
    assert reports[-1] == ("complete", 5, 5, None)


async def test_converts_with_one_resident_worker_into_a_48khz_wav(settings, monkeypatch, tmp_path):
    install_model(settings, monkeypatch)
    converter = ReferenceConverter(settings)
    try:
        reference = reference_file(tmp_path, "ref")
        first = tmp_path / "out/first.wav"
        assert await converter.convert(reference, first) == {"seconds": 0.25}
        with wave.open(str(first), "rb") as audio:
            assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 48000)
            assert abs(audio.getnframes() - 9600) <= 2
        assert not list(first.parent.glob(".*.indextts.wav"))
        assert not list(first.parent.glob(".*.converted.wav"))
        # The second take reuses the loaded model instead of starting another one.
        assert (await converter.convert(reference, tmp_path / "out/second.wav"))["seconds"] == 0.25
        assert (tmp_path / "starts").read_text() == "1"
        request = json.loads((tmp_path / "requests").read_text().splitlines()[-1])
        assert request == {"text": CONVERSION_TEXT, "reference": str(reference),
                           "output": str(tmp_path / "out/.second.wav.indextts.wav"), "lang": "ja"}
        assert "IndexTTS-2.5" in (tmp_path / "indextts.log").read_text()
    finally:
        await converter.close()
    assert converter.process is None


async def test_a_failed_take_keeps_the_worker_for_the_next_one(settings, monkeypatch, tmp_path):
    install_model(settings, monkeypatch)
    converter = ReferenceConverter(settings)
    try:
        with pytest.raises(RuntimeError, match="テスト失敗"):
            await converter.convert(reference_file(tmp_path, "fail"), tmp_path / "fail-out.wav")
        assert converter.process.returncode is None
        assert await converter.convert(reference_file(tmp_path, "ref"), tmp_path / "out.wav")
        assert (tmp_path / "starts").read_text() == "1"
    finally:
        await converter.close()


async def test_a_dead_worker_is_reported_with_the_log_and_started_again(settings, monkeypatch, tmp_path):
    install_model(settings, monkeypatch)
    converter = ReferenceConverter(settings)
    try:
        with pytest.raises(RuntimeError, match="indextts.log"):
            await converter.convert(reference_file(tmp_path, "crash"), tmp_path / "out.wav")
        assert converter.process is None
        assert (await converter.convert(reference_file(tmp_path, "ref"), tmp_path / "out2.wav"))["seconds"] == 0.25
        assert (tmp_path / "starts").read_text() == "2"
    finally:
        await converter.close()


async def test_a_take_that_never_answers_times_out(settings, monkeypatch, tmp_path):
    install_model(settings, monkeypatch)
    monkeypatch.setattr(reference_conversion, "CONVERSION_TIMEOUT", 0.5)
    converter = ReferenceConverter(settings)
    try:
        with pytest.raises(RuntimeError, match="0.5秒"):
            await converter.convert(reference_file(tmp_path, "hang"), tmp_path / "out.wav")
        assert converter.process is None
    finally:
        await converter.close()


async def test_a_startup_that_never_becomes_ready_times_out(settings, monkeypatch, tmp_path):
    install_model(settings, monkeypatch)
    monkeypatch.setattr(reference_conversion, "WORKER_START_TIMEOUT", 0.5)
    monkeypatch.setenv("FAKE_INDEXTTS_STARTUP_SLEEP", "10")
    converter = ReferenceConverter(settings)
    try:
        with pytest.raises(RuntimeError, match="起動が0.5秒"):
            await converter.convert(reference_file(tmp_path, "ref"), tmp_path / "out.wav")
        assert converter.process is None
    finally:
        await converter.close()


async def test_a_worker_that_exits_on_startup_is_reported_with_the_log(settings, monkeypatch, tmp_path):
    install_model(settings, monkeypatch)
    (settings.root / "scripts/indextts_worker.py").write_text(
        'import sys\nprint("boom", file=sys.stderr)\nsys.exit(3)\n')
    converter = ReferenceConverter(settings)
    try:
        with pytest.raises(RuntimeError, match="indextts.log"):
            await converter.convert(reference_file(tmp_path, "ref"), tmp_path / "out.wav")
    finally:
        await converter.close()


async def test_converting_an_unfetched_model_asks_for_the_download(settings, tmp_path):
    converter = ReferenceConverter(settings)
    try:
        with pytest.raises(FileDubbingError, match="ダウンロードを許可"):
            await converter.convert(reference_file(tmp_path, "ref"), tmp_path / "out.wav")
    finally:
        await converter.close()


# The real worker script with a stand-in indextts package: the model's prints must
# go to stderr, stdout must carry only the protocol, and SIGTERM must end it.
WORKER_PACKAGE = '''import json, sys, wave
from pathlib import Path


class IndexTTS2:
    def __init__(self, cfg_path, model_dir, use_bf16, use_qwen_emo, device):
        assert (cfg_path, model_dir, use_bf16, use_qwen_emo) == (
            "checkpoints/config.yaml", "checkpoints", False, False)
        print("model loaded")
        self.device = "mps"

    def infer(self, spk_audio_prompt, text, lang, output_path, verbose):
        print("inference ran")
        if Path(spk_audio_prompt).stem == "fail":
            raise ValueError("fake inference failure")
        with wave.open(output_path, "wb") as audio:
            audio.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
            audio.writeframes(bytes(4410 * 2))
'''


async def test_the_worker_speaks_only_json_on_stdout_and_exits_on_sigterm(tmp_path):
    package = tmp_path / "package/indextts"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "infer_v2_5.py").write_text(WORKER_PACKAGE)
    # The worker frees MPS memory after every conversion.
    (tmp_path / "package/torch.py").write_text(
        "import sys, types\nmps = types.SimpleNamespace(empty_cache=lambda: print('mps cache emptied', file=sys.stderr))\n")
    model = tmp_path / "model"
    (model / "checkpoints").mkdir(parents=True)
    (tmp_path / "ref.wav").write_bytes(b"RIFF")
    (tmp_path / "fail.wav").write_bytes(b"RIFF")
    script = Path(__file__).resolve().parents[1] / "scripts/indextts_worker.py"
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-B", "-u", str(script),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, cwd=model,
        env={"PYTHONPATH": str(tmp_path / "package"), "PATH": "/usr/bin:/bin:/usr/local/bin"})
    try:
        ready = json.loads(await asyncio.wait_for(process.stdout.readline(), 30))
        assert ready == {"ready": True, "device": "mps"}
        for reference, expected in (("ref", {"ok": True}), ("fail", {"ok": False})):
            request = {"text": CONVERSION_TEXT, "reference": str(tmp_path / f"{reference}.wav"),
                       "output": str(tmp_path / f"{reference}-out.wav"), "lang": "ja"}
            process.stdin.write((json.dumps(request) + "\n").encode())
            await process.stdin.drain()
            reply = json.loads(await asyncio.wait_for(process.stdout.readline(), 30))
            assert {key: reply[key] for key in expected} == expected
        process.terminate()
        await asyncio.wait_for(process.wait(), 10)
        assert process.returncode == 0
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    stderr = (await asyncio.wait_for(process.stderr.read(), 30)).decode()
    # The model's own prints went to stderr, not between the protocol lines.
    assert "model loaded" in stderr
    assert "ValueError: fake inference failure" in stderr
