from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from transiroyt.app import create_app
from transiroyt.config import Settings

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("identity,profile", [
    (None, "notary"),
    ("Apple Development: developer", "notary"),
    ("Developer ID Application: developer", None),
])
def test_public_release_requires_distribution_signing(identity, profile):
    with pytest.raises(ValueError):
        script("build_dmg").release_configuration("release", identity, profile)


def test_preview_does_not_require_distribution_signing():
    build = script("build_dmg")
    build.release_configuration("preview", None, None)
    build.release_configuration("release", "Developer ID Application: developer", "notary")


@pytest.mark.parametrize("identity,profile", [
    (None, "notary"),
    ("-", "notary"),
    ("Apple Development: developer", "notary"),
    ("Developer ID Application: developer", None),
])
def test_notarized_preview_requires_distribution_signing_and_credentials(identity, profile):
    with pytest.raises(ValueError):
        script("build_dmg").notarization_configuration("preview", identity, profile, requested=True)


def test_preview_notarization_is_opt_in_and_keeps_public_release_guards():
    build = script("build_dmg")
    identity = "Developer ID Application: developer"
    assert not build.notarization_configuration("preview", None, None)
    assert not build.notarization_configuration("preview", identity, "notary")
    assert build.notarization_configuration("preview", identity, "notary", requested=True)
    assert build.notarization_configuration("release", identity, "notary")
    build.release_configuration("preview", identity, "notary")
    build.release_configuration("release", identity, "notary")
    with pytest.raises(ValueError):
        build.release_configuration("release", "Apple Development: developer", "notary")


@pytest.mark.parametrize("error", [b"The timestamp service is not available.", b"library.so: Input/output error",
                                  b"library.so: A timestamp was expected but was not found.",
                                  b"library.so: Write permissions error."])
def test_transient_signing_failure_is_retried_without_removing_timestamp(monkeypatch, error, tmp_path):
    build = script("build_dmg")
    monkeypatch.setattr(build.time, "sleep", lambda seconds: None)
    calls = []

    def sign(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, args, stderr=error)

    monkeypatch.setattr(build, "run", sign)
    path = tmp_path / "library.so"
    path.write_bytes(b"binary")
    build.sign([path], "Developer ID Application: developer", timestamp=True)
    assert len(calls) == 2
    assert all("--timestamp" in call and "--timestamp=none" not in call for call in calls)


@pytest.mark.parametrize("error,count", [
    (b"The timestamp service is not available.", 3),
    (b"library.so: Input/output error", 3),
    (b"A timestamp was expected but was not found.", 3),
    (b"library.so: Write permissions error.", 3),
    (b"User interaction is not allowed.", 1),
    (b"Invalid signing identity.", 1),
])
def test_signing_failure_retains_diagnostics_and_bounds_retries(monkeypatch, error, count, tmp_path):
    build = script("build_dmg")
    monkeypatch.setattr(build.time, "sleep", lambda seconds: None)
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        raise subprocess.CalledProcessError(1, args, stderr=error)

    monkeypatch.setattr(build, "run", fail)
    path = tmp_path / "library.so"
    path.write_bytes(b"binary")
    with pytest.raises(RuntimeError, match=error.decode()):
        build.sign([path], "Developer ID Application: developer", timestamp=True)
    assert len(calls) == count


@pytest.mark.parametrize("status", ["Accepted", "Invalid", "In Progress"])
def test_notarization_rejects_unaccepted_submission_before_stapling(tmp_path, monkeypatch, status):
    build = script("build_dmg")
    monkeypatch.setattr(build, "BUILD", tmp_path)
    submission = {"id": "test-submission", "status": status}
    monkeypatch.setattr(build, "capture", lambda *args: json.dumps(submission))
    calls = []
    monkeypatch.setattr(build, "run", lambda *args, **kwargs: calls.append(tuple(map(str, args))))
    dmg = tmp_path / "preview.dmg"
    dmg.write_bytes(b"signed disk image")
    if status == "Accepted":
        assert build.notarize_dmg(dmg, "Developer ID Application: developer", "notary") == submission
        assert any(call[:3] == ("xcrun", "stapler", "staple") for call in calls)
        assert any(call[0] == "spctl" for call in calls)
    else:
        with pytest.raises(RuntimeError, match=status):
            build.notarize_dmg(dmg, "Developer ID Application: developer", "notary")
        assert not any("stapler" in call or call[0] == "spctl" for call in calls)
        assert any(call[:3] == ("xcrun", "notarytool", "log") for call in calls)
    assert json.loads((tmp_path / "notary-submission.json").read_text()) == submission


@pytest.mark.parametrize("exit_code", [1, 124])
def test_notary_wait_failure_preserves_submission_and_diagnostics(tmp_path, monkeypatch, exit_code):
    build = script("build_dmg")
    monkeypatch.setattr(build, "BUILD", tmp_path)
    dmg = tmp_path / "release.dmg"
    dmg.write_bytes(b"signed disk image")
    calls = []
    monkeypatch.setattr(build, "run", lambda *args, **kwargs: None)

    def capture(*args):
        calls.append(args)
        if args[2] == "submit":
            assert "--wait" not in args
            return json.dumps({"id": "saved-id", "status": "In Progress"})
        raise subprocess.CalledProcessError(exit_code, args, output="", stderr="Wait timed out")

    monkeypatch.setattr(build, "capture", capture)
    with pytest.raises(RuntimeError, match="Wait timed out"):
        build.notarize_dmg(dmg, "Developer ID Application: developer", "notary", timeout="60s")
    assert [call[2] for call in calls] == ["submit", "wait"]
    assert json.loads((tmp_path / "notary-submission.json").read_text())["id"] == "saved-id"
    artifact = json.loads((tmp_path / "notary-artifact.json").read_text())
    assert artifact == {"id": "saved-id", "sha256": hashlib.sha256(dmg.read_bytes()).hexdigest()}
    assert json.loads((tmp_path / "notary-error.json").read_text())["stderr"] == "Wait timed out"


def test_resume_notary_wait_does_not_resign_or_resubmit(tmp_path, monkeypatch):
    build = script("build_dmg")
    monkeypatch.setattr(build, "BUILD", tmp_path)
    dmg = tmp_path / "release.dmg"
    dmg.write_bytes(b"already submitted disk image")
    (tmp_path / "notary-artifact.json").write_text(json.dumps({
        "id": "saved-id", "sha256": hashlib.sha256(dmg.read_bytes()).hexdigest()}))
    calls = []
    monkeypatch.setattr(build, "capture", lambda *args: calls.append(args) or json.dumps({"id": "saved-id", "status": "Accepted"}))
    monkeypatch.setattr(build, "run", lambda *args, **kwargs: calls.append(args))
    build.notarize_dmg(dmg, "Developer ID Application: developer", "notary", submission_id="saved-id")
    assert calls[0][:3] == ("xcrun", "notarytool", "wait")
    assert not any(call[0] == "codesign" or "submit" in call for call in calls)
    assert any(call[:3] == ("xcrun", "stapler", "staple") for call in calls)
    calls.clear()
    dmg.write_bytes(b"changed")
    with pytest.raises(ValueError, match="ハッシュ"):
        build.notarize_dmg(dmg, "Developer ID Application: developer", "notary", submission_id="saved-id")
    assert not calls


def test_notary_ticket_is_not_attached_if_dmg_changes_during_wait(tmp_path, monkeypatch):
    build = script("build_dmg")
    monkeypatch.setattr(build, "BUILD", tmp_path)
    dmg = tmp_path / "release.dmg"
    dmg.write_bytes(b"submitted image")
    (tmp_path / "notary-artifact.json").write_text(json.dumps({
        "id": "saved-id", "sha256": hashlib.sha256(dmg.read_bytes()).hexdigest()}))
    calls = []
    monkeypatch.setattr(build, "run", lambda *args, **kwargs: calls.append(args))

    def changed(*args):
        dmg.write_bytes(b"different image")
        return json.dumps({"id": "saved-id", "status": "Accepted"})

    monkeypatch.setattr(build, "capture", changed)
    with pytest.raises(ValueError, match="待機中にDMGが変更"):
        build.notarize_dmg(dmg, "Developer ID Application: developer", "notary", submission_id="saved-id")
    assert not calls


def test_preview_build_prefers_apple_development_signing():
    build = script("build_dmg")
    development = "88C044FBFC099CD1431AD132DA6F5BC9A07A8D2A"
    output = "\n".join([
        "  1) E60EE32102423D6209815F4BFE4BECB9ED561774 \"Apple Development: other@example.com (TEAM123456)\"",
        f"  2) {development} \"Apple Development: developer@example.com (TEAM123456)\"",
        "     2 valid identities found",
    ])
    assert build.apple_development_identity(output) == "E60EE32102423D6209815F4BFE4BECB9ED561774"
    assert build.apple_development_identity("     2 valid identities found") is None
    # Sparkle compares build numbers, so each preview build carries a new one.
    version = build.preview_build_version()
    assert len(version) == 15 and version[8] == "."


@pytest.mark.parametrize("channel,authority,accepted", [
    ("preview", "Apple Development: developer (TEAM123456)", True),
    ("preview", "Developer ID Application: developer (TEAM123456)", True),
    ("release", "Developer ID Application: developer (TEAM123456)", True),
    ("release", "Apple Development: developer (TEAM123456)", False),
    ("preview", None, False),
    ("preview", "Apple Distribution: developer (TEAM123456)", False),
])
def test_signature_verification_respects_channel_and_explicit_preview_signer(
        tmp_path, monkeypatch, channel, authority, accepted):
    build = script("build_dmg")
    monkeypatch.setattr(build, "signature_authority", lambda path: authority)
    path = tmp_path / "TransiroYT.app"
    if accepted:
        build.verify_signature(path, channel)
    else:
        with pytest.raises(RuntimeError):
            build.verify_signature(path, channel)


def test_runtime_copy_preserves_relocatable_bytecode_and_rejects_stale_source(tmp_path):
    build = script("build_dmg")
    runtime = tmp_path / "staging/Runtime"
    package = runtime / "library"
    package.mkdir(parents=True)
    source = package / "example.py"
    source.write_text("value = 42\n")
    # Developer caches are still excluded from source copies.
    build.compile_bytecode(sys.executable, runtime, package)
    clean = tmp_path / "source-copy"
    build.copy_tree(runtime, clean)
    assert not list(clean.rglob("*.pyc"))

    installed = tmp_path / "Installed.app/Runtime"
    build.copy_tree(runtime, installed, include_bytecode=True)
    assert list(installed.rglob("*.pyc"))
    code = '''
import importlib.util, sys
from pathlib import Path
source = Path(sys.argv[1])
compiled = []
sys.addaudithook(lambda event, args: compiled.append(event) if event == "compile" else None)
spec = importlib.util.spec_from_file_location("example", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.value == int(sys.argv[2])
assert bool(compiled) == (sys.argv[3] == "recompiled"), compiled
'''
    target = installed / "library/example.py"
    cached = next(installed.rglob("*.pyc"))
    bytecode = cached.read_bytes()
    assert b"TransiroYT/Runtime/library/example.py" in bytecode
    assert str(runtime).encode() not in bytecode
    # -B / read-only app contents must still consume the prebuilt bytecode.
    target.chmod(0o444)
    cached.chmod(0o444)
    for path in [cached.parent, target.parent]:
        path.chmod(0o555)
    try:
        subprocess.run([sys.executable, "-I", "-B", "-c", code, str(target), "42", "cached"],
                       check=True, capture_output=True, text=True)
    finally:
        cached.parent.chmod(0o755)
        target.parent.chmod(0o755)
        target.chmod(0o644)
    # Same length and mtime: a hash-checked cache must still notice the edit.
    original = target.stat()
    target.write_text("value = 43\n")
    os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
    subprocess.run([sys.executable, "-I", "-B", "-c", code, str(target), "43", "recompiled"],
                   check=True, capture_output=True, text=True)
    assert cached.read_bytes() == bytecode


def test_consumer_data_and_configuration_do_not_write_into_the_app(monkeypatch, tmp_path):
    config = tmp_path / "settings.env"
    config.write_text("TRANSIROYT_TTS_STEPS=6\n")
    monkeypatch.setenv("TRANSIROYT_DATA_DIR", str(tmp_path / "Support/.local"))
    monkeypatch.setenv("TRANSIROYT_CONFIG_FILE", str(config))
    monkeypatch.delenv("TRANSIROYT_TTS_STEPS", raising=False)
    settings = Settings.from_env()
    assert settings.data == tmp_path / "Support/.local"
    assert settings.steps == 6
    assert (settings.data / "token").stat().st_mode & 0o777 == 0o600
    assert Settings.from_env().token == settings.token


async def test_consumer_status_does_not_require_downloaded_models(tmp_path):
    runtime, data = tmp_path / "Runtime", tmp_path / "Support"
    runtime.mkdir()
    data.mkdir()
    (runtime / "distribution.json").write_text("{}")
    app = create_app(Settings(root=runtime, data=data, token="consumer-test"))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1",
                                     headers={"Authorization": "Bearer consumer-test"}) as client:
            assert (await client.get("/health")).status_code == 200
            status = (await client.get("/api/status")).json()
            assert status == {"ready": True, "problems": []}
            # The retired realtime API is gone.
            assert (await client.get("/api/sessions")).status_code == 404


def test_distribution_rejects_ffmpeg_without_https(monkeypatch):
    build = script("build_dmg")
    monkeypatch.setattr(build, "capture", lambda *args: "Input:\n file\n pipe\nOutput:\n file\n")
    with pytest.raises(RuntimeError, match="HTTPS"):
        build.check_ffmpeg_protocols(Path("ffmpeg"))
    monkeypatch.setattr(build, "capture", lambda *args: "Input:\n https\n http\n tls\n tcp\n file\n")
    build.check_ffmpeg_protocols(Path("ffmpeg"))


@pytest.mark.parametrize("encoder,decoder", [("h264_videotoolbox", "libdav1d"), ("", "libdav1d"), ("h264_videotoolbox", "")])
def test_distribution_requires_video_conversion(monkeypatch, encoder, decoder):
    build = script("build_dmg")
    monkeypatch.setattr(build, "capture", lambda *args: encoder if args[-1] == "-encoders" else decoder)
    if encoder and decoder:
        build.check_ffmpeg_video(Path("ffmpeg"))
    else:
        with pytest.raises(RuntimeError, match="H.264出力・AV1入力"):
            build.check_ffmpeg_video(Path("ffmpeg"))


def test_consumer_runtime_does_not_accept_system_tools(tmp_path, monkeypatch):
    from transiroyt.config import Settings
    from transiroyt.installation import check_consumer_runtime

    (tmp_path / "distribution.json").write_text("{}")
    monkeypatch.setenv("PATH", "/opt/homebrew/bin:/usr/bin:/bin")
    with pytest.raises(RuntimeError, match="再インストール"):
        check_consumer_runtime(Settings(root=tmp_path))


@pytest.mark.parametrize("missing", [
    "bin/transiro-coreml", ".local/system-transcribe",
])
def test_consumer_runtime_requires_file_dubbing_helpers(tmp_path, missing):
    from transiroyt.installation import check_consumer_runtime

    runtime = tmp_path / "Contents/Resources/Runtime"
    runtime.mkdir(parents=True)
    (runtime / "distribution.json").write_text("{}")
    for relative in ("bin/llama-server", "bin/ffmpeg", "bin/ffprobe", "bin/node",
                     "bin/transiro-coreml", ".local/system-transcribe",
                     "python-api/bin/python3.12", "python-tts/bin/python3.11",
                     "python-indextts/bin/python3.11"):
        path = runtime / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    check_consumer_runtime(Settings(root=runtime))
    (runtime / missing).unlink()
    with pytest.raises(RuntimeError, match="再インストール"):
        check_consumer_runtime(Settings(root=runtime))


def test_distribution_minimum_os_matches_product_requirement(tmp_path):
    assert script("build_dmg").audit(tmp_path) == ([], "27.0")


def test_source_copy_excludes_ffmpeg_environment_dump(tmp_path):
    build = script('build_dmg')
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.mkdir()
    (source / 'FFmpeg-config.log').write_text('PRIVATE_CREDENTIAL=not-for-distribution')
    (source / 'FFmpeg-LGPL-2.1.txt').write_text('license')
    build.copy_tree(source, destination)
    assert not (destination / 'FFmpeg-config.log').exists()
    assert (destination / 'FFmpeg-LGPL-2.1.txt').read_text() == 'license'


def test_slim_runtime_shares_identical_sources_and_preserves_different_versions(tmp_path):
    build = script('build_dmg')
    roots = [tmp_path / f'python-{kind}/lib/python3.11/site-packages' for kind in ('tts', 'indextts')]
    for number, root in enumerate(roots):
        for name in ('sympy', 'networkx', 'mpmath', 'wandb', 'wandb-1.0.dist-info'):
            (root / name).mkdir(parents=True)
            (root / name / '__init__.py').write_text(f'value = {number if name == "networkx" else 42}\n')
        cache = root / 'sympy/__pycache__'
        cache.mkdir()
        (cache / 'old.pyc').write_bytes(bytes([number]))
        (root / 'mpmath/native.so').write_bytes(b'native')
    result = build.slim_python_runtime(tmp_path)
    assert result['shared'] == ['sympy']
    assert (roots[1] / 'sympy').is_symlink()
    assert (roots[1] / 'sympy/__init__.py').read_text() == 'value = 42\n'
    assert (roots[1] / 'sympy').resolve().is_relative_to(tmp_path)
    assert not (roots[1] / 'networkx').is_symlink()
    assert not (roots[1] / 'mpmath').is_symlink()
    assert not any(root.joinpath('wandb').exists() for root in roots)
    assert not any(root.joinpath('wandb-1.0.dist-info').exists() for root in roots)
    assert build.slim_python_runtime(tmp_path) == {'removed': [], 'shared': []}


def test_shared_native_wheels_require_identical_binaries_and_keep_links_internal(tmp_path):
    build = script('build_dmg')
    roots = [tmp_path / f'python-{kind}/lib/python3.11/site-packages' for kind in ('tts', 'indextts')]
    for number, root in enumerate(roots):
        for name in ('torch', 'numpy'):
            (root / name / 'include').mkdir(parents=True)
            (root / name / 'include/build-only.h').write_text(str(number))
            (root / name / '__init__.py').write_text('')
            (root / name / 'runtime.dylib').write_bytes(b'matching' if name == 'torch' else bytes([number]))
    result = build.slim_python_runtime(tmp_path)
    assert result['shared'] == ['torch']
    assert not (roots[0] / 'torch/include').exists()
    assert not (roots[1] / 'torch/include').exists()
    assert (roots[1] / 'torch/runtime.dylib').resolve() == roots[0] / 'torch/runtime.dylib'
    assert (roots[1] / 'torch/runtime.dylib').read_bytes() == b'matching'
    assert not (roots[1] / 'numpy').is_symlink()
    assert (roots[1] / 'numpy/runtime.dylib').read_bytes() == b'\x01'
