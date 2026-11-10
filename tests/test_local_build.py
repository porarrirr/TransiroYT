import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("local_build_test", ROOT / "scripts/build_local_app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_input_changes_reject_reuse_but_product_code_can_change(tmp_path, monkeypatch):
    build = load(monkeypatch)
    import build_dmg
    monkeypatch.setattr(build_dmg, "capture", lambda *args: "apple/SystemTranscribe.swift\napple/Inference/Package.swift\ndistribution/Runtime.entitlements")
    for name in ("uv.lock", "backend/app.py", "apple/Desktop/Sources/View.swift", "apple/SystemTranscribe.swift",
                 "distribution/Runtime.entitlements", "apple/Inference/Package.swift"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("before")
    app = tmp_path / "Preview.app"
    runtime = app / "Contents/Resources/Runtime"
    runtime.mkdir(parents=True)
    (runtime / "distribution.json").write_text(json.dumps({"channel": "preview"}))
    original = build.runtime_inputs(tmp_path)
    (runtime / "local-build-inputs.json").write_text(json.dumps(original))
    (tmp_path / "backend/app.py").write_text("after")
    (tmp_path / "apple/Desktop/Sources/View.swift").write_text("after")
    assert build.validate_base(app, tmp_path) == runtime
    for name in ("uv.lock", "apple/SystemTranscribe.swift", "distribution/Runtime.entitlements", "apple/Inference/Package.swift"):
        path = tmp_path / name
        path.write_text("changed")
        with pytest.raises(ValueError, match="構成記録"):
            build.validate_base(app, tmp_path)
        path.write_text("before")
    (runtime / "local-build-inputs.json").unlink()
    with pytest.raises(ValueError, match="構成記録"):
        build.validate_base(app, tmp_path)
    (runtime / "distribution.json").write_text(json.dumps({"channel": "release"}))
    with pytest.raises(ValueError, match="preview"):
        build.validate_base(app, tmp_path)


def test_local_source_sync_removes_retired_code_and_preserves_native_runtime(tmp_path, monkeypatch):
    build = load(monkeypatch)
    root, runtime = tmp_path / "source", tmp_path / "Runtime"
    for name, content in [("backend/package/new.py", "new"), ("backend/package/__pycache__/bad.pyc", "cache"),
                          ("guide/index.html", "guide")]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (root / "scripts").mkdir()
    for name in ("speaker_embedding.py", "speaker_activity.py", "separate_vocals.py", "whisper_worker.py", "indextts_worker.py"):
        (root / "scripts" / name).write_text("worker")
    for name in ("backend/old.py", "help/old.html", "scripts/old.py", "python-tts/native.so", "licenses/NOTICE"):
        path = runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserve")
    for name in ("distribution/licenses/models/PLaMo/LICENSE", "distribution/licenses/models.json", "distribution/fallback-models.lock.json", "docs/licenses.md"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model terms")
    (runtime / "licenses/supplemental/models").mkdir(parents=True)
    build.copy_sources(runtime, root)
    assert (runtime / "licenses/supplemental/models/PLaMo/LICENSE").read_text() == "model terms"
    assert (runtime / "licenses/supplemental/models.json").read_text() == "model terms"
    assert (runtime / "backend/package/new.py").read_text() == "new"
    assert not (runtime / "backend/old.py").exists()
    assert not list((runtime / "backend").rglob("*.pyc"))
    assert not (runtime / "help/old.html").exists()
    assert not (runtime / "scripts/old.py").exists()
    assert (runtime / "scripts/whisper_worker.py").read_text() == "worker"
    assert (runtime / "python-tts/native.so").read_bytes() == b"preserve"
    assert (runtime / "licenses/NOTICE").read_bytes() == b"preserve"


def test_installer_keeps_running_export_and_installed_bundle(tmp_path, monkeypatch):
    build = load(monkeypatch)
    installed, candidate = tmp_path / "Installed.app", tmp_path / "Candidate.app"
    installed.mkdir()
    candidate.mkdir()
    (installed / "original").write_text("keep")
    support = tmp_path / "Library/Application Support/TransiroYT/.local"
    support.mkdir(parents=True)
    (support / "token").write_text("test-token")
    monkeypatch.setattr(build.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(build, "app_processes", lambda app: [123])
    def request(client, request, **kwargs):
        import httpx
        return httpx.Response(200, json={"instance_id": "old"} if request.url.path == "/health"
                              else {"job": {"state": "running"}}, request=request)
    monkeypatch.setattr(build.httpx.Client, "send", request)
    monkeypatch.setattr(build, "run", lambda *args: pytest.fail("must not terminate or launch an app"))
    with pytest.raises(ValueError, match="吹き替え実行中"):
        build.install_app(candidate, installed, tmp_path)
    assert (installed / "original").read_text() == "keep"
    assert candidate.is_dir()


def test_unchanged_source_preserves_compiled_cache(tmp_path, monkeypatch):
    build = load(monkeypatch)
    source, destination = tmp_path / "source.py", tmp_path / "backend"
    source.write_text("value = 42")
    destination.mkdir()
    (destination / "source.py").write_text("value = 42")
    cached = destination / "__pycache__/source.pyc"
    cached.parent.mkdir()
    cached.write_bytes(b"compiled")
    assert not build.sync_files({Path("source.py"): source}, destination)
    assert cached.read_bytes() == b"compiled"
    source.write_text("value = 43")
    assert build.sync_files({Path("source.py"): source}, destination)
    assert not cached.exists()
    assert (destination / "source.py").read_text() == "value = 43"


@pytest.mark.parametrize("options", [
    ["--channel", "release", "--reuse-runtime"],
    ["--channel", "release", "--app-only"],
    ["--channel", "preview", "--app-only", "--notarize"],
])
def test_local_build_options_cannot_skip_release_requirements(options, tmp_path):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(ROOT / "scripts/build_dmg.py"), *options,
                             "--build-dir", str(tmp_path / "build"), "--output-dir", str(tmp_path / "dist")],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert not (tmp_path / "build").exists()
    assert not (tmp_path / "dist").exists()
