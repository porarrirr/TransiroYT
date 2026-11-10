"""Build a relocatable Mac app, corresponding sources and a DMG.

Public releases fail closed without Developer ID and notarization.
Preview builds are clearly named and never described as notarized/public releases.
An explicitly requested notarized preview remains a non-public preparation build.
Run with the project's Python, after scripts/setup.sh --with-tts on the build Mac.
"""
from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import time
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.5.2"
DMG_FORMAT = "ULMO"
BUILD = ROOT / ".local/distribution-build"
OUTPUT = ROOT / "dist"
MACHO = {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"}


def run(*args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def capture(*args):
    return run(*args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip()


def apple_development_identity(output):
    """Return the first Apple Development signing identity in `security` output."""
    for line in output.splitlines():
        match = re.search(r'\b([0-9A-F]{40})\b\s+"Apple Development:', line)
        if match:
            return match.group(1)
    return None


def preview_signing_identity():
    try:
        return apple_development_identity(capture("security", "find-identity", "-p", "codesigning", "-v"))
    except (OSError, subprocess.CalledProcessError):
        return None


def preview_build_version():
    # Sparkle compares CFBundleVersion and update tags include it, so each build
    # gets a new, increasing number.
    # CFBundleVersion allows up to three period-separated 32-bit integers.
    return datetime.datetime.now().strftime("%Y%m%d.%H%M%S")


def release_configuration(channel, identity, profile):
    if channel == "release":
        if not identity or not identity.startswith("Developer ID Application:"):
            raise ValueError("公開版にはDeveloper ID Application証明書が必要です。--channel previewで検証版を作れます。")
        if not profile:
            raise ValueError("公開版には--notary-profileが必要です。")


def notarization_configuration(channel, identity, profile, *, requested=False):
    notarize = channel == "release" or requested
    if notarize:
        if not identity or not identity.startswith("Developer ID Application:"):
            raise ValueError("公証にはDeveloper ID Application証明書を明示指定してください。")
        if not profile:
            raise ValueError("公証には--notary-profileが必要です。")
    return notarize


def ignored(directory, names, *, include_bytecode=False):
    excluded = {".git", ".venv", ".DS_Store", "runtime-config.js", ".pytest_cache", "FFmpeg-config.log"}
    suffixes = (".a",) if include_bytecode else (".pyc", ".pyo", ".a")
    if not include_bytecode:
        excluded.add("__pycache__")
    return {name for name in names if name in excluded or name.endswith(suffixes)}


def copy_tree(source, target, *, include_bytecode=False):
    # copytree(dirs_exist_ok=True) replaces files but cannot replace symlinks.
    for path in source.rglob("*"):
        if path.is_symlink():
            destination = target / path.relative_to(source)
            if destination.is_symlink() or destination.is_file():
                destination.unlink()
    shutil.copytree(source, target, symlinks=True,
                    ignore=functools.partial(ignored, include_bytecode=include_bytecode),
                    dirs_exist_ok=True)


def compile_bytecode(python, runtime, *paths):
    # Use each bundled interpreter's own bytecode format, before signing. Hash
    # validation survives copying/moving the app and rejects stale source code.
    # Keep source files, and omit developer/staging paths from code objects.
    run(python, "-I", "-B", "-m", "compileall", "-q", "-j", "1",
        "--invalidation-mode", "checked-hash", "-s", runtime, "-p", "TransiroYT/Runtime",
        *paths)


def check_ffmpeg_protocols(ffmpeg):
    protocols = capture(ffmpeg, "-hide_banner", "-protocols").split()
    if not all(protocol in protocols for protocol in ("http", "https", "tls", "tcp")):
        raise RuntimeError("配布用FFmpegにHTTPS入力がありません。build_consumer_tools.shで再ビルドしてください。")


def check_ffmpeg_video(ffmpeg):
    encoders = capture(ffmpeg, "-hide_banner", "-encoders").split()
    decoders = capture(ffmpeg, "-hide_banner", "-decoders").split()
    if "h264_videotoolbox" not in encoders or "libdav1d" not in decoders:
        raise RuntimeError("配布用FFmpegにH.264出力・AV1入力がありません。build_consumer_tools.shで再ビルドしてください。")


# IndexTTS-2.5's declared dependency set includes packages that its inference path
# never imports but that carry GPL code: the opencv-python wheel embeds a GPL FFmpeg
# (libx264/libx265), Distance is GPL (only required by the unused g2p_en), and
# textstat/pyphen serve an unused helper module. Keep them out of the MIT bundle.
GPL_ONLY_PACKAGES = (
    "cv2", "opencv_python-*.dist-info",
    "distance", "distance-*.dist-info",
    "g2p_en", "g2p_en-*.dist-info",
    "textstat", "textstat-*.dist-info",
    "pyphen", "pyphen-*.dist-info",
)


# The desktop runs Core ML for synthesis. These packages belong to the upstream
# training/web UI, and are not part of the shipped worker inference paths.
UNUSED_INFERENCE_PACKAGES = (
    "gradio", "gradio-*.dist-info", "gradio_client", "gradio_client-*.dist-info",
    "wandb", "wandb-*.dist-info", "datasets", "datasets-*.dist-info", "pyarrow", "pyarrow-*.dist-info",
)


# These native stacks are pinned to the same wheels in both Python 3.11
# environments. Sharing happens before relocation and signing, so each loader
# path is rewritten for the one physical copy.
SHARED_NATIVE_PACKAGES = ("torch", "torchaudio", "torio", "numpy", "scipy", "llvmlite", "numba")


def slim_python_runtime(runtime):
    """Prune build-only files and share identical packages within the app."""
    environments = [runtime / f"python-{kind}/lib/python3.11/site-packages" for kind in ("tts", "indextts")]
    removed, shared = [], []
    for packages in environments:
        for pattern in UNUSED_INFERENCE_PACKAGES:
            for path in packages.glob(pattern):
                removed.append(str(path.relative_to(runtime)))
                shutil.rmtree(path)
        for name in ("torch/include", "torch/share/cmake", "mpmath/tests", "pip"):
            path = packages / name
            if path.is_dir():
                removed.append(str(path.relative_to(runtime)))
                shutil.rmtree(path)
        for path in packages.glob("pip-*.dist-info"):
            removed.append(str(path.relative_to(runtime)))
            shutil.rmtree(path)
    source, destination = environments
    for name in ("sympy", "mpmath", "networkx", *SHARED_NATIVE_PACKAGES):
        original, duplicate = source / name, destination / name
        if duplicate.is_symlink() or not original.is_dir() or not duplicate.is_dir():
            continue
        # Bytecode is rebuilt for the final bundle. Source and data (including
        # native binaries) must match exactly before sharing.
        trees = []
        for package in (original, duplicate):
            trees.append({str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in package.rglob("*") if path.is_file() and "__pycache__" not in path.parts})
        if trees[0] != trees[1] or (name not in SHARED_NATIVE_PACKAGES and any(
                Path(path).suffix in {".so", ".dylib"} for path in trees[0])):
            continue
        shutil.rmtree(duplicate)
        duplicate.symlink_to(os.path.relpath(original, duplicate.parent), target_is_directory=True)
        shared.append(name)
    return {"removed": removed, "shared": shared}


def bundle_python(environment, target, *, prune_gpl=False):
    from distribution_licenses import remove_optional_metadata_editor, validate_native_inputs

    python = environment / "bin/python"
    base = Path(capture(python, "-c", "import sys; print(sys.base_prefix)"))
    minor = capture(python, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    copy_tree(base, target)
    packages = target / f"lib/python{minor}/site-packages"
    copy_tree(environment / f"lib/python{minor}/site-packages", packages)
    remove_optional_metadata_editor(packages)
    validate_native_inputs(ROOT, packages)
    if prune_gpl:
        for pattern in GPL_ONLY_PACKAGES:
            for path in packages.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
    # Compiling extensions or installing packages inside a signed app is unsupported.
    # Retain Python headers: patchlevel.h identifies the matching CPython notices.
    bootstrap = target / f"lib/python{minor}/ensurepip"
    if bootstrap.exists():
        shutil.rmtree(bootstrap)
    # Absolute editable installs and developer provenance must not enter a release.
    for path in packages.glob("*.pth"):
        if "import" not in path.read_text() and any(line.startswith("/") for line in path.read_text().splitlines()):
            path.unlink()
    for path in packages.glob("*.dist-info/direct_url.json"):
        path.unlink()
    for path in [*(packages / name for name in ("pytest", "_pytest", "ruff", "_virtualenv.py", "_virtualenv.pth", "pip")),
                 *packages.glob("pip-*.dist-info")]:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    # Only FFmpeg 8 is shipped. Other optional torchcodec ABI variants cannot load
    # without their corresponding (unshipped) FFmpeg shared libraries.
    codec = packages / "torchcodec"
    if codec.exists():
        for path in codec.glob("libtorchcodec*"):
            if re.search(r"[4-7]\.(?:dylib|so)$", path.name):
                path.unlink()
    # TorchAudio's optional FFmpeg 4–6 extensions cannot use the bundled FFmpeg 8.
    for path in (packages / "torio/lib").glob("*torio_ffmpeg*.so"):
        path.unlink()
    # The optional SoX backend requires libsox, which is not bundled.
    for path in (packages / "torchaudio/lib").glob("*torchaudio_sox.so"):
        path.unlink()


def is_macho(path):
    if path.is_symlink() or not path.is_file():
        return False
    with path.open("rb") as stream:
        return stream.read(4) in MACHO


@functools.lru_cache(maxsize=4096)
def _load_commands(path, mtime, size):
    return capture("otool", "-l", path)


def load_commands(path):
    info = path.stat()
    return _load_commands(path, info.st_mtime_ns, info.st_size)


def dependencies(path):
    # otool -L also prints LC_ID_DYLIB (the library's own install ID), which
    # must not be treated as an unresolved dependency for Python extension modules.
    return re.findall(r"cmd LC_(?:LOAD_DYLIB|LOAD_WEAK_DYLIB|REEXPORT_DYLIB|LOAD_UPWARD_DYLIB|LAZY_LOAD_DYLIB)\s+cmdsize \d+\s+name ([^\n]+?) \(offset", load_commands(path))


def local_rpath(path, dependency):
    if not dependency.startswith("@rpath/"):
        return None
    commands = load_commands(path)
    for raw in re.findall(r"cmd LC_RPATH\s+cmdsize \d+\s+path ([^\n]+?) \(offset", commands):
        if raw.startswith("@loader_path"):
            candidate = Path(raw.replace("@loader_path", str(path.parent), 1)) / dependency.removeprefix("@rpath/")
            if candidate.exists():
                return candidate
    return None


def relocate(runtime):
    """Resolve all non-system dylibs into the bundle, including torch/NumPy wheels."""
    binaries = [path for path in runtime.rglob("*") if is_macho(path)]
    tools = ROOT / ".local/distribution-build/tools"
    known = {path.name: path for path in binaries if path.suffix in {".dylib", ".so"}}
    per_environment = {(path.relative_to(runtime).parts[0], path.name): path for path in binaries
                       if path.suffix in {".dylib", ".so"}}
    for path in runtime.rglob("*.dylib"):
        if path.is_symlink() and path.exists():
            known[path.name] = path.resolve()
    for binary in binaries:
        binary.chmod(binary.stat().st_mode | 0o200)
        for dependency in dependencies(binary):
            if dependency.startswith(("/usr/lib/", "/System/Library/")):
                continue
            if local_rpath(binary, dependency):
                continue
            if dependency.startswith("@loader_path/"):
                resolved = binary.parent / dependency.removeprefix("@loader_path/")
                if resolved.exists():
                    continue
            name = Path(dependency).name
            sibling = binary.parent / name
            environment = binary.relative_to(runtime).parts[0]
            target = sibling if sibling.is_file() else per_environment.get((environment, name), known.get(name))
            if target is None:
                for candidate in [tools / "lib" / name, Path(dependency)]:
                    if candidate.is_file():
                        target = runtime / "lib" / name
                        target.parent.mkdir(exist_ok=True)
                        shutil.copy2(candidate.resolve(), target)
                        known[name] = target
                        binaries.append(target)
                        break
            if target is None:
                # Python uses the OS-provided Swift runtime on recent macOS.
                if dependency.startswith("@rpath/libswift") and Path("/usr/lib/swift", name).exists():
                    continue
                raise ValueError(f"同梱できない依存ライブラリ: {binary.relative_to(runtime)} -> {dependency}")
            relative = "@loader_path/" + os.path.relpath(target, binary.parent)
            run("install_name_tool", "-change", dependency, relative, binary,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return binaries


def audit(runtime):
    binaries = []
    # Keep the currently validated deployment target; older macOS is not tested.
    minimum = (27, 0)
    for path in runtime.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(runtime.resolve()):
            raise ValueError(f"配布物の外を参照するリンク: {path.relative_to(runtime)}")
        if not is_macho(path):
            continue
        binaries.append(path)
        for dependency in dependencies(path):
            if dependency.startswith(("/opt/", "/Users/", "/usr/local/")):
                raise ValueError(f"外部の開発環境を参照しています: {dependency}")
            if dependency.startswith("@loader_path/") and not (path.parent / dependency.removeprefix("@loader_path/")).exists():
                raise ValueError(f"存在しない同梱ライブラリ: {path} -> {dependency}")
            if dependency.startswith("@rpath/") and not local_rpath(path, dependency):
                raise ValueError(f"解決できないrpath: {path} -> {dependency}")
        for value in re.findall(r"\bminos (\d+(?:\.\d+)+)", load_commands(path)):
            version = tuple(int(part) for part in value.split(".")[:2])
            minimum = max(minimum, version)
    return binaries, ".".join(str(part) for part in minimum)


def inventory(runtime):
    result = {}
    for kind in ["api", "tts", "indextts"]:
        packages = []
        for path in (runtime / f"python-{kind}").rglob("*.dist-info/METADATA"):
            lines = path.read_text(errors="replace").splitlines()
            values = {key: next((line[len(key) + 2:] for line in lines if line.startswith(key + ": ")), "") for key in ["Name", "Version", "License-Expression", "License"]}
            packages.append(values)
        result[kind] = sorted(packages, key=lambda value: value["Name"].lower())
    (runtime / "licenses/dependency-inventory.json").write_text(json.dumps(result, indent=2) + "\n")


def sign(paths, identity, *, timestamp):
    paths = list(paths)
    for index, path in enumerate(paths):
        args = ["codesign", "--force", "--sign", identity]
        if identity != "-":
            args += ["--timestamp" if timestamp else "--timestamp=none",
                     "--options", "runtime", "--entitlements", str(ROOT / "distribution/Runtime.entitlements")]
        for attempt in range(3):
            try:
                run(*args, path, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                break
            except subprocess.CalledProcessError as exc:
                error = exc.stderr or b""
                error = error.decode(errors="replace") if isinstance(error, bytes) else error
                transient = (timestamp and "timestamp" in error.lower() and any(
                    text in error.lower() for text in ("service is not available", "timed out", "timeout",
                                                       "a timestamp was expected but was not found")))
                transient = transient or error.strip().endswith(": Input/output error")
                transient = transient or (error.strip().endswith(": Write permissions error.")
                                           and path.is_file() and os.access(path, os.W_OK))
                if not transient or attempt == 2:
                    raise RuntimeError(f"署名に失敗しました: {path}\n{error.strip()}") from exc
                print(f"署名を再試行します ({attempt + 1}/2): {path.name}\n{error.strip()}", flush=True)
                time.sleep(attempt + 1)
        if len(paths) > 1 and ((index + 1) % 25 == 0 or index + 1 == len(paths)):
            print(f"署名: {index + 1}/{len(paths)}", flush=True)


def signature_authority(path):
    result = subprocess.run(["codesign", "-dv", "--verbose=2", str(path)],
                            capture_output=True, text=True, check=False)
    for line in result.stderr.splitlines():
        if line.startswith("Authority="):
            return line.removeprefix("Authority=")
    return None


def verify_signature(path, channel):
    prefixes = (("Developer ID Application:",) if channel == "release" else
                ("Apple Development:", "Developer ID Application:"))
    authority = signature_authority(path)
    if not authority or not authority.startswith(prefixes):
        raise RuntimeError(f"{path} is not signed with {' or '.join(prefixes)}: {authority}")


def notary_response(*args):
    try:
        return json.loads(capture("xcrun", "notarytool", *args))
    except subprocess.CalledProcessError as exc:
        def decoded(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

        diagnostics = {"operation": args[0], "exit_code": exc.returncode,
                       "stdout": decoded(exc.stdout), "stderr": decoded(exc.stderr)}
        (BUILD / "notary-error.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        raise RuntimeError(f"公証の{args[0]}が終了しました。提出済みの処理はApple側で継続する場合があります。"
                           f"{BUILD / 'notary-error.json'} を確認してください。\n{diagnostics['stderr']}") from exc


def notarize_dmg(dmg, identity, profile, *, submission_id=None, timeout="20m"):
    with dmg.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    artifact_path = BUILD / "notary-artifact.json"
    if submission_id:
        artifact = json.loads(artifact_path.read_text())
        if artifact["id"] != submission_id or artifact["sha256"] != digest:
            raise ValueError("提出IDまたはDMGのハッシュが一致しないため、公証待機を再開できません。")
    else:
        run("codesign", "--force", "--sign", identity, "--timestamp", dmg)
        submission = notary_response("submit", dmg, "--keychain-profile", profile, "--no-s3-acceleration", "--output-format", "json")
        # Persist the receipt before waiting: a disconnected waiter must not
        # lose a successful submission or upload the same artifact again.
        (BUILD / "notary-submission.json").write_text(json.dumps(submission, indent=2) + "\n")
        submission_id = submission["id"]
        with dmg.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        artifact_path.write_text(json.dumps({"id": submission_id, "sha256": digest}, indent=2) + "\n")
    submission = notary_response("wait", submission_id, "--keychain-profile", profile,
                                 "--timeout", timeout, "--output-format", "json")
    (BUILD / "notary-submission.json").write_text(json.dumps(submission, indent=2) + "\n")
    if submission.get("status") != "Accepted":
        if submission.get("id"):
            run("xcrun", "notarytool", "log", submission["id"], "--keychain-profile", profile,
                BUILD / "notary-log.json")
        raise RuntimeError(f"公証が承認されませんでした: {submission.get('status')}。{BUILD} の記録を確認してください。")
    with dmg.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
            raise ValueError("公証待機中にDMGが変更されたため、チケットを添付できません。")
    run("xcrun", "stapler", "staple", dmg)
    run("xcrun", "stapler", "validate", dmg)
    run("spctl", "--assess", "--type", "open", "--context", "context:primary-signature", "--verbose=2", dmg)
    return submission


def runtime_inputs(root=ROOT):
    """Inputs requiring fresh native/Python dependencies rather than a code update."""
    names = {"uv.lock", "pyproject.toml", "requirements-tts-mac.txt",
             "apple/Desktop/Package.swift", "apple/Desktop/Package.resolved",
             "scripts/build_dmg.py", "scripts/build_consumer_tools.sh", "scripts/distribution_licenses.py",
             "scripts/build_coreml.sh", "scripts/build_system_transcription.sh",
             "scripts/build_sparkle.py", "scripts/build_desktop.sh", "scripts/setup.sh",
             "scripts/build_native_audio.py", "scripts/native_audio/lib.rs"}
    names.update(capture("git", "-C", root, "ls-files", "--cached", "--others", "--exclude-standard", "--",
                         "distribution", "apple/Inference",
                         "apple/SystemTranscribe.swift", "apple/Mobile/Notices").splitlines())
    for project in ("Irodori-TTS", "index-tts"):
        names.update(f"third_party/{project}/{name}" for name in ("uv.lock", "pyproject.toml", "requirements.txt"))
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() if (root / name).is_file() else None
            for name in sorted(names)}


def main():
    global BUILD, OUTPUT
    from build_sparkle import sign_sparkle
    from distribution_licenses import prepare_licenses
    from update_feed import configure_app, create_feed, update_configuration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", choices=["preview", "release"], default="preview")
    parser.add_argument("--sign-identity", default=os.getenv("TRANSIROYT_SIGN_IDENTITY"))
    parser.add_argument("--notary-profile", default=os.getenv("TRANSIROYT_NOTARY_PROFILE"))
    parser.add_argument("--notarize", action="store_true", help="Notarize a Developer ID-signed preview; does not make it a public release")
    parser.add_argument("--update-feed", action="store_true", help="Sign an update feed for a preview; release builds always sign it")
    parser.add_argument("--app-only", action="store_true", help="Build a signed preview app without a DMG or source archive")
    parser.add_argument("--reuse-runtime", action="store_true", help="Reuse a locally assembled runtime for UI iteration")
    parser.add_argument("--build-dir", type=Path, default=BUILD)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    BUILD, OUTPUT = args.build_dir.resolve(), args.output_dir.resolve()
    if args.channel == "release" and args.reuse_runtime:
        parser.error("正式公開ではランタイムを再利用できません。")
    if args.app_only and (args.channel != "preview" or args.notarize or args.update_feed):
        parser.error("--app-onlyはローカルpreview専用です。")
    release_configuration(args.channel, args.sign_identity, args.notary_profile)
    notarize = notarization_configuration(args.channel, args.sign_identity, args.notary_profile,
                                         requested=args.notarize)
    identity = args.sign_identity
    if args.channel == "preview" and identity is None:
        identity = preview_signing_identity()
        if identity:
            print(f"署名: Apple Development ({identity})", flush=True)
        else:
            print("警告: Apple Development証明書がないためad-hoc署名になります。", flush=True)
    identity = identity or "-"
    if sys.platform != "darwin" or os.uname().machine != "arm64":
        parser.error("Apple Silicon Macでビルドしてください。")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = BUILD / "TransiroYT.app"
    # Keep the partial runtime outside the launchable .app path. Clean copies
    # directly into that path have failed with EPERM midway through assembly.
    app = BUILD / "app-staging"
    runtime = app / "Contents/Resources/Runtime"
    print("1/6 配布用ツールとSwiftアプリをビルド", flush=True)
    run(ROOT / "scripts/build_consumer_tools.sh")
    run(ROOT / "scripts/build_desktop.sh")
    app.parent.mkdir(parents=True, exist_ok=True)
    if app.exists():
        shutil.rmtree(app)
    copy_tree(ROOT / ".local/TransiroYT.app", app)
    # Reuse only the runtime; the rest of a previous bundle may hold removed parts.
    if args.reuse_runtime and (destination / "Contents/Resources/Runtime").exists():
        (destination / "Contents/Resources/Runtime").rename(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    print("2/6 Pythonと音声ランタイムを同梱", flush=True)
    if not args.reuse_runtime or not (runtime / "python-tts").exists():
        bundle_python(ROOT / ".venv", runtime / "python-api")
        bundle_python(ROOT / ".local/file-workers", runtime / "python-tts")
    if not args.reuse_runtime or not (runtime / "python-indextts").exists():
        bundle_python(ROOT / "third_party/index-tts/.venv", runtime / "python-indextts", prune_gpl=True)
    slimming = slim_python_runtime(runtime)
    (runtime / "runtime-slimming.json").write_text(json.dumps(slimming, indent=2) + "\n")
    copy_tree(ROOT / "backend", runtime / "backend")
    (runtime / ".local").mkdir(exist_ok=True)
    shutil.copy2(ROOT / ".local/system-transcribe", runtime / ".local/system-transcribe")
    (runtime / "scripts").mkdir(exist_ok=True)
    workers = ["speaker_embedding.py", "speaker_activity.py", "separate_vocals.py", "whisper_worker.py", "indextts_worker.py"]
    # A reused runtime may still hold retired workers, helpers and browser integration.
    for path in (runtime / "scripts").glob("*.py"):
        if path.name not in workers:
            path.unlink()
    (runtime / ".local/system-translate").unlink(missing_ok=True)
    shutil.rmtree(runtime / "extension", ignore_errors=True)
    shutil.rmtree(runtime / "third_party", ignore_errors=True)
    for name in workers:
        shutil.copy2(ROOT / "scripts" / name, runtime / "scripts" / name)
    upstream = ROOT / "third_party/Irodori-TTS"
    for folder in ["bin", "lib", "licenses"]:
        copy_tree(ROOT / ".local/distribution-build/tools" / folder, runtime / folder)
    # configure's raw log dumps the builder's entire environment. Ship only
    # FFmpeg's reproducible flags, with the machine-specific source root removed.
    (runtime / "licenses/FFmpeg-build-configuration.txt").write_text(
        capture(runtime / "bin/ffmpeg", "-hide_banner", "-buildconf").replace(str(ROOT), "<source-root>") + "\n")
    shutil.copy2(ROOT / ".local/bin/transiro-coreml", runtime / "bin/transiro-coreml")
    copy_tree(ROOT / "apple/Mobile/Notices", runtime / "licenses/mobile-inference")
    shutil.copy2(upstream / "LICENSE", runtime / "licenses/Irodori-TTS-MIT.txt")
    shutil.copy2(ROOT / "third_party/index-tts/LICENSE", runtime / "licenses/IndexTTS-bilibili-LICENSE.txt")
    shutil.copy2(ROOT / "distribution/THIRD_PARTY_NOTICES.txt", runtime / "licenses/THIRD_PARTY_NOTICES.txt")
    shutil.copy2(ROOT / "distribution/models.lock.json", runtime / "models.lock.json")
    shutil.copy2(ROOT / "distribution/fallback-models.lock.json", runtime / "fallback-models.lock.json")
    copy_tree(ROOT / "guide", runtime / "help")
    for path in runtime.rglob("__pycache__"):
        shutil.rmtree(path)
    for name in ("ffmpeg", "ffprobe", "node", "llama-server", "transiro-coreml"):
        if not os.access(runtime / "bin" / name, os.X_OK):
            raise ValueError(f"必要な実行ファイルが同梱されていません: {name}")
    inventory(runtime)
    print("3/6 依存ライブラリを再配置・監査", flush=True)
    relocate(runtime)
    binaries, minimum = audit(runtime)
    check_ffmpeg_protocols(runtime / "bin/ffmpeg")
    check_ffmpeg_video(runtime / "bin/ffmpeg")
    help_path = runtime / "help/index.html"
    help_text = re.sub(r"macOS \d+(?:\.\d+)?以降", f"macOS {minimum}以降", help_path.read_text())
    help_path.write_text(help_text)
    distribution = {"version": VERSION, "channel": args.channel, "minimum_macos": minimum,
                    "architecture": "arm64", "macho_files": len(binaries)}
    (runtime / "distribution.json").write_text(json.dumps(distribution, indent=2) + "\n")
    plist_path = app / "Contents/Info.plist"
    with plist_path.open("rb") as stream:
        info = plistlib.load(stream)
    info.pop("TransiroYTProjectPath", None)
    build_version = preview_build_version()
    info.update(CFBundleIdentifier="com.porarri.transiroyt.desktop", CFBundleShortVersionString=VERSION,
                CFBundleVersion=build_version, LSMinimumSystemVersion=minimum,
                CFBundleIconFile="AppIcon")
    configure_app(info, args.channel)
    with plist_path.open("wb") as stream:
        plistlib.dump(info, stream)
    corresponding_sources = prepare_licenses(ROOT, runtime, BUILD)
    (runtime / "local-build-inputs.json").write_text(json.dumps(runtime_inputs(), sort_keys=True) + "\n")
    print("Pythonの初回コンパイルをビルド時に実行", flush=True)
    compile_bytecode(runtime / "python-api/bin/python3.12", runtime,
                     runtime / "python-api/lib", runtime / "backend")
    compile_bytecode(runtime / "python-tts/bin/python3.11", runtime,
                     runtime / "python-tts/lib", runtime / "scripts")
    compile_bytecode(runtime / "python-indextts/bin/python3.11", runtime, runtime / "python-indextts/lib")
    # Resolve all runtime dependencies before exposing the assembled app at its
    # launch path; sign that final location, never overwrite signed contents.
    if destination.exists():
        shutil.rmtree(destination)
    app.rename(destination)
    binaries = [destination / path.relative_to(app) for path in binaries]
    app = destination
    runtime = app / "Contents/Resources/Runtime"
    print("4/6 内部コードとアプリに署名", flush=True)
    timestamp = notarize
    sign(binaries, identity, timestamp=timestamp)
    sign_sparkle(app, identity, timestamp=timestamp)
    sign([app], identity, timestamp=timestamp)
    if identity == "-":
        print("警告: ad-hoc署名です。", flush=True)
    else:
        verify_signature(app, "release" if notarize else args.channel)
    run("codesign", "--verify", "--deep", "--strict", app)
    # Basic runtime checks with only the bundle and OS tools on PATH.
    env = {**os.environ, "PATH": str(runtime / "bin") + ":/usr/bin:/bin", "PYTHONNOUSERSITE": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "NUMBA_CACHE_DIR": str(BUILD / "numba-cache")}
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    run(runtime / "python-api/bin/python3.12", "-I", "-B", "-c", "import fastapi, uvicorn, httpx, yt_dlp, huggingface_hub, yt_dlp_ejs", env=env)
    run(runtime / "python-tts/bin/python3.11", "-I", "-B", "-c", "import torch, torchaudio, tiktoken; assert torch.__version__ == '2.8.0'; "
        "from speechbrain.lobes.features import Fbank; from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN; "
        "from speechbrain.processing.features import InputNormalization; "
        "import soundfile; import mlx_whisper.whisper; from mlx_whisper.tokenizer import get_tokenizer; "
        "get_tokenizer(True, num_languages=100, language='ja', task='transcribe'); "
        "import torchvision, transformers; from transformers import AutoModel; "
        "assert tuple(map(int, transformers.__version__.split('.')[:2])) >= (5, 19); "
        "import mlx.core as mx; assert torch.backends.mps.is_available(); mx.eval(mx.ones(1) + 1)", env=env)
    run(runtime / "python-tts/bin/python3.11", "-B", "-s", "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from transiroyt.mel_roformer.model import MelRoFormer; "
        "from transiroyt.mel_roformer.config import MelRoFormerConfig; "
        "model = MelRoFormer(MelRoFormerConfig()); "
        "import mlx.core as mx; assert mx.__version__ == '0.32.4'; mx.eval(model.parameters())",
        runtime / "backend", env=env)
    run(runtime / "python-indextts/bin/python3.11", "-I", "-B", "-c",
        "import importlib.util; "
        "assert all(importlib.util.find_spec(name) is None "
        "for name in ('cv2', 'distance', 'g2p_en', 'textstat', 'pyphen')); "
        "import torch, transformers, modelscope, librosa, wetext, soundfile, indextts, indextts.infer_v2_5; "
        "assert transformers.__version__ == '4.52.1'; "
        "assert torch.backends.mps.is_available()", env=env)
    pinned = {item["name"]: item["version"] for item in tomllib.loads((ROOT / "uv.lock").read_text())["package"]
              if item["name"] in {"yt-dlp", "yt-dlp-ejs", "certifi"}}
    actual = json.loads(run(runtime / "python-api/bin/python3.12", "-I", "-B", "-c",
                                "import importlib.metadata as m, json; print(json.dumps({name:m.version(name) for name in "
                                + repr(list(pinned)) + "}))", env=env, stdout=subprocess.PIPE, text=True).stdout)
    if actual != pinned:
        raise ValueError(f"同梱動画取得ツールがuv.lockと一致しません: {actual} != {pinned}")
    run(runtime / "bin/ffmpeg", "-version", stdout=subprocess.DEVNULL, env=env)
    run(runtime / "bin/node", "--version", stdout=subprocess.DEVNULL, env=env)
    run(runtime / "bin/llama-server", "--version", stdout=subprocess.DEVNULL, env=env)
    run(runtime / "python-api/bin/python3.12", "-B", "-s", "-c",
        "from transiroyt.gemma_audio import llama_server_binary; import os; "
        "assert llama_server_binary() == os.environ['TRANSIROYT_RESOURCE_ROOT'] + '/bin/llama-server'; "
        "from transiroyt.installation import check_consumer_runtime; "
        "from transiroyt.config import Settings; check_consumer_runtime(Settings())",
        env={**env, "PATH": str(runtime / "bin") + ":/usr/bin:/bin",
             "PYTHONPATH": str(runtime / "backend"), "TRANSIROYT_RESOURCE_ROOT": str(runtime)})
    run("codesign", "--verify", "--deep", "--strict", app)
    if args.app_only:
        print(f"完了: {app}（ローカルpreview、DMG作成なし）", flush=True)
        return
    print("5/6 対応ソースを作成", flush=True)
    with zipfile.ZipFile(OUTPUT / f"TransiroYT-{VERSION}-third-party-sources.zip", "w", zipfile.ZIP_STORED) as archive:
        for file in [ROOT / f".local/distribution-build/sources/llama.cpp-{json.loads((ROOT / 'distribution/llama.lock.json').read_text())['revision']}.tar.gz",
                     ROOT / "distribution/llama.lock.json",
                     ROOT / ".local/distribution-build/sources/ffmpeg.tar.xz", ROOT / ".local/distribution-build/sources/dav1d-1.5.3.tar.xz", *corresponding_sources,
                     ROOT / "scripts/build_consumer_tools.sh",
                     ROOT / "distribution/third-party-sources.lock.json",
                     ROOT / "distribution/REBUILDING-LGPL.md"]:
            archive.write(file, file.name)
        for file in sorted((runtime / "licenses").rglob("*")):
            if file.is_file():
                archive.write(file, "licenses/" + str(file.relative_to(runtime / "licenses")))
    print("6/6 DMGを作成", flush=True)
    staging = BUILD / "dmg"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    # This is the already signed, sanitized app. Preserve its generated bytecode
    # as well as the signature, rather than applying the source-copy filter again.
    copy_tree(app, staging / "TransiroYT.app", include_bytecode=True)
    run("codesign", "--verify", "--deep", "--strict", staging / "TransiroYT.app")
    (staging / "Applications").symlink_to("/Applications", target_is_directory=True)
    copy_tree(runtime / "help", staging / "はじめに")
    suffix = "-preview" if args.channel == "preview" else ""
    filename = f"TransiroYT-{VERSION}-arm64{suffix}.dmg"
    dmg = BUILD / (filename.removesuffix(".dmg") + "-pending.dmg")
    unoptimized_dmg = BUILD / "unoptimized.dmg"
    run("hdiutil", "create", "-volname", "TransiroYT", "-srcfolder", staging, "-ov", "-format", "UDRW", unoptimized_dmg)
    # Direct creation reserves unused encoding space. Conversion removes that
    # padding while preserving the signed app and the LZMA image format.
    run("hdiutil", "convert", unoptimized_dmg, "-ov", "-format", DMG_FORMAT, "-o", dmg)
    unoptimized_dmg.unlink()
    if dmg.stat().st_size >= 2 * 1024**3:
        raise ValueError("DMGがGitHub Releasesの1ファイル2 GiB未満の条件を満たしません。")
    notarized = False
    submission = None
    if notarize:
        submission = notarize_dmg(dmg, identity, args.notary_profile)
        notarized = True
    final_dmg = OUTPUT / filename
    shutil.move(dmg, final_dmg)
    dmg = final_dmg
    report = {**distribution, "notarized": notarized, "dmg": dmg.name, "image_format": DMG_FORMAT,
              "signing_authority": signature_authority(app),
              "notarization_id": submission["id"] if submission else None,
              "sha256": hashlib.file_digest(dmg.open("rb"), "sha256").hexdigest(),
              "public_release_ready": args.channel == "release" and notarized}
    (OUTPUT / "release-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    # This folder can be hosted as a static download/help site without a build step.
    landing = help_text
    if args.channel == "preview":
        label = "公証済み" if notarized else "未公証"
        landing = landing.replace('class="small">Apple Silicon', f'class="small">公開準備版（{label}）<br>Apple Silicon')
    (OUTPUT / "index.html").write_text(landing)
    shutil.copy2(ROOT / "guide/guide.css", OUTPUT / "guide.css")
    (OUTPUT / f"{dmg.name}.sha256").write_text(f"{report['sha256']}  {dmg.name}\n")
    config = update_configuration()
    update_tag = f"v{VERSION}-build-{build_version}" + ("-preview" if args.channel == "preview" else "")
    archive_url = f"https://github.com/{config['repository']}/releases/download/{update_tag}/{dmg.name}"
    feed = OUTPUT / ("appcast.xml" if args.channel == "release" else "appcast-preview.xml")
    if args.channel == "release" or args.update_feed:
        create_feed(app, dmg, feed, archive_url, ROOT / ".local/DesktopBuild/artifacts/sparkle/Sparkle/bin")
    (OUTPUT / "update-manifest.json").write_text(json.dumps({"tag": update_tag, "feed": feed.name,
        "archive": dmg.name, "archive_url": archive_url, "build": build_version}, indent=2) + "\n")
    print(f"完了: {dmg}\n対応OS: macOS {minimum}以上 / Apple Silicon\n公開可能: {report['public_release_ready']}", flush=True)


if __name__ == "__main__":
    main()
