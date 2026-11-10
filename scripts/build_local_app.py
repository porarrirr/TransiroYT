"""Build and optionally install a signed local preview using an unchanged bundled runtime.

The base app must have passed a full build with the same dependency/helper inputs.
Release archives and notarization continue to use build_dmg.py.
"""
from __future__ import annotations

import argparse
import json
import os
import plistlib
import re
import shutil
import subprocess
import time
from pathlib import Path

import httpx
from build_dmg import (
    ROOT,
    compile_bytecode,
    preview_build_version,
    preview_signing_identity,
    run,
    runtime_inputs,
    sign,
    verify_signature,
)
from build_native_audio import build as build_native_audio


def validate_base(app, root=ROOT):
    runtime = app / "Contents/Resources/Runtime"
    distribution = json.loads((runtime / "distribution.json").read_text())
    if distribution["channel"] != "preview":
        raise ValueError("ローカル更新の元にはpreviewアプリを指定してください。")
    manifest = runtime / "local-build-inputs.json"
    if not manifest.is_file() or json.loads(manifest.read_text()) != runtime_inputs(root):
        raise ValueError("依存・ヘルパーの構成記録がないか変更されています。build_dmg.py --channel preview --app-onlyで再ビルドしてください。")
    return runtime


def sync_files(files, destination):
    current = {path.relative_to(destination): path for path in destination.rglob("*")
               if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    if files.keys() == current.keys() and all(path.read_bytes() == current[name].read_bytes()
                                             for name, path in files.items()):
        return False
    shutil.rmtree(destination)
    destination.mkdir()
    for name, path in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return True


def copy_sources(runtime, root=ROOT):
    changed = {}
    # Replace changed owned directories so removed modules cannot survive an update.
    for source, destination in [(root / "backend", runtime / "backend"), (root / "guide", runtime / "help")]:
        files = {path.relative_to(source): path for path in source.rglob("*")
                 if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
        changed[destination.name] = sync_files(files, destination)
    files = {Path(name): root / "scripts" / name for name in
             ("speaker_embedding.py", "speaker_activity.py", "separate_vocals.py", "whisper_worker.py", "indextts_worker.py")}
    changed["scripts"] = sync_files(files, runtime / "scripts")
    changed["model-notices"] = sync_files(
        {path.relative_to(root / "distribution/licenses/models"): path for path in (root / "distribution/licenses/models").rglob("*") if path.is_file()},
        runtime / "licenses/supplemental/models")
    shutil.copy2(root / "distribution/licenses/models.json", runtime / "licenses/supplemental/models.json")
    shutil.copy2(root / "docs/licenses.md", runtime / "licenses/README-ja.md")
    shutil.copy2(root / "distribution/fallback-models.lock.json", runtime / "fallback-models.lock.json")
    return changed


def app_processes(app):
    executable = str(app / "Contents/MacOS/TransiroYT")
    processes = subprocess.check_output(["ps", "-axo", "pid=,comm="], text=True)
    return [int(line.split(maxsplit=1)[0]) for line in processes.splitlines()
            if len(line.split(maxsplit=1)) == 2 and line.split(maxsplit=1)[1] == executable]


def installed_app():
    for app in (Path.home() / "Applications/TransiroYT.app", Path("/Applications/TransiroYT.app")):
        if app_processes(app):
            return app
    return next((app for app in (Path.home() / "Applications/TransiroYT.app", Path("/Applications/TransiroYT.app"))
                 if app.exists()), Path.home() / "Applications/TransiroYT.app")


def install_app(app, destination, work):
    support = Path.home() / "Library/Application Support/TransiroYT/.local"
    token = (support / "token").read_text().strip()
    # A running export owns files and workers; leave it running instead of interrupting it.
    old_instance = None
    if app_processes(destination):
        with httpx.Client(base_url="http://127.0.0.1:8765", headers={"Authorization": f"Bearer {token}"}) as client:
            old_instance = client.get("/health").raise_for_status().json()["instance_id"]
            job = client.get("/api/file-dubbing").raise_for_status().json()["job"]
        if job and job["state"] == "running":
            raise ValueError("吹き替え実行中です。完了後に更新してください。")
        for pid in app_processes(destination):
            run("swift", "-e", "import AppKit; _ = NSRunningApplication(processIdentifier: Int32(CommandLine.arguments[1])!)?.terminate()", pid)
        deadline = time.monotonic() + 15
        while app_processes(destination):
            if time.monotonic() > deadline:
                raise RuntimeError("アプリが終了しないため、置き換えを中止しました。")
            time.sleep(0.1)
    previous = work / "previous.app"
    destination.rename(previous)
    try:
        # work and Applications are on the same volume in the normal local workflow.
        shutil.move(str(app), destination)
        run("open", destination)
        expected = plistlib.loads((destination / "Contents/Info.plist").read_bytes())["CFBundleVersion"]
        deadline = time.monotonic() + 30
        with httpx.Client(base_url="http://127.0.0.1:8765", headers={"Authorization": f"Bearer {token}"}, timeout=1) as client:
            while time.monotonic() < deadline:
                try:
                    health = client.get("/health")
                    if (app_processes(destination) and health.status_code == 200
                            and health.json()["instance_id"] != old_instance
                            and client.get("/api/library").status_code == 200):
                        print(f"起動確認: {destination} / build {expected}", flush=True)
                        return
                except httpx.TransportError:
                    pass
                time.sleep(0.2)
        raise RuntimeError("更新したアプリの接続を確認できませんでした。previous.appを保持しています。")
    except Exception:
        if not destination.exists():
            previous.rename(destination)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", type=Path, help="Target installed app; defaults to the running installation")
    parser.add_argument("--base-app", type=Path, help="Signed preview baseline; defaults to the installed app")
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--install", action="store_true", help="Replace the installed app and launch it after validation")
    args = parser.parse_args()
    destination = (args.installed or installed_app()).expanduser().resolve()
    base = (args.base_app or destination).expanduser().resolve()
    work = args.work_dir.resolve()
    app = work / "TransiroYT.app"
    if app.exists() or (work / "previous.app").exists():
        parser.error("成果物のない新しい作業フォルダを指定してください。")
    validate_base(base)
    identity = preview_signing_identity()
    if not identity:
        parser.error("ローカル更新にはApple Development署名が必要です。")
    work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timings = {}

    def stage(name, action):
        beginning = time.monotonic()
        result = action()
        timings[name] = round(time.monotonic() - beginning, 3)
        print(f"{name}: {timings[name]:.1f}秒", flush=True)
        (work / "timings.json").write_text(json.dumps({**timings, "total": round(time.monotonic() - started, 3)}, indent=2))
        return result

    stage("元アプリの署名検証", lambda: run("codesign", "--verify", "--deep", "--strict", base))
    stage("Rust音声処理のビルド", build_native_audio)
    stage("Swiftビルド", lambda: run("swift", "build", "--package-path", ROOT / "apple/Desktop", "-c", "release",
                                    "--scratch-path", ROOT / ".local/DesktopBuild"))
    stage("署名済みランタイムの複製", lambda: run("cp", "-cR", base, app))
    runtime = app / "Contents/Resources/Runtime"
    changed = stage("コードとガイドの反映", lambda: copy_sources(runtime))
    binary_directory = subprocess.check_output(["swift", "build", "--package-path", str(ROOT / "apple/Desktop"),
        "-c", "release", "--scratch-path", str(ROOT / ".local/DesktopBuild"), "--show-bin-path"], text=True).strip()
    executable = app / "Contents/MacOS/TransiroYT"
    shutil.copy2(Path(binary_directory) / "TransiroYT", executable)
    commands = subprocess.check_output(["otool", "-l", str(executable)], text=True)
    for rpath in re.findall(r"cmd LC_RPATH\s+cmdsize \d+\s+path ([^\n]+?) \(offset", commands):
        if rpath.startswith("/") and not rpath.startswith(("/usr/lib/", "/System/Library/")):
            run("install_name_tool", "-delete_rpath", rpath, executable)
    stage("アイコン", lambda: run("swift", ROOT / "scripts/create_app_icon.swift", app / "Contents/Resources"))
    plist = app / "Contents/Info.plist"
    info = plistlib.loads(plist.read_bytes())
    info["CFBundleVersion"] = preview_build_version()
    plist.write_bytes(plistlib.dumps(info))
    for directory, python in (("backend", "python-api/bin/python3.12"), ("scripts", "python-tts/bin/python3.11")):
        if changed[directory]:
            stage(f"{directory}のコンパイル", lambda python=python, directory=directory: compile_bytecode(runtime / python, runtime, runtime / directory))
    stage("Rust音声処理の署名", lambda: sign([runtime / "backend/transiroyt/_audio_native.dylib"], identity, timestamp=False))
    stage("アプリの署名", lambda: sign([app], identity, timestamp=False))
    verify_signature(app, "preview")
    stage("完成アプリの署名検証", lambda: run("codesign", "--verify", "--deep", "--strict", app))
    env = {**os.environ, "PYTHONPATH": str(runtime / "backend"), "TRANSIROYT_RESOURCE_ROOT": str(runtime),
           "PATH": str(runtime / "bin") + ":/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
           "TRANSIROYT_DATA_DIR": str(work / "validation-data")}
    env.pop("PYTHONHOME", None)
    stage("同梱ランタイムの検証", lambda: run(runtime / "python-api/bin/python3.12", "-B", "-s", "-c",
        "from transiroyt.installation import check_consumer_runtime; from transiroyt.config import Settings; "
        "check_consumer_runtime(Settings()); from transiroyt.app import create_app; create_app(Settings())", env=env))
    if args.install:
        stage("置き換えと起動", lambda: install_app(app, destination, work))
        stage("退避アプリの整理", lambda: shutil.rmtree(work / "previous.app"))
    print(f"完了: {destination if args.install else app} / 合計{time.monotonic() - started:.1f}秒", flush=True)


if __name__ == "__main__":
    main()
