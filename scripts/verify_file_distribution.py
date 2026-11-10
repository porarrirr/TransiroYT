"""Run the shipped file-dubbing pipeline with an isolated HOME and no developer tools.

Supply a short video containing real speech (and optionally its SRT). The first
run downloads the production models into --data. Existing user data is untouched.
This checks packaging and actual inference, not the perceptual quality of a dub.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--subtitles", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    app, support = args.app.resolve(), args.data.resolve()
    if support.exists() and not args.resume:
        parser.error("既存の検証データは変更しません。新しい--dataか--resumeを指定してください。")
    support.mkdir(parents=True, exist_ok=True)
    runtime, data = app / "Contents/Resources/Runtime", support / ".local"
    home, temporary = support / "home", support / "tmp"
    for directory in (data, home, temporary):
        directory.mkdir(exist_ok=True)
    env = {
        "HOME": str(home), "TMPDIR": str(temporary), "LANG": "en_US.UTF-8",
        "DEVELOPER_DIR": str(support / "no-developer-tools"),
        "PATH": str(runtime / "bin") + ":/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(runtime / "backend"), "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "TRANSIROYT_RESOURCE_ROOT": str(runtime), "TRANSIROYT_DATA_DIR": str(data),
        "TRANSIROYT_CONFIG_FILE": str(support / ".env"),
        "TRANSIROYT_TTS_PYTHON": str(runtime / "python-tts/bin/python3.11"),
        "TRANSIROYT_TTS_SITE_PACKAGES": str(runtime / "python-tts/lib/python3.11/site-packages"),
        "TRANSIROYT_INDEXTTS_PYTHON": str(runtime / "python-indextts/bin/python3.11"),
        "HF_HOME": str(data / "huggingface"), "HF_HUB_CACHE": str(data / "huggingface/hub"),
        "HF_HUB_DISABLE_XET": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "XDG_CACHE_HOME": str(data / "cache"), "NUMBA_CACHE_DIR": str(data / "cache/numba"),
        "TORCH_HOME": str(data / "cache/torch"), "MPLCONFIGDIR": str(data / "cache/matplotlib"),
    }
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    began = time.monotonic()
    with (support / "verification.log").open("a") as log:
        server = subprocess.Popen(
            [str(runtime / "python-api/bin/python3.12"), "-B", "-s", "-m", "uvicorn",
             "transiroyt.app:create_app", "--factory", "--host", "127.0.0.1", "--port", str(port)],
            cwd=support, env=env, stdout=log, stderr=log)
        try:
            def request(path, body=None):
                token = (data / "token").read_text().strip()
                headers = {"Authorization": f"Bearer {token}"}
                if body is not None:
                    headers["Content-Type"] = "application/json"
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}{path}", headers=headers,
                    data=json.dumps(body).encode() if body is not None else None)
                with urllib.request.urlopen(req, timeout=30) as response:
                    return json.load(response)

            for _ in range(150):
                if server.poll() is not None:
                    raise RuntimeError("同梱サーバーが終了しました。verification.logを確認してください。")
                try:
                    before = request("/api/file-dubbing")
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.2)
            else:
                raise TimeoutError("同梱サーバーが起動しませんでした。")
            if not args.resume:
                assert not any(before["models"][key] for key in (
                    "coreml", "gemma", "speaker_model", "diarization_model", "separation_model", "speech_check_model", "indextts_model"))
                assert not (data / "setup-ready.json").exists()
            output = support / "dubbed.mp4"
            request("/api/file-dubbing", {
                "video": str(args.video.resolve()),
                "subtitles": str(args.subtitles.resolve()) if args.subtitles else None,
                "output": str(output), "allow_download": True, "overwrite": args.resume,
                "steps": 10, "speakers": 1,
            })
            last = None
            # A fresh run downloads the file-dubbing models too, including the
            # ~7.3 GB IndexTTS-2.5 set, so allow a slow connection to finish.
            while time.monotonic() - began < 3 * 3600:
                try:
                    job = request("/api/file-dubbing")["job"]
                except (OSError, urllib.error.URLError):
                    if server.poll() is not None:
                        raise RuntimeError("同梱サーバーが終了しました。verification.logを確認してください。") from None
                    print("状態の取得を再試行しています。", flush=True)
                    time.sleep(1)
                    continue
                if job != last:
                    (support / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2))
                    if last is None or (job["stage"], job["message"]) != (last["stage"], last["message"]):
                        print(job["stage"], job["message"], flush=True)
                    last = job
                if job["state"] != "running":
                    break
                time.sleep(1)
            else:
                raise TimeoutError("ファイル吹き替えの検証が3時間を超えました。")
            assert job["state"] == "done", job["error"]
            assert job["summary"]["ready"] > 0, "音声生成に成功した区間がありません。"
            probe = subprocess.run(
                [str(runtime / "bin/ffprobe"), "-v", "error", "-show_streams", "-of", "json", str(output)],
                env=env, capture_output=True, text=True, check=True)
            streams = json.loads(probe.stdout)["streams"]
            assert any(stream["codec_type"] == "video" and stream["codec_name"] == "h264" for stream in streams)
            assert any(stream["codec_type"] == "audio" and stream["codec_name"] == "aac" for stream in streams)
            subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)], check=True)
            report = {"app": str(app), "fresh_model_download": not args.resume,
                      "isolated_home": str(home), "path": env["PATH"],
                      "developer_tools_disabled": True,
                      "legacy_setup_required": False, "real_file_pipeline": True,
                      "summary": job["summary"], "h264_aac_output": True,
                      "signature_verified": True, "elapsed_seconds": round(time.monotonic() - began, 1)}
            (support / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
