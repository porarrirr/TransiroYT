"""Distribution runtime check; never download or load models for a connection."""
from __future__ import annotations

import os

from .config import Settings


def check_consumer_runtime(settings: Settings) -> None:
    """Validate the executable bundle, without searching the developer's PATH."""
    if not (settings.root / "distribution.json").is_file():
        return
    from importlib.metadata import PackageNotFoundError, requires, version

    from packaging.requirements import Requirement

    for relative in ("bin/llama-server", "bin/ffmpeg", "bin/ffprobe", "bin/node",
                     "bin/transiro-coreml", ".local/system-transcribe",
                     "python-api/bin/python3.12", "python-tts/bin/python3.11",
                     "python-indextts/bin/python3.11"):
        if not os.access(settings.root / relative, os.X_OK):
            raise RuntimeError("必要な実行環境がアプリに含まれていません。アプリを再インストールしてください。")
    from .native_audio import frame_energies

    if frame_energies(b"\x00\x80\xff\x7f", 2) != [2147418113]:
        raise RuntimeError("Rust音声処理の検証に失敗しました。")
    try:
        ejs = next(Requirement(item) for item in requires("yt-dlp") if item.startswith("yt-dlp-ejs"))
        if version("yt-dlp-ejs") not in ejs.specifier:
            raise ValueError("EJS version mismatch")
    except (PackageNotFoundError, StopIteration, ValueError) as exc:
        raise RuntimeError("動画取得ツールの構成が正しくありません。アプリを再インストールしてください。") from exc
