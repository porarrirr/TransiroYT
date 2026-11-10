"""Python of the dedicated file-dubbing analysis environment (PyTorch and MLX)."""
from __future__ import annotations

import os
from pathlib import Path

from .config import Settings


def tts_python(settings: Settings) -> Path:
    return Path(os.getenv("TRANSIROYT_TTS_PYTHON",
                          settings.root / ".local/file-workers/bin/python"))
