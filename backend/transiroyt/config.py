from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.getenv("TRANSIROYT_RESOURCE_ROOT", Path(__file__).resolve().parents[2]))


def load_env() -> None:
    path = Path(os.getenv("TRANSIROYT_CONFIG_FILE", ROOT / ".env"))
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@dataclass
class Settings:
    root: Path = ROOT
    data: Path = field(default_factory=lambda: Path(os.getenv("TRANSIROYT_DATA_DIR", ROOT / ".local")))
    device: str = "mps"
    steps: int = 4
    voice: str = "落ち着いた、明瞭で自然な日本語のナレーション。"
    cookies_browser: str | None = None
    token: str = ""

    @property
    def system_transcribe_binary(self):
        return self.root / ".local/system-transcribe"

    @classmethod
    def from_env(cls) -> Settings:
        load_env()
        settings = cls(
            device=os.getenv("TRANSIROYT_DEVICE", "mps"),
            steps=int(os.getenv("TRANSIROYT_TTS_STEPS", "4")),
            voice=os.getenv("TRANSIROYT_VOICE", cls.voice),
            cookies_browser=os.getenv("TRANSIROYT_COOKIES_BROWSER") or None,
        )
        if not 1 <= settings.steps <= 32:
            raise ValueError("TRANSIROYT_TTS_STEPS must be between 1 and 32")
        settings.data.mkdir(mode=0o700, parents=True, exist_ok=True)
        settings.data.chmod(0o700)
        token_file = settings.data / "token"
        if not token_file.exists():
            # Exclusive creation avoids two startup processes generating different secrets.
            try:
                with open(token_file, "x", opener=lambda p, f: os.open(p, f, 0o600)) as f:
                    f.write(secrets.token_urlsafe(32))
            except FileExistsError:
                pass
        settings.token = token_file.read_text().strip()
        return settings
