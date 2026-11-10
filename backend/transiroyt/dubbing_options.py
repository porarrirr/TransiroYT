"""Input paths and validated options for file dubbing."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import FileDubbingError

LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")


@dataclass
class FileDubbingOptions:
    video: Path
    subtitles: Path
    output: Path
    # Level of the original speech under a dubbed line; background stays at 1.
    original_volume: float = 0.0
    dub_volume: float = 1.0
    source_language: str = "auto"
    speakers: int | None = None
    steps: int | None = None
    allow_download: bool = False
    title: str | None = None
    uploader: str | None = None
    thumbnail_url: str | None = None
    # No subtitles exist: ``subtitles`` is where the SRT transcribed from the speech is written.
    transcribe: bool = False

    def validate(self, *, require_inputs=True):
        self.video, self.subtitles = Path(self.video).expanduser(), Path(self.subtitles).expanduser()
        self.output = Path(self.output).expanduser()
        if require_inputs and not self.video.is_file():
            raise FileDubbingError(f"動画ファイルが見つかりません: {self.video}")
        if require_inputs and not self.transcribe and not self.subtitles.is_file():
            raise FileDubbingError(f"字幕ファイルが見つかりません: {self.subtitles}")
        if self.subtitles.suffix.lower() != ".srt":
            raise FileDubbingError("字幕はSRT形式（.srt）を指定してください。")
        if self.output.suffix.lower() != ".mp4":
            raise FileDubbingError("出力ファイルの拡張子は .mp4 にしてください。")
        if self.output.resolve() in {self.video.resolve(), self.subtitles.resolve()}:
            raise FileDubbingError("出力先に入力ファイルを指定することはできません。")
        if not self.output.parent.is_dir():
            raise FileDubbingError(f"出力先のフォルダがありません: {self.output.parent}")
        for name, value in (("吹き替え中の元の声", self.original_volume), ("吹き替え音声", self.dub_volume)):
            if not (isinstance(value, int | float) and math.isfinite(value) and 0 <= value <= 2):
                raise FileDubbingError(f"{name}の音量は0〜2の範囲で指定してください。")
        if self.dub_volume == 0:
            raise FileDubbingError("吹き替え音声の音量が0では吹き替えになりません。")
        if self.speakers is not None and not 1 <= self.speakers <= 20:
            raise FileDubbingError("話者数は1〜20で指定してください。")
        if self.steps is not None and not 1 <= self.steps <= 32:
            raise FileDubbingError("生成ステップ数は1〜32で指定してください。")
        if self.source_language != "auto" and not LANGUAGE_CODE.fullmatch(self.source_language):
            raise FileDubbingError("元の言語はenやzh-Hansのような言語コードで指定してください。")
