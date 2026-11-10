"""Validated request contracts shared by the HTTP routes."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .errors import FileDubbingError


class StartFileDubbing(BaseModel):
    video: str | None = Field(default=None, min_length=1, max_length=4096)
    subtitles: str | None = Field(default=None, min_length=1, max_length=4096)
    youtube_url: str | None = Field(default=None, min_length=1, max_length=2048)
    youtube_quality: Literal["best", "2160", "1440", "1080", "720", "480", "360"] = "best"
    output: str = Field(min_length=1, max_length=4096)
    # Original speech under dubbed lines; the background keeps its own level.
    original_volume: float = Field(default=0.0, ge=0, le=2, allow_inf_nan=False)
    dub_volume: float = Field(default=1.0, gt=0, le=2, allow_inf_nan=False)
    language: str = Field(default="auto", pattern=r"^(auto|[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*)$")
    speakers: int | None = Field(default=None, ge=1, le=20, strict=True)
    # Irodori sampling steps; None uses the file-dubbing default.
    steps: int | None = Field(default=None, ge=1, le=32, strict=True)
    allow_download: bool = Field(default=False, strict=True)
    overwrite: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def input_source(self):
        if self.youtube_url:
            if self.video or self.subtitles:
                raise ValueError("YouTube URLとローカルファイルは同時に指定できません。")
            from .youtube_file import youtube_video_id
            try:
                youtube_video_id(self.youtube_url)
            except FileDubbingError as exc:
                raise ValueError(str(exc)) from exc
        elif not self.video:
            raise ValueError("動画、またはYouTube URLを指定してください。")
        return self
