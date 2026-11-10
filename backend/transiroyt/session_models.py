"""Dubbing segment state and report snapshots, independent of orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Segment:
    index: int
    start: float
    end: float
    source: str
    text: str | None = None
    state: str = "pending"
    error: str | None = None
    skip_reason: str | None = None
    audio: dict = field(default_factory=dict)
    playback_start: float | None = None
    playback_end: float | None = None
    boundary_before: bool = False
    speech_limit: float | None = None

    @property
    def speech_start(self):
        return self.start if self.playback_start is None else self.playback_start

    @property
    def speech_end(self):
        return self.end if self.playback_end is None else self.playback_end

    def public(self):
        return {"index": self.index, "start": self.speech_start, "end": self.speech_end,
                "source_start": self.start, "source_end": self.end,
                "source": self.source, "text": self.text, "state": self.state,
                "error": self.error, "skip_reason": self.skip_reason,
                **self.audio}
