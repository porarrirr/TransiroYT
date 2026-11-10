"""Local SubRip (SRT) subtitles for file dubbing."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .caption_annotations import voiced_annotations
from .captions import Cue, clean_text, speaker_parts, voiced_cues

_TIME = r"(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})"
_TIMING = re.compile(rf"^\s*{_TIME}\s*-->\s*{_TIME}")
_ASS_TAG = re.compile(r"\{\\[^}]*\}")
# "- Hello. - Hi." / "-Hello" marks one line per speaker in subtitle conventions.
_DIALOGUE = re.compile(r"^\s*[-‐–—]\s*(?=\S)")
_SENTENCE_END = re.compile(r"[.!?。！？…][\"'”’」』)）]*$")
# A pause this long inside a Whisper segment starts a new subtitle.
TRANSCRIPT_PAUSE_SECONDS = 1.0


class SubtitleError(ValueError):
    pass


def _seconds(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(text: str) -> list[Cue]:
    """Parse SRT strictly by timing lines; text lines keep their speaker turns.

    A cue whose lines start with dialogue dashes is split into consecutive cues,
    one per marked line; these keep a ">>" speaker marker for spoken_cues. Timing
    inside the original cue is proportional to the text length; only the original
    cue's endpoints are exact.
    """
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [line for line in block.split("\n") if line.strip()]
        timing_index = next((i for i, line in enumerate(lines[:2]) if _TIMING.match(line)), None)
        if timing_index is None:
            if lines:
                raise SubtitleError(f"SRTの時刻行を解釈できません: {lines[0][:60]}")
            continue
        match = _TIMING.match(lines[timing_index])
        start, end = _seconds(*match.groups()[:4]), _seconds(*match.groups()[4:])
        if end <= start:
            raise SubtitleError(f"SRTの終了時刻が開始時刻以前です: {lines[timing_index].strip()}")
        body = [clean_text(_ASS_TAG.sub("", line)) for line in lines[timing_index + 1:]]
        body = [line for line in body if line]
        if not body:
            continue
        cues.extend(_split_dialogue(start, end, body))
    if not cues:
        raise SubtitleError("SRTに字幕がありません。")
    return sorted(cues, key=lambda cue: cue.start)


def _timestamp(seconds):
    milliseconds = max(0, round(seconds * 1000))
    seconds, ms = divmod(milliseconds, 1000)
    minutes, s = divmod(seconds, 60)
    h, m = divmod(minutes, 60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def format_srt(cues: list[Cue]) -> str:
    return "\n\n".join(f"{i}\n{_timestamp(c.start)} --> {_timestamp(c.end)}\n{c.text}"
                       for i, c in enumerate(cues, 1)) + "\n"


def transcribed_cues(segments: list[dict]) -> list[Cue]:
    """Subtitles from Whisper segments, one per sentence or pause.

    A Whisper segment can hold several sentences and speakers; speaker turns
    are judged per subtitle, so segments are split on sentence ends and pauses
    using the word times. A subtitle without a time span was not placed in the
    audio and is dropped.
    """
    cues = []

    def flush(words):
        text = clean_text("".join(word["word"] for word in words))
        if text and words[-1]["end"] > words[0]["start"]:
            cues.append(Cue(words[0]["start"], words[-1]["end"], text))

    for segment in segments:
        words = [word for word in segment["words"] if word["word"].strip()]
        current = []
        for word in words:
            if current and word["start"] - current[-1]["end"] >= TRANSCRIPT_PAUSE_SECONDS:
                flush(current)
                current = []
            current.append(word)
            if _SENTENCE_END.search(word["word"].strip()):
                flush(current)
                current = []
        if current:
            flush(current)
    return cues


def _split_dialogue(start: float, end: float, lines: list[str]) -> list[Cue]:
    if len(lines) < 2 or not all(_DIALOGUE.match(line) for line in lines):
        joined = " ".join(_DIALOGUE.sub("", line) if len(lines) == 1 else line for line in lines)
        return [Cue(start, end, joined)]
    turns = [">> " + _DIALOGUE.sub("", line) for line in lines]
    weights = [max(1, len(turn)) for turn in turns]
    total, cursor, result = sum(weights), start, []
    for turn, weight in zip(turns, weights, strict=True):
        following = cursor + (end - start) * weight / total
        result.append(Cue(cursor, following, turn))
        cursor = following
    result[-1].end = end
    return result


@dataclass
class CaptionEvent:
    """Annotations heard outside the dubbed words: reactions, music, sounds."""

    start: float
    end: float
    labels: list[str]
    # Only the words of one speaker are dubbed; laughter or applause from
    # others, music and sound effects stay in the original soundtrack.
    kinds: list[str] = field(default_factory=list)


def spoken_cues(cues: list[Cue]) -> list[Cue]:
    """Spoken cues, with the speaker's own non-verbal sounds as voice emojis.

    Rolling auto-caption windows are removed and each cue ends no later than the
    next one starts, as for captions fetched from YouTube. Speaker markers are
    never spoken: a cue is split at each ">>", proportionally to text length, and
    the part after a marker is flagged as a change of speaker. A marker before
    annotations only (">> [laughter]") still means someone else took over: the
    next spoken words start a new turn even without a marker of their own.
    Music, sound effects and audience reactions are removed (caption_events).
    """
    return [cue for cue in voiced_cues(cues) if voiced_annotations(cue.text)[0]]


def caption_events(cues: list[Cue]) -> list[CaptionEvent]:
    """Annotations that are not voiced in the dub, on the source clock.

    An annotation-only part spans its own caption time. Inline annotations
    other than the speaker's voice cover the whole caption part: captions do
    not time individual words.
    """
    events = []
    for start, end, part, _, _ in speaker_parts(cues):
        text, annotations = voiced_annotations(part)
        silent = [a for a in annotations if not (text and a.emoji)]
        if silent:
            events.append(CaptionEvent(start, end, [a.label for a in silent], [a.kind for a in silent]))
    return events
