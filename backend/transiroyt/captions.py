from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field

from .caption_annotations import classify, strip_non_speech_annotations, voiced_annotations
from .config import Settings

logger = logging.getLogger(__name__)

# Closed captions (YouTube, broadcast) start each new speaker's words with ">>".
# A caption without it continues the previous speaker, even across cues.
_SPEAKER_MARKER = re.compile(r"\s*>>+\s*")
# Square-bracketed proper names at the start of a caption or sentence are
# closed-caption speaker labels. Inline mentions and unknown prose stay intact.
_NAME = r"[A-Z][A-Za-z]*(?:[-'’][A-Za-z]+)*"
_NAMED_SPEAKER = re.compile(rf"(?:^|(?<=[.!?。！？]))\s*\[(?P<name>{_NAME}(?: {_NAME}){{0,2}})\]\s*")


class CaptionError(RuntimeError):
    pass


@dataclass
class Cue:
    start: float
    end: float
    text: str
    boundary_before: bool = field(default=False, compare=False)
    # A subtitle marker (">>", dialogue dash) says a different person starts here.
    speaker_change: bool = field(default=False, compare=False)
    speaker_label: str | None = field(default=None, compare=False)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", "", text))).strip()


def parse_json3(data: dict) -> list[Cue]:
    cues = []
    for event in data.get("events", []):
        text = clean_text("".join(s.get("utf8", "") for s in event.get("segs", [])))
        if not text or "tStartMs" not in event:
            continue
        start = max(0, float(event["tStartMs"]) / 1000)
        end = start + max(0.01, float(event.get("dDurationMs", 2000)) / 1000)
        if (event.get("aAppend") and cues and start <= cues[-1].end + 0.1
                and strip_non_speech_annotations(text)
                and strip_non_speech_annotations(cues[-1].text)):
            cues[-1].text = clean_text(cues[-1].text + " " + text)
            cues[-1].end = max(cues[-1].end, end)
            continue
        cues.append(Cue(start, end, text))
    return normalize_cues(cues)


def normalize_cues(cues: list[Cue]) -> list[Cue]:
    """Remove rolling ASR windows and bound every cue by the next cue's onset."""
    ordered = sorted(cues, key=lambda c: c.start)
    unique: list[Cue] = []
    for cue in ordered:
        if unique and abs(cue.start - unique[-1].start) < 0.03:
            # An independent sound label can share the speech onset. Prefer
            # actual words even when the label is longer than the dialogue.
            spoken = strip_non_speech_annotations(cue.text)
            previous_spoken = strip_non_speech_annotations(unique[-1].text)
            if (bool(spoken), len(spoken)) >= (bool(previous_spoken), len(previous_spoken)):
                unique[-1] = cue
            continue
        # YouTube ASR often includes the previous line again in a rolling window.
        # A sound-only event still bounds the preceding speech even if that
        # same annotation was already present at the end of a rolling line.
        if unique and cue.start < unique[-1].end and strip_non_speech_annotations(cue.text):
            previous = unique[-1].text
            if cue.text.startswith(previous + " "):
                cue = Cue(cue.start, cue.end, cue.text[len(previous):].strip())
            elif previous.endswith(cue.text):
                continue
        unique.append(Cue(cue.start, cue.end, cue.text))
    bounded = []
    for i, cue in enumerate(unique):
        end = min(cue.end, unique[i + 1].start) if i + 1 < len(unique) else cue.end
        if end - cue.start >= 0.05:
            bounded.append(Cue(cue.start, end, cue.text))
    return bounded


def speaker_parts(cues: list[Cue]):
    """(start, end, text, marked, label) for each caption speaker's part.

    Timing inside a cue is proportional to text length; only the cue's own
    endpoints are exact.
    """
    for cue in normalize_cues(cues):
        parts = []
        for number, part in enumerate(_SPEAKER_MARKER.split(cue.text)):
            cursor, label, marked = 0, None, number > 0
            for match in _NAMED_SPEAKER.finditer(part):
                if classify(match['name']) is not None:
                    continue
                before = part[cursor:match.start()].strip()
                if before:
                    parts.append((before, marked, label))
                label, marked, cursor = match['name'], True, match.end()
            parts.append((part[cursor:], marked, label))
        weights = [max(1, len(part)) for part, _, _ in parts]
        total, cursor = sum(weights), cue.start
        for number, ((part, marked, label), weight) in enumerate(zip(parts, weights, strict=True)):
            following = cue.end if number == len(parts) - 1 else cursor + (cue.end - cue.start) * weight / total
            yield cursor, following, part, marked, label
            cursor = following


def voiced_cues(cues: list[Cue]) -> list[Cue]:
    """Use caption metadata instead of reading it: speaker turns and reactions.

    ">>" is never spoken; the words after it are flagged as a change of
    speaker, also when the marker introduced only an annotation (">> [laughter]"
    followed by unmarked words). The speaker's own laughter, sighs and similar
    sounds become Irodori style emojis (caption_annotations). Annotation-only
    parts are kept unchanged, so grouping still treats them as boundaries.
    """
    result, changed, pending_label = [], False, None
    for start, end, part, marked, label in speaker_parts(cues):
        text, _ = voiced_annotations(part)
        if marked:
            pending_label = label
        changed = changed or marked
        if text:
            result.append(Cue(start, end, text, speaker_change=changed, speaker_label=pending_label))
            changed, pending_label = False, None
        elif part.strip():
            result.append(Cue(start, end, part))
    return result


def select_track(info: dict, language: str = "auto") -> tuple[str, str, bool]:
    candidates = []
    original = (info.get("language") or "").lower().removesuffix("-orig")
    for automatic, key in [(False, "subtitles"), (True, "automatic_captions")]:
        tracks = info.get(key) or {}
        keys = [k for k in tracks if k != "live_chat"]
        if language != "auto":
            keys = [k for k in keys if k == language or k.startswith(language + "-")]
        for order, key in enumerate(keys):
            for track in tracks[key]:
                # Never silently use machine-translated subtitles as the original transcript.
                if track.get("ext") == "json3" and "tlang=" not in track.get("url", ""):
                    base = key.lower().removesuffix("-orig")
                    native = bool(original and (base == original or base.split("-")[0] == original.split("-")[0]))
                    # With no language metadata, preserve the uploader's track
                    # order. Alphabetical/English-first fallback selects an
                    # English translation ahead of Geekerwan's Chinese source.
                    rank = (not native, not key.endswith("-orig"), automatic, order)
                    candidates.append((rank, key, track["url"], automatic))
    if candidates:
        _, key, url, automatic = min(candidates)
        return key, url, automatic
    raise CaptionError("利用できる字幕がありません。字幕付き動画を選ぶか、ASRを設定してください。")


def _youtube_options(settings: Settings, client: str | None = None) -> dict:
    options = {
        "quiet": True, "no_warnings": True, "skip_download": True,
        "socket_timeout": 25, "retries": 2, "noplaylist": True,
        "extractor_args": {"youtube": {"skip": ["translated_subs"]}},
    }
    if client:
        options["extractor_args"]["youtube"]["player_client"] = [client]
    node = settings.root / "bin/node"
    if (settings.root / "distribution.json").is_file():
        if not node.is_file():
            raise CaptionError("動画取得ツールがアプリに含まれていません。アプリを再インストールしてください。")
        options["ffmpeg_location"] = str(settings.root / "bin")
    if node.is_file():
        options["js_runtimes"] = {"node": {"path": str(node)}}
    if settings.cookies_browser:
        options["cookiesfrombrowser"] = (settings.cookies_browser,)
    return options
