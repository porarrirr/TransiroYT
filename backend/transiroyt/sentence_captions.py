"""Sentence-first captions with independent anchors on the original media clock."""
from __future__ import annotations

import math
import re
from bisect import bisect_left, bisect_right

from .caption_annotations import strip_non_speech_annotations
from .captions import Cue

_TERMINAL = re.compile(r'''[.!?。！？]+["'”’」』）)]*''')
_ABBREVIATION = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|Fig|No)|\b[A-Za-z]|\b(?:[A-Za-z]\.)+[A-Za-z])\.$", re.I)


def sentence_ends(text: str) -> list[int]:
    """Conservative punctuation boundaries; decimals and common initials stay intact."""
    ends = []
    for match in _TERMINAL.finditer(text):
        token = match.group()
        if token.startswith("."):
            # An internal decimal/acronym/domain dot is not a sentence boundary.
            if match.end() < len(text) and not text[match.end()].isspace():
                continue
            if _ABBREVIATION.search(text[:match.start() + 1]):
                continue
            if text[max(0, match.start() - 2):match.end()].endswith(("e.g.", "i.e.")):
                continue
            if token.startswith("..."):
                continue
        ends.append(match.end())
    return ends


def group_sentences(cues: list[Cue], max_seconds: float = 30, max_characters: int = 1200) -> list[Cue]:
    """Prefer complete sentences, with bounded fallback for unpunctuated transcripts.

    Original cue endpoints are exact. Inside a cue, character-proportional timing
    is an estimate, always relative to that cue, never to previous output audio.
    Music-only cues and silence over 1.5s are hard boundaries. A change of
    speaker ends the sentence without being a timing boundary.
    """
    if max_seconds <= 0 or max_characters < 1:
        raise ValueError("Sentence limits must be positive")
    result: list[Cue] = []
    run: list[Cue] = []
    run_boundary = False
    pending_boundary = False

    def flush():
        if not run:
            return
        first_result = len(result)
        text = ""
        anchors = []
        for cue in run:
            if text:
                text += " "
            anchors.append((len(text), len(text) + len(cue.text), cue))
            text += cue.text

        starts = [a for a, _, _ in anchors]
        ends = [b for _, b, _ in anchors]

        def timestamp(offset, *, end=False):
            index = bisect_left(ends, offset) if end else bisect_right(starts, offset) - 1
            first, last, cue = anchors[index]
            return cue.start + (cue.end - cue.start) * (offset - first) / (last - first)

        start = 0
        for boundary in sorted(set([*sentence_ends(text), len(text)])):
            words = list(re.finditer(r"\S+", text[start:boundary]))
            if not words:
                start = boundary
                continue
            if not any(char.isalnum() for char in text[start:boundary]):
                # A voiced reaction after a full stop ("Tails. 🤭") belongs to
                # the sentence before it, or else to the next one; never alone.
                if len(result) > first_result:
                    previous = result[-1]
                    result[-1] = Cue(previous.start, timestamp(boundary, end=True),
                                     f"{previous.text} {text[start:boundary].strip()}")
                    start = boundary
                continue
            first = start + words[0].start()
            last = first
            spans = []
            for word in words:
                a, b = start + word.start(), start + word.end()
                count = min(b - a, max(1, math.ceil((b - a) / max_characters),
                                      math.ceil((timestamp(b, end=True) - timestamp(a)) / max_seconds)))
                spans.extend((a + (b-a)*i//count, a + (b-a)*(i+1)//count) for i in range(count))
            for a, b in spans:
                if last > first and (b - first > max_characters
                                     or timestamp(b, end=True) - timestamp(first) > max_seconds):
                    result.append(Cue(timestamp(first), timestamp(last, end=True), text[first:last]))
                    first = a
                last = b
            result.append(Cue(timestamp(first), timestamp(last, end=True), text[first:last]))
            start = boundary
        if len(result) > first_result:
            result[first_result].boundary_before = run_boundary
        run.clear()

    for cue in cues:
        text = strip_non_speech_annotations(cue.text)
        if not text:
            flush()
            pending_boundary = True
            continue
        # A sentence never continues across a change of speaker (">>").
        if run and (cue.start - run[-1].end > 1.5 or cue.start < run[-1].end or cue.speaker_change):
            flush()
        if not run:
            run_boundary = pending_boundary or cue.boundary_before
            pending_boundary = False
        run.append(Cue(cue.start, cue.end, text))
    flush()
    return result
