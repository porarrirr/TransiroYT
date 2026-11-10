"""Bounded speech windows anchored to the source clock, never cumulative drift."""
from __future__ import annotations

MAX_LAG = 2.0
PREFERRED_TEMPO = 1.35
# A one-word reply squeezed between two lines ("Hit." in 0.3 s, C782A726)
# cannot fit at 1.25x; it may start this much earlier, in the silence after
# the previous line, instead of staying in the original voice.
EARLY_START = 0.2


def prepare_speech_limits(segments, video_end=None):
    """Reserve original speaking time before hard boundaries, backwards once.

    Borrowing two seconds must not squeeze the last short sentence before
    music or the end of the video into a fraction of its original window.
    """
    following = None
    for segment in reversed(segments):
        limit = segment.end + MAX_LAG
        if following is not None:
            if following.boundary_before:
                limit = min(limit, segment.end)
            else:
                limit = min(limit, following.speech_limit - (following.end - following.start))
                if following.start - segment.end > 0.15:
                    limit = min(limit, following.start)
        elif isinstance(video_end, (int, float)) and video_end >= segment.end:
            limit = min(limit, video_end)
        segment.speech_limit = max(segment.end, limit)
        following = segment


def speech_window(segments, segment):
    start = segment.start
    if segment.index:
        previous = segments[segment.index - 1]
        if previous.state == "ready":
            start = max(start, previous.speech_end)
    end = segment.speech_limit if segment.speech_limit is not None else segment.end + MAX_LAG
    if segment.index + 1 < len(segments):
        following = segments[segment.index + 1]
        if following.boundary_before:
            return start, max(start + 0.05, segment.end)
        gap = following.start - segment.end
        # Use silence, but preserve the next sentence's original speaking budget.
        next_limit = following.speech_limit if following.speech_limit is not None else following.end + MAX_LAG
        end = min(end, following.start if gap > 0.15 else next_limit - (following.end - following.start))
    return start, max(start + 0.05, end)


def earliest_start(segments, segment, start):
    """The earliest a too-long take may begin: never inside the previous line's original speech."""
    previous_end = segments[segment.index - 1].end if segment.index else 0.0
    return min(start, max(segment.start - EARLY_START, previous_end))


def speech_end(segment, start, limit, natural, *, preferred_tempo=PREFERRED_TEMPO):
    # Use slack only when needed. Short speech returns to its original anchor
    # immediately; dense speech can never move more than MAX_LAG seconds late.
    desired = max(segment.end, start + natural / preferred_tempo)
    return min(limit, desired)
