"""Prepare speaker-aligned subtitle sentences and their translation context."""
from __future__ import annotations

import json

from .captions import Cue
from .dubbing_options import LANGUAGE_CODE
from .errors import FileDubbingError
from .gemma_audio import text_part
from .sentence_captions import group_sentences
from .speaker_diarization import SpeakerLine

TURN_GAP_SECONDS = 1.5
# Sentences join the previous one of the same turn (up to 10 s, 0.3 s gap):
# one-word captions are too short to translate or voice on their own, and the
# dub voice reads about 5 morae a second, so a line has to reach this window
# before a Japanese sentence can be spoken clearly inside it. Of 91 captioned
# sentences, those with windows of 4 s or more were dubbed in 48 of 49 takes,
# those under 4 s in 15 of 42 (gemma-dub-loop: 16-25 characters passed 32 of 81
# takes, 26 or more 93 of 95).
# Irodori speaks at most 768 frames of 40 ms (30.72 s), and a translation is
# only condensed above 1.25x its time (dubbing_translation.CONDENSE_TEMPO): a
# longer unpunctuated run (a transcript, 30 s by default) was refused by the
# TTS ("exceeds model limit (duration)") and kept the original voice.
SENTENCE_MAX_SECONDS = 20.0
MIN_DUB_SECONDS = 4.0
SHORT_SENTENCE_SECONDS = 2.0
MERGED_MAX_SECONDS = 10.0
MERGE_GAP_SECONDS = 0.3
# Captioned reactions after the last line still belong to it for this long.
HEARD_AFTER_SECONDS = 3.0
EVENT_TOLERANCE = 0.05


def speaker_segments(lines: list[SpeakerLine]) -> list[tuple[str | None, Cue]]:
    """Sentences of one speaker's turn; utterances never span a change of speaker.

    Each sentence keeps its own anchor on the source clock, from its captions,
    so a long monologue stays in step with the picture. Translation still sees
    the neighbouring sentences as context.
    """
    result, run = [], []

    def flush():
        if not run:
            return
        cues = [Cue(a, b, text) for line in run for a, b, text in line.cues()]
        cues[0].boundary_before = run[0].boundary_before
        merged: list[Cue] = []
        for sentence in group_sentences(cues, max_seconds=SENTENCE_MAX_SECONDS):
            if merged and not sentence.boundary_before:
                previous = merged[-1]
                short = (previous.end - previous.start < MIN_DUB_SECONDS
                         or sentence.end - sentence.start < SHORT_SENTENCE_SECONDS / 2)
                if (short and sentence.start - previous.end <= MERGE_GAP_SECONDS
                        and sentence.end - previous.start <= MERGED_MAX_SECONDS):
                    merged[-1] = Cue(previous.start, sentence.end, f"{previous.text} {sentence.text}",
                                     boundary_before=previous.boundary_before)
                    continue
            merged.append(sentence)
        result.extend((run[0].speaker, sentence) for sentence in merged)
        run.clear()

    for line in lines:
        if run and (line.boundary_before or line.speaker != run[-1].speaker
                    or line.start - run[-1].end > TURN_GAP_SECONDS):
            flush()
        run.append(line)
    flush()
    return result


async def detect_language(gemma, cues: list[Cue]) -> str:
    sample = " ".join(cue.text for cue in cues)[:150]
    result = await gemma.complete_json(
        [text_part("Identify the ORIGINAL language of the text in the subtitles field below. "
                   "Do not translate it. Ignore the language of these instructions and the JSON schema. "
                   "Reply with only a JSON object with key language holding a lowercase ISO 639-1 code "
                   "(for Chinese use zh-Hans or zh-Hant).\n"
                   + json.dumps({"subtitles": sample}, ensure_ascii=False))],
        {"type": "object", "properties": {"language": {"type": "string"}}, "required": ["language"]},
        max_tokens=40)
    language = str(result.get("language", "")).strip()
    if not LANGUAGE_CODE.fullmatch(language):
        raise FileDubbingError(f"字幕の言語を判定できませんでした（{language!r}）。言語コードを指定してください。")
    return language


def reactions_after(segments, events):
    """Reactions and sounds captioned from each line until the next one.

    Laughter or applause after a line tells the translator it was a joke or
    a highlight. Music is left out: it says nothing about the words.
    """
    result = []
    for number, segment in enumerate(segments):
        until = segments[number + 1].start if number + 1 < len(segments) else segment.end + HEARD_AFTER_SECONDS
        labels = []
        for event in events:
            # Saved speaker turns round times to 1 ms; an inline note starts with its line.
            if segment.start - EVENT_TOLERANCE <= event.start < until - EVENT_TOLERANCE:
                labels += [label for label, kind in zip(event.labels, event.kinds, strict=True)
                           if kind != "music" and label not in labels]
        result.append(tuple(labels))
    return result
