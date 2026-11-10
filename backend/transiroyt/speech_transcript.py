"""Create subtitles with macOS speech recognition and review them with Whisper.

macOS supplies timing and transcript windows. Whisper reviews the words;
windows omitted by Whisper retain the macOS transcript."""
from __future__ import annotations

import difflib
import json
import logging
import platform
import re
from dataclasses import dataclass, field

from .captions import Cue, clean_text
from .process import ProcessError, run_process
from .subtitle_files import transcribed_cues
from .whisper_transcription import MODEL_IDENTITY as WHISPER_IDENTITY

logger = logging.getLogger(__name__)

SYSTEM_IDENTITY = f"apple-speechtranscriber:macos-{platform.mac_ver()[0]}"
UNSUPPORTED_LANGUAGE = 3
# Whisper reads 30 s; the padding on both sides stays inside that.
WINDOW_SECONDS = 29.0
WINDOW_PADDING = 0.2
PAUSE_SECONDS = 0.2
# A pause that only touches a word's edge does not belong to it.
PAUSE_OVERLAP = 0.02
# A longer clause is split at Apple's punctuation, else before its slowest
# word: Apple gives the time of a pause it did not punctuate to the next word.
CLAUSE_SECONDS = 10.0
# Apple's words Whisper left out are kept from this many tokens on (a skipped
# sentence); fewer are fillers (啊, 呢, um) Whisper leaves out on purpose.
KEPT_APPLE_TOKENS = 4
# Windows of the programme whose summed Whisper language probabilities decide
# the spoken language of a video without subtitles.
LANGUAGE_WINDOWS = 5
LANGUAGE_WINDOW_SECONDS = 30.0
# Review windows per Whisper request, between progress updates.
REVIEW_BATCH = 8

# One CJK character, a word of other letters or digits, or one symbol.
_CJK = r"\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uac00-\ud7af"
_TOKEN = re.compile(rf"[{_CJK}]|[^\W_{_CJK}]+(?:['’.][^\W_{_CJK}]+)*%?|[^\w\s]")
_PUNCTUATION = re.compile(r"[^\w\s]+")


class SpeechTranscriptionUnsupported(RuntimeError):
    """macOS has no speech transcriber for the language."""


@dataclass
class Token:
    key: str
    # With the space before a word, where the transcript had one.
    text: str
    start: float
    end: float


@dataclass
class Clause:
    tokens: list[Token] = field(default_factory=list)
    # Reviewed text; None until Whisper's words are placed.
    words: list[str] | None = None

    @property
    def start(self):
        return self.tokens[0].start

    @property
    def end(self):
        return self.tokens[-1].end

    @property
    def text(self):
        return clean_text("".join(self.words if self.words is not None else (t.text for t in self.tokens)))


async def system_transcript(binary, wav, language: str, *, timeout: float) -> dict:
    """{"locale", "segments": [{"start", "end", "text", "words": [{"start", "end", "word", "confidence"}]}]}."""
    try:
        return json.loads(await run_process(str(binary), str(wav), language, timeout=timeout))
    except ProcessError as exc:
        if exc.returncode == UNSUPPORTED_LANGUAGE:
            raise SpeechTranscriptionUnsupported(str(exc).strip()) from exc
        raise


def tokens(text: str) -> list[tuple[str, str]]:
    """(comparison key, text) per token; punctuation stays with the token before it."""
    result = []
    for match in _TOKEN.finditer(text):
        word = match.group()
        if _PUNCTUATION.fullmatch(word):
            if result:
                result[-1] = (result[-1][0], result[-1][1] + word)
            continue
        space = " " if match.start() and text[match.start() - 1].isspace() else ""
        result.append((word.lower(), space + word))
    return result


def clauses(segments: list[dict], pauses: list[list[float]]) -> list[Clause]:
    """Apple's words in clauses split at pauses of the separated speech.

    Apple's word times are contiguous: the time of a pause usually goes to the
    word after it, sometimes to the word before, and between its results there
    is a gap. A clause ends where the pause starts; the next begins where it ends.
    """
    words = [token for segment in segments for token in _timed_tokens(segment["words"])]
    result, current, upcoming = [], [], iter(sorted(pauses))
    pause = next(upcoming, None)
    for token in words:
        # From the middle of the last word: a pause before that was placed before it.
        after = (current[-1].start + current[-1].end) / 2 if current else 0.0
        while pause is not None and pause[1] <= after:
            pause = next(upcoming, None)
        # A pause lasting past this word (a quiet 呢, "GPU") ends the clause after it.
        if (pause is not None and pause[1] < token.end
                and min(pause[1], token.end) - max(pause[0], after) > PAUSE_OVERLAP):
            if current:
                current[-1].end = max(current[-1].start, min(current[-1].end, pause[0]))
                result.extend(_split_long(current))
                current = []
            token.start = max(token.start, pause[1])
            pause = next(upcoming, None)
        current.append(token)
    result.extend(_split_long(current))
    return result


def _timed_tokens(words):
    result = []
    for word in words:
        parts = tokens(word["word"])
        if not parts and result:
            # Apple times its punctuation as a word of its own.
            result[-1].text += word["word"].strip()
        step = (word["end"] - word["start"]) / max(1, len(parts))
        for index, (key, text) in enumerate(parts):
            result.append(Token(key, text, word["start"] + index * step, word["start"] + (index + 1) * step))
    return result


def _split_long(tokens):
    """Split at the punctuation nearest the middle, else before the slowest word.

    Per character, so a number read out ("20%") is not taken for a pause.
    """
    if not tokens:
        return []
    if tokens[-1].end - tokens[0].start <= CLAUSE_SECONDS or len(tokens) < 2:
        return [Clause(tokens)]
    punctuated = [i for i in range(1, len(tokens)) if len(tokens[i - 1].text.strip()) > len(tokens[i - 1].key)]
    cut = (min(punctuated, key=lambda i: abs(i - len(tokens) / 2)) if punctuated
           else max(range(1, len(tokens)), key=lambda i: (tokens[i].end - tokens[i].start) / len(tokens[i].key)))
    return _split_long(tokens[:cut]) + _split_long(tokens[cut:])


def windows(items: list[Clause], duration: float) -> list[tuple[float, float, list[Clause]]]:
    """Consecutive clauses within 30 s of audio, padded a little where there is room."""
    groups = []
    for clause in items:
        if groups and clause.end - groups[-1][0].start <= WINDOW_SECONDS:
            groups[-1].append(clause)
        else:
            groups.append([clause])
    result = []
    for index, group in enumerate(groups):
        before = groups[index - 1][-1].end if index else 0.0
        after = groups[index + 1][0].start if index + 1 < len(groups) else duration
        result.append((max(before, group[0].start - WINDOW_PADDING),
                       min(after, group[-1].end + WINDOW_PADDING), group))
    return result


def review(group: list[Clause], heard: str) -> dict:
    """Place Whisper's words of a window into its clauses; return token counts.

    Matching tokens keep their clause; a misheard run is replaced in place, and
    words only Whisper heard join the clause before them.
    """
    apple = [(clause, token) for clause in group for token in clause.tokens]
    whisper = tokens(heard)
    for clause in group:
        clause.words = []
    counts = {"whisper": len(whisper), "apple_kept": 0}
    matcher = difflib.SequenceMatcher(None, [token.key for _, token in apple], [key for key, _ in whisper],
                                      autojunk=False)
    for operation, a1, a2, w1, w2 in matcher.get_opcodes():
        if operation == "delete":
            if a2 - a1 >= KEPT_APPLE_TOKENS:
                counts["apple_kept"] += a2 - a1
                for clause, token in apple[a1:a2]:
                    clause.words.append(token.text)
            continue
        for index in range(w1, w2):
            at = a1 + (index - w1) * (a2 - a1) // (w2 - w1) if operation != "insert" else max(0, a1 - 1)
            apple[at][0].words.append(whisper[index][1])
    return counts


def cues(items: list[Clause]) -> list[Cue]:
    return [Cue(clause.start, clause.end, clause.text) for clause in items if clause.text]


def language_windows(duration: float) -> list[tuple[float, float]]:
    """Several parts of the programme, so an intro with music or another language does not decide alone."""
    length = min(LANGUAGE_WINDOW_SECONDS, duration)
    return [(start, start + length) for start in sorted({
        round(max(0.0, min(duration - length, duration * (i + 0.5) / LANGUAGE_WINDOWS - length / 2)), 2)
        for i in range(LANGUAGE_WINDOWS)})]


async def reviewed_transcript(transcriber, recognized, speech, language: str, duration: float,
                              progress=lambda fraction, message: None) -> dict:
    """Subtitle cues from macOS's transcript (awaitable ``recognized``), each window re-heard by Whisper.

    ``speech`` is the separated voice at 16 kHz. Languages macOS does not support are transcribed
    by Whisper alone with its word times.
    """
    try:
        recognized = await recognized
    except SpeechTranscriptionUnsupported as exc:
        logger.warning("file dubbing: %s Whisper transcribes the speech alone", exc)
        progress(0.15, "macOS標準の音声認識が対応していない言語のため、Whisperで字幕を作成しています")
        timed = await transcriber.transcribe_timed(speech, language, timeout=600 + duration)
        return {"language": timed["language"], "source": f"{WHISPER_IDENTITY} (transcribed speech)",
                "cues": transcribed_cues(timed["segments"])}
    items = clauses(recognized["segments"], await transcriber.pauses(speech, PAUSE_SECONDS))
    planned = windows(items, duration)
    summary = {"windows": len(planned), "incomplete_windows": 0}
    for first in range(0, len(planned), REVIEW_BATCH):
        progress(0.15 + 0.3 * first / len(planned), f"Whisperで字幕を確認しています（{first}/{len(planned)}区間）")
        batch = planned[first:first + REVIEW_BATCH]
        heard = await transcriber.transcribe_windows(speech, [(a, b) for a, b, _ in batch], language)
        for (a, b, group), (text, complete) in zip(batch, heard, strict=True):
            if not complete:
                # A looping transcript is no review: the window keeps macOS's words.
                logger.warning("file dubbing Whisper review did not finish window=%.2f-%.2f", a, b)
                summary["incomplete_windows"] += 1
                continue
            for key, count in review(group, text).items():
                summary[key] = summary.get(key, 0) + count
    return {"language": language, "review": summary,
            "source": (f"{SYSTEM_IDENTITY} {recognized['locale']} (transcribed speech), "
                       f"reviewed by {WHISPER_IDENTITY}"),
            "cues": cues(items)}
