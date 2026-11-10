"""Whether speech says its text: the Whisper transcript compared with the expected line.

Used for generated takes (accept or retake) and for reference clips of a speaker.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

from .japanese_reading import align_latin_spellings, reading
from .whisper_transcription import TranscriptOutputLimit

SPEECH_ACCEPT_RATIO = 0.75
# A take is also regenerated when half of any 20 letters of its line's reading
# were not heard: in a 57-minute dub, 30 of 214 first takes had such a garbled
# phrase (two confirmed by ear, at 1:04 and in #90), most passing the whole-line
# ratio.
SPEECH_COVERAGE = 0.5


def speech_verified(ratio, coverage):
    return ratio >= SPEECH_ACCEPT_RATIO and coverage >= SPEECH_COVERAGE


def normalized_words(text: str) -> str:
    # Punctuation, apostrophes and spacing are not speech ("I'm" == "im").
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


def transcript_ratio(expected: str, heard: str) -> float:
    a, b = normalized_words(expected), normalized_words(heard)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def transcript_coverage(expected: str, heard: str, width: int = 20) -> float:
    """The lowest share of heard reading in any ``width`` letters of a Japanese line's reading.

    The whole-line ratio hides a garbled phrase in a long take: "物理アドレスになる"
    heard as "ぶつけアドレス" still scored 0.84 over a 120-character line.
    Readings, not characters, so Whisper's own spelling (限界 for 玄界) is no error.
    """
    a, b = reading(expected), reading(heard)
    if len(a) < width:
        return 1.0
    heard_at = [False] * len(a)
    for start, _, size in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        heard_at[start:start + size] = [True] * size
    return min(sum(heard_at[i:i + width]) for i in range(len(a) - width + 1)) / width


# A run of letters, digits and number punctuation: "1,000", "4.9GHz", "A20", "8EE6".
_ALNUM = re.compile(r"[A-Za-z0-9][A-Za-z0-9.,/:]*[A-Za-z0-9]|[A-Za-z0-9]")
_DIGITS = re.compile(r"\d+(?:[.,/:]\d+)*")
_KANJI_DIGITS = dict(zip("〇一二三四五六七八九", range(10), strict=True))
_KANJI_UNITS = {"十": 10, "百": 100, "千": 1000}
_KANJI_SECTIONS = {"万": 10**4, "億": 10**8}
_SPOKEN_NUMBER = re.compile(r"\d+(?:[.,/:]\d+)*|[0-9〇一二三四五六七八九十百千万億]+")


def _kanji_number(text: str) -> int:
    """千 -> 1000, 1万 -> 10000, 二十 -> 20: how Whisper writes counts the line spelled in digits."""
    total = section = digits = 0
    for char in text:
        if char.isdigit() or char in _KANJI_DIGITS:
            digits = digits * 10 + (int(char) if char.isdigit() else _KANJI_DIGITS[char])
        elif char in _KANJI_UNITS:
            section += (digits or 1) * _KANJI_UNITS[char]
            digits = 0
        else:
            total += (section + digits or 1) * _KANJI_SECTIONS[char]
            section = digits = 0
    return total + section + digits


def numbers_heard(expected: str, heard: str) -> bool:
    """Every number of the line is in the transcript ("1/100" heard as "100万円" is not).

    Digits joined to Latin letters ("LPDDR6", "8E", "4.9GHz") belong to a word
    whose spelling Whisper varies (4W heard as 4万両) and are compared with it.
    """
    wanted = [run.replace(",", "") for run in _ALNUM.findall(unicodedata.normalize("NFKC", expected))
              if _DIGITS.fullmatch(run)]
    spoken = set()
    for number in _SPOKEN_NUMBER.findall(unicodedata.normalize("NFKC", heard)):
        spoken.add(number.replace(",", ""))
        if not re.fullmatch(r"[\d.,/:]+", number):
            spoken.add(str(_kanji_number(number)))
    return all(number in spoken for number in wanted)


def compare(expected: str, heard: str) -> tuple[float, float]:
    """(whole-line ratio, worst local coverage), with Whisper's spellings of names respelled.

    A respelled name can lift the score past a wrong number ("1/100" heard as
    "100万円" beside Navier-Stokes as ナビアストックス), and readings drop digits,
    so names are respelled only when every number of the line was heard.
    """
    if same_reading(expected, heard):
        return 1.0, 1.0
    comparison = align_latin_spellings(expected, heard)
    if not numbers_heard(expected, heard):
        comparison = expected, heard
    return transcript_ratio(*comparison), transcript_coverage(*comparison)


def same_reading(expected: str, heard: str) -> bool:
    """The transcript reads exactly as the line, with the same numbers: a spelling, not an error.

    Whisper writes じゃんけん as ジャンケン, モロイー as もろい, 哨戒艇 as 紹介亭;
    the characters differ (ratio 0.55-0.72) but the voice said the line. Only an
    identical reading counts: a similar one is not (katakana read as hiragana let
    garbled takes such as あやくさんサンタ pass). Readings drop digits, so the
    numbers must agree both ways. Over 4432 past takes this passed 20 failed
    ones, none garbled.
    """
    want = reading(expected)
    return bool(want) and want == reading(heard) and numbers_heard(expected, heard) and numbers_heard(heard, expected)


async def check_take(transcriber, wav, text: str) -> dict:
    """Transcribe a Japanese take and score it against its line."""
    try:
        heard = await transcriber.transcribe(wav, text)
    except TranscriptOutputLimit:
        # Unintelligible takes make the transcript loop: no complete transcript is a failed take.
        heard = ""
    ratio, coverage = compare(text, heard)
    return {"heard": heard, "ratio": ratio, "coverage": coverage, "verified": speech_verified(ratio, coverage)}
