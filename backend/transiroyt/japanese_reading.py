"""Latin readings of Japanese text from macOS (CFStringTokenizer), for comparing speech.

Whisper writes a heard word in its own script: 玄界 as 限界, アンドロイド as
Android. Comparing readings ("genkai") instead of characters keeps such
spellings from counting as garbled speech. The reading of a lone homophone is
still the tokenizer's guess (角 -> "tsuno" where 核 was spoken).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import difflib
import re
import unicodedata

_UTF8 = 0x08000100
_LATIN_TRANSCRIPTION = 1 << 16


class _Range(ctypes.Structure):
    _fields_ = [("location", ctypes.c_long), ("length", ctypes.c_long)]


_cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
_cf.CFStringCreateWithBytes.restype = ctypes.c_void_p
_cf.CFStringCreateWithBytes.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32,
                                        ctypes.c_bool]
_cf.CFLocaleCreate.restype = ctypes.c_void_p
_cf.CFLocaleCreate.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_cf.CFStringTokenizerCreate.restype = ctypes.c_void_p
_cf.CFStringTokenizerCreate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, _Range, ctypes.c_ulong, ctypes.c_void_p]
_cf.CFStringTokenizerAdvanceToNextToken.restype = ctypes.c_ulong
_cf.CFStringTokenizerAdvanceToNextToken.argtypes = [ctypes.c_void_p]
_cf.CFStringTokenizerCopyCurrentTokenAttribute.restype = ctypes.c_void_p
_cf.CFStringTokenizerCopyCurrentTokenAttribute.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
_cf.CFStringGetLength.restype = ctypes.c_long
_cf.CFStringGetLength.argtypes = [ctypes.c_void_p]
_cf.CFStringGetCString.restype = ctypes.c_bool
_cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
_cf.CFRelease.argtypes = [ctypes.c_void_p]


def _string(text: str):
    data = text.encode()
    return _cf.CFStringCreateWithBytes(None, data, len(data), _UTF8, False)


_JAPANESE = _cf.CFLocaleCreate(None, _string("ja"))


def reading(text: str) -> str:
    """Lowercase ASCII letters of the text's reading; digits and symbols are dropped."""
    source = _string(text)
    tokenizer = _cf.CFStringTokenizerCreate(None, source, _Range(0, _cf.CFStringGetLength(source)), 0, _JAPANESE)
    parts = []
    try:
        while _cf.CFStringTokenizerAdvanceToNextToken(tokenizer):
            token = _cf.CFStringTokenizerCopyCurrentTokenAttribute(tokenizer, _LATIN_TRANSCRIPTION)
            if not token:
                continue
            buffer = ctypes.create_string_buffer(4 * _cf.CFStringGetLength(token) + 1)
            _cf.CFStringGetCString(token, buffer, len(buffer), _UTF8)
            _cf.CFRelease(token)
            parts.append(buffer.value.decode())
    finally:
        _cf.CFRelease(tokenizer)
        _cf.CFRelease(source)
    # "pēji" -> "peji": long-vowel marks are spelling, not a different sound here.
    plain = unicodedata.normalize("NFKD", "".join(parts)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", plain.lower())


# A run of Latin letters or katakana (with digits and URL punctuation inside):
# one word or name, which Whisper may write in either script.
_FOREIGN = re.compile(r"[A-Za-z0-9ァ-ヺー・][A-Za-z0-9ァ-ヺー・.\-/:'’ ]*[A-Za-z0-9ァ-ヺー]|[A-Za-zァ-ヺ]")
_LETTER = re.compile(r"[A-Za-zァ-ヺ]")
# Spellings of one sound in English and in romanized katakana.
_SOUNDS = [("ph", "f"), ("th", "s"), ("sh", "s"), ("ch", "C"), ("ck", "k"), ("c", "k"), ("q", "k"), ("x", "ks"),
           ("v", "b"), ("l", "r"), ("j", "z")]
# Below this the two spellings are different words ("Amazon" heard as "Google").
SAME_WORD_SKELETON = 0.6


def _skeleton(span: str) -> str:
    """Consonants of a word as spoken: "software" and ソフトウェア are both close to "sft"."""
    text = re.sub(r"(?<=[A-Za-z0-9])\.(?=[A-Za-z])", " dot ", span)
    text = re.sub(r"[ァ-ヺー]+", lambda m: reading(m.group()), text.replace("/", " slash ")).lower()
    text = re.sub(r"[^a-z]", "", text)
    for spelled, sound in _SOUNDS:
        text = text.replace(spelled, sound)
    return re.sub(r"(.)\1+", r"\1", re.sub(r"[aeiouhwy]", "", text))


def _same_word(a: str, b: str) -> bool:
    x, y = _skeleton(a), _skeleton(b)
    if not x or not y:
        return x == y
    # Short skeletons agree by chance ("gns" Guinness, "nbs" ニューブス); a word starts alike.
    return x[0] == y[0] and difflib.SequenceMatcher(None, x, y, autojunk=False).ratio() >= SAME_WORD_SKELETON


def _tokens(text: str) -> list[tuple[str, bool]]:
    """(token, foreign): foreign words whole; other letters one by one, without punctuation and spaces."""
    tokens, last = [], 0
    for match in _FOREIGN.finditer(text):
        tokens += [(c, False) for c in text[last:match.start()] if c.isalnum()]
        if _LETTER.search(match.group()):
            tokens.append((match.group(), True))
        else:
            tokens += [(c, False) for c in match.group() if c.isalnum()]
        last = match.end()
    return tokens + [(c, False) for c in text[last:] if c.isalnum()]


def align_latin_spellings(expected: str, heard: str) -> tuple[str, str]:
    """The heard text with each name or English word respelled as in the expected line.

    Whisper writes a word in its own script and spelling: アマゾン・ドット・コム
    as amazon.com, Software as ソフトウェア, ナヴィエ・ストークス as Navie Stokes.
    A word is respelled only where it stands at the same place in both texts,
    one side is written in Latin letters, and their consonants agree, so a
    different word ("Amazon" heard as "Google") and every number still count.
    """
    texts = [unicodedata.normalize("NFKC", text) for text in (expected, heard)]
    if not any(re.search(r"[A-Za-z]", text) for text in texts):
        return texts[0], texts[1]
    want, got = _tokens(texts[0]), _tokens(texts[1])
    respelled = [token for token, _ in got]
    keys = [["\0" if foreign else token for token, foreign in side] for side in (want, got)]
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, *keys, autojunk=False).get_opcodes():
        a = [i for i in range(i1, i2) if want[i][1]]
        b = [j for j in range(j1, j2) if got[j][1]]
        # Pair words the alignment matched, or the one word each side has in a replaced stretch.
        if tag == "equal" or (tag == "replace" and len(a) == len(b) == 1):
            for i, j in zip(a, b, strict=True):
                word, spelling = want[i][0], got[j][0]
                if re.search(r"[A-Za-z]", word + spelling) and _same_word(word, spelling):
                    respelled[j] = word
    return "".join(token for token, _ in want), "".join(respelled)
