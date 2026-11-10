"""Approximate natural Japanese speaking length, for an explicit TTS duration."""
from __future__ import annotations

import unicodedata

from .caption_annotations import reaction_emojis, without_reaction_emojis

# Time for a voiced reaction emoji (laughter, sigh). Measured 2026-10-25 with
# the host's voice: without extra time Irodori
# hardly voiced laughter; of 0, 0.4 and 0.6 s, 0.4 s kept most lines intelligible.
REACTION_SECONDS = 0.4
_SMALL_KANA = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")
_SHORT_PAUSE = set("、，,・…")
_LONG_PAUSE = set("。．.！？!?")


def mora_count(text: str) -> float:
    """Kana are one mora (small kana zero), kanji average 1.8, Latin letters 0.6, digits 2."""
    total = 0.0
    for character in unicodedata.normalize("NFKC", text):
        if character in _SMALL_KANA:
            continue
        if "ぁ" <= character <= "ヿ":  # Hiragana, katakana, prolonged sound mark.
            total += 1
        elif unicodedata.name(character, "").startswith("CJK UNIFIED IDEOGRAPH"):
            total += 1.8
        elif character.isdigit():
            total += 2
        elif character.isalpha():
            total += 0.6
    return total


def pause_seconds(text: str) -> float:
    stripped = text.rstrip("。．.！？!?」』）) ")
    return (sum(c in _SHORT_PAUSE for c in stripped) * 0.15
            + sum(c in _LONG_PAUSE for c in stripped) * 0.3)


def natural_seconds(text: str, mora_per_second: float) -> float:
    words = without_reaction_emojis(text)
    return (mora_count(words) / mora_per_second + pause_seconds(words)
            + REACTION_SECONDS * len(reaction_emojis(text)))
