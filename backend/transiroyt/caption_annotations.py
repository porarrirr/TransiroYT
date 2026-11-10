"""Conservative recognition of non-speech caption annotations.

Only known descriptions inside caption brackets are recognized. Ordinary words,
speaker names, unknown bracketed text and lyrics remain speech. See
caption_annotations.py for supported forms and intentional limits.

A recognized annotation is never read out as words. The speaker's own
non-verbal sounds ("[laughs]", "[sighs]") can instead be voiced: Irodori-TTS
v4.1 renders style emojis in its input text (irodori_tts/duration.py,
ALLOWED_ANNOTATION_EMOJIS). Audience reactions, music and sound effects are
not the speaker's voice; they are kept as events for context.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Match paired delimiters, never arbitrary text between mismatched brackets.
_BRACKETS = re.compile(
    r"\[[^\[\]()（）［］【】]*\]|\([^\[\]()（）［］【】]*\)"
    r"|［[^\[\]()（）［］【】]*］|（[^\[\]()（）［］【】]*）"
    r"|【[^\[\]()（）［］【】]*】"
)
_NOTES = re.compile(r"[♪♫♬♩🎵🎶]\ufe0f?")
_BARE_MARKERS = re.compile(r"(?:MUSIC|APPLAUSE|LAUGHTER)(?:\s+(?:MUSIC|APPLAUSE|LAUGHTER))*")

# Deliberately bounded grammar: matching a keyword anywhere in brackets would
# erase dialogue such as [I love music] or [music is important].
_MUSIC = (
    r"(?:(?:background|instrumental|soft|gentle|quiet|loud|upbeat|dramatic|tense|"
    r"suspenseful|ominous|sad|happy|cheerful|classical|piano|guitar|orchestral|frantic) )*"
    r"music(?: (?:plays|playing|continues|starts|stops|ends|fades(?: in| out)?))?"
)
# The speaker's own voice, by the emoji Irodori renders for it.
_VOCAL = {
    "laughter": r"laughter|laughing|laughs|chuckling|chuckles|giggling|giggles|snickering|snickers",
    "sigh": r"sighing|sighs|exhaling|exhales",
    "gasp": r"gasping|gasps",
    "scream": r"screaming|screams|shrieking|shrieks",
    "crying": r"sobbing|sobs|crying|cries",
    "cough": (r"coughing|coughs|clears throat|clearing throat|snorting|snorts|sniffing|sniffs|"
              r"sniffling|sniffles|sneezing|sneezes"),
}
EMOJI = {"laughter": "🤭", "sigh": "😮‍💨", "gasp": "😮", "scream": "😱", "crying": "😭", "cough": "🤧"}
# Several people, or the audience: not the voice of the person speaking.
_GROUP = r"(?:audience|crowd|people|children|all|everyone|both)"
_REACTIONS = (
    r"cheering|cheers|clapping|applause|raucous applause|groaning|groans|whistling|booing|"
    r"indistinct chatter|crowd noise"
)
_SOUNDS = (
    r"(?:sound effects?|sfx|silence|inaudible|unintelligible|background noise|thunder|thunder rumbling|"
    r"rain|wind blowing|footsteps|ring|knock|knocking|bark|barking|"
    r"dogs? barking|birds? chirping|phone ringing|telephone ringing|bell ringing|"
    r"doorbell ringing|door (?:slams|slamming|creaks|creaking)|engine revving|"
    r"alarm (?:rings|ringing)|beeping|bleep|bleeping|explosion|gunshots?)"
)
_VOCAL_PATTERN = re.compile("|".join(
    rf"(?P<{label}>(?:(?:man|woman|boy|girl|he|she) )?(?:{words}))" for label, words in _VOCAL.items()))
_GROUP_PATTERN = re.compile(rf"{_GROUP} (?:{'|'.join(_VOCAL.values())}|{_REACTIONS})|{_REACTIONS}")
_MUSIC_PATTERN = re.compile(_MUSIC)
_SOUND_PATTERN = re.compile(_SOUNDS)
_LOCALIZED = {
    "音楽": "music", "音乐": "music", "音樂": "music", "背景音乐": "music", "背景音樂": "music",
    "música": "music", "musique": "music", "musik": "music", "음악": "music",
    "笑い": "laughter", "笑い声": "laughter", "笑声": "laughter", "笑聲": "laughter",
    "risas": "laughter", "rires": "laughter", "lachen": "laughter", "웃음": "laughter",
    "拍手": "applause", "掌声": "applause", "掌聲": "applause", "aplausos": "applause",
    "applaudissements": "applause", "applaus": "applause", "박수": "applause",
    "歓声": "cheering", "欢呼": "cheering", "歡呼": "cheering", "笑い声と拍手": "audience",
    "効果音": "sound", "無音": "sound", "聞き取り不能": "sound",
}


@dataclass(frozen=True)
class Annotation:
    """One recognized description. ``kind`` is vocal, audience, music or sound."""

    label: str
    kind: str

    @property
    def emoji(self) -> str | None:
        return EMOJI.get(self.label) if self.kind == "vocal" else None


def _classify_part(part: str) -> Annotation | None:
    localized = _LOCALIZED.get(part)
    if localized:
        if localized in EMOJI:
            return Annotation(localized, "vocal")
        kind = "music" if localized == "music" else "sound" if localized == "sound" else "audience"
        return Annotation(localized, kind)
    if _MUSIC_PATTERN.fullmatch(part):
        return Annotation("music", "music")
    match = _VOCAL_PATTERN.fullmatch(part)
    if match:
        return Annotation(match.lastgroup, "vocal")
    if _GROUP_PATTERN.fullmatch(part):
        return Annotation(part, "audience")
    if _SOUND_PATTERN.fullmatch(part):
        return Annotation(part, "sound")
    return None


def classify(text: str) -> list[Annotation] | None:
    """The annotations described by bracket contents, or None for anything else."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    normalized = normalized.strip(" .!…")
    single = _classify_part(normalized)
    if single:
        return [single]
    # Multiple sound descriptions can share a single pair of brackets.
    parts = re.split(r"\s*(?:,|&|/|\+|\band\b)\s*", normalized)
    if len(parts) < 2:
        return None
    found = [_classify_part(part) for part in parts]
    return found if all(found) else None


def _render(text: str, *, voiced: bool) -> tuple[str, list[Annotation]]:
    found: list[Annotation] = []

    def replace(match: re.Match) -> str:
        annotations = classify(match.group()[1:-1])
        if annotations is None:
            return match.group()
        found.extend(annotations)
        emojis = "".join(a.emoji for a in annotations if a.emoji) if voiced else ""
        # Do not glue words on either side of an annotation.
        return f" {emojis} " if emojis else " "

    text = _BRACKETS.sub(replace, text)
    text, notes = _NOTES.subn("", text)
    text = " ".join(text.split())
    if found:
        text = re.sub(r"\s+([,.;:!?。！？、])", r"\1", text)
    # Bare uppercase labels are conventional; lowercase 'music' can be speech.
    if _BARE_MARKERS.fullmatch(text):
        found.extend(_classify_part(word.casefold()) for word in text.split())
        return "", found
    # Words are what make a caption speech; an emoji alone is not spoken.
    if (found or notes) and not any(char.isalnum() for char in text):
        return "", found
    return text, found


def strip_non_speech_annotations(text: str) -> str:
    """Remove known sound labels and note symbols, preserving actual words."""
    return _render(text, voiced=False)[0]


def voiced_annotations(text: str) -> tuple[str, list[Annotation]]:
    """Speech with the speaker's own non-verbal sounds as Irodori emojis.

    Other annotations are removed. Text without words is empty: a caption that
    is only "[laughter]" is not a line to dub. Every recognized annotation is
    returned so callers can use reactions and music as context.
    """
    return _render(text, voiced=True)


# Longest first: "😮‍💨" (sigh) begins with "😮" (gasp).
REACTION_EMOJI = re.compile("|".join(sorted(map(re.escape, EMOJI.values()), key=len, reverse=True)))


def reaction_emojis(text: str) -> list[str]:
    """The voiced reaction emojis in a line, in order."""
    return REACTION_EMOJI.findall(text)


def without_reaction_emojis(text: str) -> str:
    text = " ".join(REACTION_EMOJI.sub(" ", text).split())
    return re.sub(r"\s+([,.;:!?。！？、])", r"\1", text)
