import pytest
from transiroyt.caption_annotations import strip_non_speech_annotations


@pytest.mark.parametrize("text", [
    "[MUSIC]", "[Applause]", "[laughter]", "(Music)", "［ＭＵＳＩＣ］", "（拍手）", "【音楽】",
    "[soft piano music playing]", "[Music fades out]", "[raucous applause]",
    "[audience laughs]", "[crowd cheering]", "[coughing]", "[sighs]", "[gasping]",
    "[thunder]", "[phone ringing]", "[door slams]", "[dog barking]", "[birds chirping]",
    "[footsteps]", "[inaudible]", "[unintelligible]", "[sound effects]",
    "[Music and Applause]", "[laughter, applause]", "[Music] [Applause]",
    "[音樂]", "[背景音乐]", "[掌聲]", "[笑声]", "[笑い声]", "[歓声]",
    "[música]", "[musique]", "[Musik]", "[risas]", "[박수]", "[음악]",
    "♪", "♫ ♬", "[♪♪]", "🎵", "[Music] ...", "MUSIC", "APPLAUSE", "LAUGHTER",
])
def test_known_sound_annotations_are_not_speech(text):
    assert strip_non_speech_annotations(text) == ""


@pytest.mark.parametrize("text,expected", [
    ("[Music] Hello there [applause]", "Hello there"),
    ("Hello[laughter]world", "Hello world"),
    ("Thank you [applause].", "Thank you."),
    ("[audience laughs] Welcome back.", "Welcome back."),
    ("[音乐] 今天介绍电脑。", "今天介绍电脑。"),
    ("♪ These are the lyrics ♪♪", "These are the lyrics"),
    ("music", "music"), ("Music", "Music"), ("I love MUSIC.", "I love MUSIC."),
    ("The music starts now.", "The music starts now."),
    ("[I love music]", "[I love music]"),
    ("[music is important]", "[music is important]"),
    ("(for example)", "(for example)"), ("[Alice] Hello", "[Alice] Hello"),
    ("[speaking French] Bonjour", "[speaking French] Bonjour"),
    ("[unknown annotation]", "[unknown annotation]"),
    ("[Music and dialogue]", "[Music and dialogue]"),
    ("[musical instruments]", "[musical instruments]"),
    ("[Music)", "[Music)"),
])
def test_dialogue_lyrics_and_unknown_brackets_are_preserved(text, expected):
    assert strip_non_speech_annotations(text) == expected
