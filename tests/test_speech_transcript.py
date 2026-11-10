import sys

import pytest
from transiroyt.process import ProcessError
from transiroyt.speech_transcript import (
    SpeechTranscriptionUnsupported,
    clauses,
    cues,
    review,
    system_transcript,
    tokens,
    windows,
)


def characters(start, text, step=0.2):
    """Apple's per-character runs, contiguous like its transcripts."""
    return [{"start": start + i * step, "end": start + (i + 1) * step, "word": c, "confidence": 1.0}
            for i, c in enumerate(text)]


def test_tokens_split_cjk_characters_from_latin_words():
    # Unbroken Whisper text: a Latin word must not swallow the characters after it.
    assert tokens("苹果A20 Pro的能效，spec CPU2026来看看") == [
        ("苹", "苹"), ("果", "果"), ("a20", "A20"), ("pro", " Pro"), ("的", "的"), ("能", "能"),
        ("效", "效，"), ("spec", "spec"), ("cpu2026", " CPU2026"), ("来", "来"), ("看", "看"), ("看", "看")]
    assert tokens("It's 20%, isn't it?") == [("it's", "It's"), ("20%", " 20%,"), ("isn't", " isn't"), ("it", " it?")]


def test_clauses_end_at_pauses_that_apple_gives_to_the_next_word():
    words = characters(0.0, "这是超大核") + [{"start": 1.0, "end": 1.8, "word": "这", "confidence": 1.0}] \
        + characters(1.8, "是中核")
    items = clauses([{"words": words}], [[1.1, 1.6]])
    assert [(round(c.start, 2), round(c.end, 2), c.text) for c in items] == [
        (0.0, 1.0, "这是超大核"), (1.6, 2.4, "这是中核")]


def test_word_inside_a_pause_ends_the_clause_before_the_pause():
    # A quiet sentence-final 呢 measures as silence; it is still the clause's last word.
    words = characters(0.0, "为什么") + [{"start": 0.6, "end": 0.84, "word": "呢"},
                                       {"start": 0.84, "end": 1.08, "word": " ？"}] + characters(1.08, "今年")
    items = clauses([{"words": words}], [[0.62, 1.1]])
    assert [(round(c.start, 2), round(c.end, 2), c.text) for c in items] == [
        (0.0, 0.62, "为什么呢？"), (1.1, 1.48, "今年")]


def test_long_clause_splits_at_punctuation_else_before_the_slowest_word(monkeypatch):
    monkeypatch.setattr("transiroyt.speech_transcript.CLAUSE_SECONDS", 2.0)
    punctuated = characters(0.0, "一二三四") + [{"start": 0.8, "end": 1.0, "word": " ，"}] + characters(1.0, "五六七八九十")
    assert [c.text for c in clauses([{"words": punctuated}], [])] == ["一二三四，", "五六七八九十"]
    # A number read out lasts long, but per character it is no pause.
    unpunctuated = (characters(0.0, "一二三") + [{"start": 0.6, "end": 1.4, "word": " 20%"}]
                    + characters(1.4, "四五") + [{"start": 1.8, "end": 2.4, "word": "六"}] + characters(2.4, "七八"))
    assert [c.text for c in clauses([{"words": unpunctuated}], [])] == ["一二三 20%四五", "六七八"]


def test_windows_pack_clauses_within_whisper_input_and_pad_into_silence():
    items = clauses([{"words": characters(0.0, "一" * 50) + characters(12.0, "二" * 50)
                      + characters(24.0, "三" * 50)}], [[10.0, 12.0], [22.0, 24.0]])
    assert [(round(a, 2), round(b, 2), len(group)) for a, b, group in windows(items, 40.0)] == [
        (0.0, 22.2, 2), (23.8, 34.2, 1)]


def test_review_takes_whisper_words_and_keeps_sentences_it_skipped():
    items = clauses([{"words": characters(0.0, "这是它的超大河啊") + characters(1.6, "我们还是掏出参考价值来看")
                      + characters(4.0, "这是中和")}], [[1.6, 1.7], [4.0, 4.1]])
    counts = review(items, "这是它的超大核我们来看这是中核")
    # 超大河 -> 超大核 and 中和 -> 中核 come from Whisper, the filler 啊 is
    # dropped, and the words Whisper skipped keep Apple's.
    assert [c.text for c in items] == ["这是它的超大核", "我们还是掏出参考价值来看", "这是中核"]
    assert counts == {"whisper": 15, "apple_kept": 8}
    assert [(round(c.start, 2), round(c.end, 2), c.text) for c in cues(items)] == [
        (0.0, 1.6, "这是它的超大核"), (1.7, 4.0, "我们还是掏出参考价值来看"), (4.1, 4.8, "这是中核")]


def test_words_only_whisper_heard_join_the_clause_before_them():
    items = clauses([{"words": characters(0.0, "我们看") + characters(1.0, "好的")}], [[0.6, 1.0]])
    review(items, "我们来看 好的呢")
    assert [c.text for c in items] == ["我们来看", "好的呢"]


async def test_system_transcript_reports_unsupported_languages(tmp_path):
    binary = tmp_path / "system-transcribe"
    binary.write_text(f"#!{sys.executable}\n" + '''import json, sys
if sys.argv[2] == "xx":
    print("xxはmacOS標準の文字起こしに対応していません。", file=sys.stderr); sys.exit(3)
if sys.argv[2] == "broken":
    print("asset download failed", file=sys.stderr); sys.exit(1)
print(json.dumps({"locale": "zh-CN", "segments": [{"start": 0, "end": 1, "text": sys.argv[1], "words": []}]}))
''')
    binary.chmod(0o700)
    assert (await system_transcript(binary, tmp_path / "a.wav", "zh", timeout=10))["segments"][0]["text"] == str(
        tmp_path / "a.wav")
    with pytest.raises(SpeechTranscriptionUnsupported, match="xx"):
        await system_transcript(binary, tmp_path / "a.wav", "xx", timeout=10)
    with pytest.raises(ProcessError, match="asset download failed"):
        await system_transcript(binary, tmp_path / "a.wav", "broken", timeout=10)
