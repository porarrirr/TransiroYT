import pytest
from transiroyt.captions import Cue
from transiroyt.sentence_captions import group_sentences

EXAMPLE = [
    "Say you wanna drive", "from New York to San Francisco.",
    "How could you calculate the absolute shortest path?",
    "There are over 64 million intersections", "in the North American road system,",
    "so a simplified estimate gives around 10 to the 220 possible routes.",
    "Even if you could check a billion routes per second,", "testing all of them",
    "would still take over 10 to the 200 years.",
]


def example():
    return [Cue(i * 3, (i + 1) * 3, text) for i, text in enumerate(EXAMPLE)]


def test_user_example_is_four_complete_sentences_with_original_anchors():
    assert group_sentences(example()) == [
        Cue(0, 6, " ".join(EXAMPLE[:2])), Cue(6, 9, EXAMPLE[2]),
        Cue(9, 18, " ".join(EXAMPLE[3:6])), Cue(18, 27, " ".join(EXAMPLE[6:])),
    ]


def test_multiple_sentences_inside_a_cue_share_local_boundary_without_drift():
    cues = group_sentences([Cue(10, 20, "Hello. How are you?"), Cue(20, 22, "Fine!")])
    assert [c.text for c in cues] == ["Hello.", "How are you?", "Fine!"]
    assert cues[0].start == 10
    assert cues[0].end <= cues[1].start
    assert cues[1].end == 20
    assert cues[2] == Cue(20, 22, "Fine!")


@pytest.mark.parametrize("text", [
    "Dr. Smith paid 3.14 dollars.", "The U.S. road system is large.",
    "Use e.g. this route.", "J. Smith agrees.", 'He said "Go!"',
])
def test_abbreviations_decimals_and_quoted_endings(text):
    assert group_sentences([Cue(0, 4, text), Cue(4, 5, "Next.")]) == [
        Cue(0, 4, text), Cue(4, 5, "Next.")]


def test_music_and_silence_bound_unfinished_sentences():
    assert group_sentences([Cue(0, 1, "Before"), Cue(1, 1.1, "[Music]"),
                            Cue(1.1, 2, "after"), Cue(5, 6, "silence.")]) == [
        Cue(0, 1, "Before"), Cue(1.1, 2, "after"), Cue(5, 6, "silence.")]


def test_unpunctuated_long_transcript_is_bounded_and_keeps_every_word():
    source = [Cue(i, i + 1, f"word{i}") for i in range(100)]
    result = group_sentences(source)
    assert all(c.end - c.start <= 30 and len(c.text) <= 1200 for c in result)
    assert " ".join(c.text for c in result) == " ".join(c.text for c in source)
    assert result[-1].end == 100


def test_long_single_cue_without_spaces_is_bounded_without_losing_text():
    text = "字幕" * 1500
    cues = group_sentences([Cue(0, 90, text)])
    assert "".join(c.text for c in cues) == text
    assert all(c.end - c.start <= 30 and len(c.text) <= 1200 for c in cues)
    assert cues[0].start == 0 and cues[-1].end == 90
