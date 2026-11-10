import pytest
from transiroyt.captions import Cue
from transiroyt.session_models import Segment
from transiroyt.speech_timing import MAX_LAG, speech_end, speech_window


def test_slack_recovers_at_original_anchors_and_cannot_accumulate():
    segments = [Segment(i, i * 4, i * 4 + 4, str(i)) for i in range(100)]
    for segment in segments:
        start, limit = speech_window(segments, segment)
        end = speech_end(segment, start, limit, 10)
        segment.playback_start, segment.playback_end, segment.state = start, end, 'ready'
        assert segment.start <= start <= segment.start + MAX_LAG
        assert segment.end <= end <= segment.end + MAX_LAG
        if segment.index:
            assert start >= segments[segment.index - 1].speech_end
    # A shorter utterance pays back the lag rather than continuing it forever.
    segment = segments[10]
    start, limit = speech_window(segments, segment)
    assert speech_end(segment, start, limit, 1) == segment.end


def test_silence_and_short_next_sentence_bound_slack():
    current = Segment(0, 0, 2, 'first')
    following = Segment(1, 3, 4, 'after silence')
    assert speech_window([current, following], current) == (0, 3)
    following.start, following.end = 2, 2.7
    assert speech_window([current, following], current) == (0, 4)


def test_music_boundary_is_kept_when_allowing_speech_slack():
    from transiroyt.sentence_captions import group_sentences
    cues = group_sentences([Cue(0, 2, 'Before.'), Cue(2, 2.1, '[Music]'), Cue(2.1, 5, 'After.')])
    segments = [Segment(i, c.start, c.end, c.text, boundary_before=c.boundary_before) for i, c in enumerate(cues)]
    assert segments[1].boundary_before
    assert speech_window(segments, segments[0]) == (0, 2)


def test_a_tiny_following_sentence_cannot_shorten_original_speech():
    first = Segment(0, 0, 2, 'first')
    second = Segment(1, 2, 2.1, 'second')
    assert speech_window([first, second], first) == pytest.approx((0, 4))


def test_music_boundary_reserves_each_preceding_sentence_budget():
    from transiroyt.speech_timing import prepare_speech_limits
    segments = [Segment(i, i * 2, i * 2 + 2, str(i)) for i in range(4)]
    segments[3].boundary_before = True
    prepare_speech_limits(segments)
    for segment in segments[:3]:
        start, limit = speech_window(segments, segment)
        end = speech_end(segment, start, limit, 4)
        segment.playback_start, segment.playback_end, segment.state = start, end, 'ready'
        assert end - start >= segment.end - segment.start
        assert end == segment.end  # No 7x catch-up at the final short sentence.


def test_silence_repays_lag_and_video_end_is_reserved():
    from transiroyt.speech_timing import prepare_speech_limits
    segments = [Segment(0, 0, 3, 'first'), Segment(1, 3, 5, 'second'), Segment(2, 8, 10, 'last')]
    prepare_speech_limits(segments, 10)
    assert segments[0].speech_limit == 5
    assert segments[1].speech_limit == 7
    assert segments[2].speech_limit == 10


def test_reserved_windows_never_make_a_fast_sentence_faster_than_its_original_slot():
    import random

    from transiroyt.speech_timing import prepare_speech_limits
    rng = random.Random(42)
    segments = []
    position = 0
    for i in range(100):
        duration = rng.uniform(.2, 8)
        segments.append(Segment(i, position, position + duration, str(i), boundary_before=i % 13 == 0))
        position += duration + (rng.uniform(.2, 3) if i % 7 == 0 else 0)
    prepare_speech_limits(segments, position)
    for segment in segments:
        natural = rng.uniform(.5, 15)
        start, limit = speech_window(segments, segment)
        end = speech_end(segment, start, limit, natural)
        segment.playback_start, segment.playback_end, segment.state = start, end, 'ready'
        assert end - segment.end <= MAX_LAG + 1e-9
        assert natural / (end - start) <= max(1.35, natural / (segment.end - segment.start)) + 1e-9
        if segment.index:
            assert start >= segments[segment.index - 1].speech_end
