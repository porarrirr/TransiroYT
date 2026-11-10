from pathlib import Path

import pytest
from transiroyt.captions import Cue
from transiroyt.dubbing_captions import speaker_segments
from transiroyt.file_dubbing import FileDubber
from transiroyt.speaker_diarization import SpeakerDiarizer, SpeakerLine, speaker_turns
from transiroyt.speech_timing import prepare_speech_limits, speech_window
from transiroyt.subtitle_files import caption_events, parse_srt, spoken_cues

MRBEAST = '''1
00:00:22,990 --> 00:00:24,958
Flying cars that can take you anywhere.

2
00:00:25,025 --> 00:00:25,492
[Karl] What?

3
00:00:25,559 --> 00:00:26,727
[Jimmy] And so much more.

4
00:00:27,027 --> 00:00:27,861
No way.

5
00:00:27,928 --> 00:00:30,000
All starting with this $1 light,
'''


def test_mrbeast_named_turns_are_not_spoken_or_joined_with_dominant_speaker():
    parsed = parse_srt(MRBEAST)
    cues = spoken_cues(parsed)
    assert [(c.start, c.end, c.text, c.speaker_label) for c in cues] == [
        (22.990, 24.958, 'Flying cars that can take you anywhere.', None),
        (25.025, 25.492, 'What?', 'Karl'), (25.559, 26.727, 'And so much more.', 'Jimmy'),
        (27.027, 27.861, 'No way.', None), (27.928, 30, 'All starting with this $1 light,', None)]
    assert caption_events(parsed) == []
    turns = speaker_turns(cues)
    assert len(turns) == 5  # Sparse name labels do not absorb subsequent unmarked speakers.
    lines = [SpeakerLine(i, c.start, c.end, c.text, 'S1', boundary_before=c.speaker_change,
                         speaker_label=c.speaker_label) for i, c in enumerate(cues)]
    segments = speaker_segments(lines)
    assert [c.text for _, c in segments][:3] == [
        'Flying cars that can take you anywhere.', 'What?', 'And so much more.']
    assert lines[1].public()['speaker_label'] == 'Karl'
    assert lines[1].public()['boundary_before']
    timed, _ = FileDubber._segments(lines)
    prepare_speech_limits(timed, 30)
    assert timed[1].boundary_before and timed[2].boundary_before
    assert speech_window(timed, timed[0]) == (22.990, 24.958)
    assert speech_window(timed, timed[1]) == (25.025, 25.492)


@pytest.mark.parametrize('text,expected', [
    ('[Alice] Hello.', ['Hello.']),
    ('[Bob Smith] Hello.', ['Hello.']),
    ('[Karl] What? [Jimmy] More.', ['What?', 'More.']),
    ('>> [Karl] What?', ['What?']),
    ('[Music] Hello.', ['Hello.']),
    ('[unknown annotation] Hello.', ['[unknown annotation] Hello.']),
    ('[Music and dialogue] Hello.', ['[Music and dialogue] Hello.']),
    ('I met [Karl] yesterday.', ['I met [Karl] yesterday.']),
    ('(Karl) What?', ['(Karl) What?']),
])
def test_name_labels_use_caption_position_and_keep_unknown_brackets(text, expected):
    cues = spoken_cues([Cue(0, 3, text)])
    assert [c.text for c in cues] == expected
    assert cues[-1].end == 3


def test_label_before_nonverbal_sound_preserves_next_turn_and_sound_event():
    cues = [Cue(0, 1, 'Hello.'), Cue(1, 2, '[Karl] [laughter]'), Cue(2, 3, 'What?')]
    spoken = spoken_cues(cues)
    assert [(c.text, c.speaker_change, c.speaker_label) for c in spoken] == [
        ('Hello.', False, None), ('What?', True, 'Karl')]
    assert caption_events(cues)[0].labels == ['laughter']


async def test_named_turn_with_unmatched_ecapa_uses_own_voice_and_keeps_acoustic_evidence(monkeypatch):
    lines = [SpeakerLine(0, 0, 2, 'Cars.', 'S1', score=0.8, source='acoustic', acoustic_speaker='S1'),
             SpeakerLine(1, 2, 2.467, 'What?', 'S1', score=0.0705, source='acoustic',
                         acoustic_speaker='S1', boundary_before=True, speaker_label='Karl'),
             SpeakerLine(2, 3, 4.168, 'More.', 'S1', score=0.4675, source='acoustic',
                         acoustic_speaker='S1', boundary_before=True, speaker_label='Jimmy')]
    diarizer = SpeakerDiarizer(None)
    diarizer.item_of = {0: 0, 1: 1, 2: 2}
    diarizer.groups = {'S1': [0, 1, 2]}
    async def assign(*_):
        return lines, [lines[1], lines[2]]
    monkeypatch.setattr(diarizer, 'assign_voices', assign)
    result = await diarizer.run([], None, Path('.'))
    assert result[1].speaker is None
    assert result[1].acoustic_speaker == 'S1'
    assert result[1].source == 'caption_voice_unverified'
    assert result[2].speaker == 'S1'
    assert diarizer.groups == {'S1': [0, 2]}
    assert [c.text for _, c in speaker_segments(result)] == ['Cars.', 'What?', 'More.']
