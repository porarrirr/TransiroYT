import array
import random
import sys

import pytest
from transiroyt.native_audio import frame_energies


@pytest.mark.parametrize("frame", [1, 320, 960])
@pytest.mark.parametrize("length", [0, 1, 319, 320, 961, 32001])
def test_energy_matches_integer_pcm_reference(frame, length):
    rng = random.Random(17)
    values = [rng.randrange(-32768, 32768) for _ in range(length)]
    samples = array.array("h", values)
    if sys.byteorder != "little":
        samples.byteswap()
    pcm = samples.tobytes()
    expected = [sum(x * x for x in values[i:i + frame]) for i in range(0, length, frame)]
    assert frame_energies(pcm, frame) == expected
    assert frame_energies(pcm, frame, partial=False) == expected[:length // frame]
    assert frame_energies(pcm, frame, start=3, stop=length - 2) == [
        sum(x * x for x in values[i:min(length - 2, i + frame)])
        for i in range(3, length - 2, frame)]


def test_full_scale_and_unaligned_pcm():
    assert frame_energies(b"\x00\x80\xff\x7f", 2) == [2147418113]
    with pytest.raises(ValueError):
        frame_energies(b"\x00", 320)
    with pytest.raises(ValueError):
        frame_energies(b"", 0)


def test_streamed_track_has_exact_silence_placement_and_truncation(tmp_path):
    import wave

    from transiroyt.dubbing_media import place_segments
    from transiroyt.session_models import Segment

    clips = tmp_path / "clips"
    clips.mkdir()
    for index, pcm in enumerate([b"\x00\x80" * 3, b"\xff\x7f" * 4]):
        with wave.open(str(clips / f"segment-{index}.wav"), "wb") as wav:
            wav.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
            wav.writeframes(pcm)
    first = Segment(0, 2 / 48000, 5 / 48000, "a")
    last = Segment(1, 8 / 48000, 12 / 48000, "b")
    target = tmp_path / "track.wav"
    place_segments([last, first], tmp_path, target, 10 / 48000)
    with wave.open(str(target)) as wav:
        assert wav.getnframes() == 10
        assert wav.readframes(10) == b"\x00" * 4 + b"\x00\x80" * 3 + b"\x00" * 6 + b"\xff\x7f" * 2


def test_hour_track_uses_bounded_memory(tmp_path):
    import tracemalloc
    import wave

    from transiroyt.dubbing_media import place_segments

    tracemalloc.start()
    try:
        place_segments([], tmp_path, tmp_path / "hour.wav", 3600)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2_000_000
    with wave.open(str(tmp_path / "hour.wav")) as wav:
        assert wav.getnframes() == 3600 * 48000
        wav.setpos(wav.getnframes() - 1)
        assert wav.readframes(1) == b"\x00\x00"
