import math
import struct
import wave

import pytest
from transiroyt.audio import fit_audio, trim_generated_tail

RATE = 24000


def generated(path, parts):
    samples = []
    for duration, amplitude in parts:
        samples.extend(round(amplitude * math.sin(i * 440 * math.tau / RATE))
                       for i in range(round(duration * RATE)))
    pcm = struct.pack(f"<{len(samples)}h", *samples)
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        wav.writeframes(pcm)
    return pcm


def read(path):
    with wave.open(str(path), "rb") as wav:
        return wav.readframes(wav.getnframes())


def test_post_speech_silence_and_isolated_burst_are_not_part_of_natural_duration(tmp_path):
    path = tmp_path / "raw.wav"
    original = generated(path, [(0.7, 0), (2, 9000), (1.7, 0), (0.18, 18000), (0.8, 0)])
    result = trim_generated_tail(path)
    assert result["tail_cleanup"] == "isolated_tail"
    assert len(read(path)) / 2 / RATE == pytest.approx(2.82, abs=0.02)
    assert read(path)[:round(2.7 * RATE) * 2] == original[:round(2.7 * RATE) * 2]
    assert result["tail_removed_seconds"] > 2


@pytest.mark.parametrize("parts", [
    [(0.8, 0), (2, 9000)],  # Leading silence cannot end an utterance.
    [(2, 9000), (1.6, 0), (1, 9000), (0.8, 0)],  # Continuation after a long pause.
    [(2, 9000), (0.8, 0), (0.18, 9000), (0.8, 0)],  # An ordinary sentence pause.
    [(0.5, 9000), (1.6, 0), (0.18, 9000), (0.8, 0)],  # Early pause is not a tail.
    [(2, 9000), (1.6, 0), (0.18, 9000)],  # No terminal silence: keep the ending.
    [(2, 9000), (1.6, 0), (0.3, 9000), (0.8, 0)],  # Substantial final syllables stay.
    [(1, 0)],  # All-quiet output stays available for diagnosis.
])
def test_preserves_speech_and_interior_pauses(tmp_path, parts):
    path = tmp_path / "raw.wav"
    original = generated(path, parts)
    trim_generated_tail(path)
    # Only trailing silence may disappear, never any preceding signal or pause.
    signal_end = sum(duration for duration, _ in parts)
    if parts[-1][1] == 0 and any(amplitude for _, amplitude in parts):
        signal_end -= parts[-1][0]
    frames = round(signal_end * RATE)
    assert read(path)[:frames * 2] == original[:frames * 2]


async def test_cleaned_audio_fits_exactly_without_changing_source_speech(tmp_path):
    source, target = tmp_path / "raw.wav", tmp_path / "fit.wav"
    generated(source, [(2, 9000), (1.5, 0)])
    cleanup = trim_generated_tail(source)
    result = await fit_audio(source, target, 3)
    assert cleanup["tail_cleanup"] == "silence"
    assert result["natural_duration"] == pytest.approx(2.12, abs=0.02)
    with wave.open(str(target), "rb") as wav:
        assert wav.getnframes() == 3 * 48000


def test_cleanup_is_idempotent(tmp_path):
    path = tmp_path / "raw.wav"
    generated(path, [(2, 9000), (1.5, 0)])
    trim_generated_tail(path)
    first = path.read_bytes()
    assert trim_generated_tail(path)["tail_removed_seconds"] == 0
    assert path.read_bytes() == first
