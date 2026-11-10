"""Streaming overlap assembly using synthetic context-sensitive inference.

Run with the worker Python and pytest to exercise the optional numpy/soundfile path.
"""
from __future__ import annotations

import io

import pytest

np = pytest.importorskip('numpy')
sf = pytest.importorskip('soundfile')

from transiroyt.mel_roformer.separation import SAMPLE_RATE, WINDOW, separate_stream  # noqa: E402


@pytest.mark.parametrize('length', [1, 1000, WINDOW - 1, WINDOW, WINDOW + 1,
                                    WINDOW * 2, 1323000, 1323001])
def test_streaming_keeps_every_sample_and_channels(length):
    audio = np.random.default_rng(4).normal(0, 0.05, (length, 2)).astype(np.float32)
    buffers = [io.BytesIO() for _ in range(3)]
    sf.write(buffers[0], audio, SAMPLE_RATE, format='WAV', subtype='FLOAT')
    buffers[0].seek(0)
    windows = []

    def forward(x):
        windows.append(x.shape)
        return x

    with sf.SoundFile(buffers[0]) as source, \
            sf.SoundFile(buffers[1], 'w', SAMPLE_RATE, 2, format='WAV', subtype='FLOAT') as vocals, \
            sf.SoundFile(buffers[2], 'w', SAMPLE_RATE, 2, format='WAV', subtype='FLOAT') as background:
        separate_stream(source, vocals, background, forward)
    for buffer in buffers[1:]:
        buffer.seek(0)
    voice, _ = sf.read(buffers[1], dtype='float32', always_2d=True)
    background, _ = sf.read(buffers[2], dtype='float32', always_2d=True)
    assert voice.shape == audio.shape
    assert set(windows) == {(1, 2, WINDOW)}
    np.testing.assert_allclose(voice, audio, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(voice + background, audio, rtol=1e-6, atol=1e-7)


def test_invalid_model_output_is_rejected(tmp_path):
    source_path = tmp_path / 'input.wav'
    sf.write(source_path, np.ones((1000, 2), dtype=np.float32), SAMPLE_RATE)
    with sf.SoundFile(source_path) as source, \
            sf.SoundFile(tmp_path / 'vocals.wav', 'w', SAMPLE_RATE, 2) as vocals, \
            sf.SoundFile(tmp_path / 'background.wav', 'w', SAMPLE_RATE, 2) as background:
        with pytest.raises(ValueError, match='invalid samples'):
            separate_stream(source, vocals, background, lambda x: x * np.nan)
