"""Stream the approved 7.7 s / 0.25 s overlap windows with bounded memory."""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 44100
WINDOW = round(7.7 * SAMPLE_RATE)
OVERLAP = round(0.25 * SAMPLE_RATE)


def separate_stream(source, vocals_out, background_out, forward):
    if source.samplerate != SAMPLE_RATE or source.channels != 2 or source.frames == 0:
        raise ValueError("separation input must be nonempty 44.1 kHz stereo")
    length = source.frames
    starts = list(range(0, max(1, length - WINDOW + 1), WINDOW - OVERLAP))
    tail_start = max(0, length - WINDOW)
    if starts[-1] != tail_start:
        starts.append(tail_start)
    pending = np.zeros((0, 2), dtype=np.float32)
    pending_weight = np.zeros(0, dtype=np.float32)
    for i, start in enumerate(starts):
        valid = min(WINDOW, length - start)
        source.seek(start)
        mixture = source.read(valid, dtype="float32", always_2d=True)
        part = np.pad(mixture.T, ((0, 0), (0, WINDOW - valid)))
        voice = np.asarray(forward(part[None]))[0, :, :valid].T
        if voice.shape != mixture.shape or not np.isfinite(voice).all():
            raise ValueError("separation produced invalid samples")
        window = np.ones(valid, dtype=np.float32)
        if i:
            overlap = min(valid, starts[i - 1] + WINDOW - start)
            window[:overlap] *= np.linspace(0, 1, overlap, dtype=np.float32)
        if i + 1 < len(starts):
            overlap = min(valid, start + valid - starts[i + 1])
            window[-overlap:] *= np.linspace(1, 0, overlap, dtype=np.float32)
        result = voice * window[:, None]
        weights = window.copy()
        result[:len(pending)] += pending
        weights[:len(pending_weight)] += pending_weight
        final = starts[i + 1] - start if i + 1 < len(starts) else valid
        output = result[:final] / weights[:final, None]
        vocals_out.write(output)
        background_out.write(mixture[:final] - output)
        # The end-aligned last window can overlap two preceding windows. Retain
        # the unnormalized sum and weights until all contributors have arrived.
        pending = result[final:].copy()
        pending_weight = weights[final:].copy()
