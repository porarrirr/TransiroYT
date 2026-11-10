from __future__ import annotations

import array
import math
import sys
import wave
from pathlib import Path

from .native_audio import frame_energies
from .process import run_process

SAMPLE_RATE = 48000
# Silence kept around a take's voice by trim_edge_silence.
EDGE_LEAD_SECONDS = 0.03
EDGE_TAIL_SECONDS = 0.12


def trim_generated_tail(path: Path) -> dict:
    """Remove sustained terminal silence and an isolated post-silence burst.

    Work on the generated PCM, before measuring its natural speaking duration.
    Never close ordinary pauses or join audio across them. A burst is eligible
    only in the latter half, after 1.2s of silence, with at most 240ms of signal
    remaining and at least 500ms of terminal silence. Longer continuations stay.
    """
    with wave.open(str(path), "rb") as wav:
        params = wav.getparams()
        if params.nchannels != 1 or params.sampwidth != 2:
            return {"tail_removed_seconds": 0.0}
        pcm = wav.readframes(wav.getnframes())
        samples = array.array("h", pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    rate = params.framerate
    frame = max(1, round(rate * 0.02))
    # A conservative noise floor, plus 120ms of retained tail for quiet endings.
    # This is deliberately not a general-purpose speech detector.
    threshold = (32768 * 10 ** (-50 / 20)) ** 2
    active = [energy / min(frame, len(samples) - i * frame) > threshold
              for i, energy in enumerate(frame_energies(pcm, frame))]
    voiced = [i for i, value in enumerate(active) if value]
    if not voiced:  # Do not turn a failed/all-quiet synthesis into a tiny clip.
        return {"tail_removed_seconds": 0.0}
    last = voiced[-1] + 1
    end = len(samples)
    reason = "silence"
    if (len(samples) - last * frame) / rate >= 0.5:
        end = min(end, last * frame + round(rate * 0.12))
        # Search backwards, so an earlier in-sentence pause cannot hide a later
        # substantial continuation. Leading silence is never an endpoint.
        for right, left in zip(reversed(voiced[1:]), reversed(voiced[:-1]), strict=True):
            silence_start = left + 1
            if silence_start * frame < len(samples) / 2:
                break
            if (right - silence_start) * frame / rate < 1.2:
                continue
            if (last - right) * frame / rate <= 0.24:
                end = min(end, silence_start * frame + round(rate * 0.12))
                reason = "isolated_tail"
            break
    if end >= len(samples):
        return {"tail_removed_seconds": 0.0}
    trimmed = samples[:end]
    # End inside the retained silence, with a short fade to avoid a new click.
    fade = min(len(trimmed), round(rate * 0.005))
    for i in range(fade):
        trimmed[-fade + i] = round(trimmed[-fade + i] * (fade - i - 1) / fade)
    if sys.byteorder != "little":
        trimmed.byteswap()
    temporary = path.with_suffix(".tail.tmp.wav")
    try:
        with wave.open(str(temporary), "wb") as wav:
            wav.setparams(params)
            wav.writeframes(trimmed.tobytes())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"tail_removed_seconds": (len(samples) - end) / rate, "tail_cleanup": reason}


def trim_edge_silence(path: Path) -> float:
    """Cut silence before the first and after the last voiced 20 ms frame; return the seconds removed.

    A one-word take ("命中", 0.35 s of voice) comes with ~0.1 s of leading and
    ~0.2 s of trailing silence, which counted as speaking time: it could not fit
    a 0.45 s slot at 1.25x (C782A726). Uses trim_generated_tail's -50 dBFS floor
    and keeps the same 120 ms after the voice for quiet endings. Idempotent.
    """
    with wave.open(str(path), "rb") as wav:
        params = wav.getparams()
        pcm = wav.readframes(wav.getnframes())
        samples = array.array("h", pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    rate = params.framerate
    frame = max(1, round(rate * 0.02))
    threshold = (32768 * 10 ** (-50 / 20)) ** 2
    voiced = [i * frame for i, energy in enumerate(frame_energies(pcm, frame))
              if energy / min(frame, len(samples) - i * frame) > threshold]
    if not voiced:
        return 0.0
    start = max(0, voiced[0] - round(rate * EDGE_LEAD_SECONDS))
    end = min(len(samples), voiced[-1] + frame + round(rate * EDGE_TAIL_SECONDS))
    # Less than a frame is the frame grid, not silence: leave it, so a second pass changes nothing.
    start = start if start >= frame else 0
    end = end if len(samples) - end >= frame else len(samples)
    if start == 0 and end == len(samples):
        return 0.0
    trimmed = samples[start:end]
    fade = min(len(trimmed) // 2, round(rate * 0.005))
    for i in range(fade):  # No click at either new edge.
        trimmed[i] = trimmed[i] * i // fade
        trimmed[-1 - i] = trimmed[-1 - i] * i // fade
    if sys.byteorder != "little":
        trimmed.byteswap()
    temporary = path.with_suffix(".edges.tmp.wav")
    try:
        with wave.open(str(temporary), "wb") as wav:
            wav.setparams(params)
            wav.writeframes(trimmed.tobytes())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return (len(samples) - len(trimmed)) / rate


def atempo_chain(ratio: float) -> str:
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Invalid tempo")
    factors = []
    while ratio > 2:
        factors.append(2.0)
        ratio /= 2
    while ratio < 0.5:
        factors.append(0.5)
        ratio /= 0.5
    factors.append(ratio)
    return ",".join(f"atempo={factor:.9f}" for factor in factors)


async def fit_audio(source: Path, target: Path, duration: float) -> dict:
    """Pitch-preserving stretch followed by exact PCM sample-count alignment."""
    with wave.open(str(source), "rb") as wav:
        natural = wav.getnframes() / wav.getframerate()
    if natural <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("音声または字幕の長さが不正です。")
    # Avoid slowing short utterances excessively; fill the remaining window with silence.
    ratio = max(0.85, natural / duration)
    frames = round(duration * SAMPLE_RATE)
    filters = (f"{atempo_chain(ratio)},aresample={SAMPLE_RATE},"
               f"apad=whole_len={frames},atrim=end_sample={frames},"
               "afade=t=in:st=0:d=0.005,alimiter=limit=0.95:latency=1")
    temp = target.with_suffix(".tmp.wav")
    await run_process("ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                      "-i", str(source), "-af", filters, "-ac", "1", "-ar", str(SAMPLE_RATE),
                      "-c:a", "pcm_s16le", str(temp), timeout=60)
    with wave.open(str(temp), "rb") as wav:
        actual = wav.getnframes()
    if actual != frames:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"音声の時間調整に失敗しました: {actual} != {frames}")
    temp.replace(target)
    return {"natural_duration": natural, "duration": frames / SAMPLE_RATE,
            "tempo": ratio, "quality_warning": ratio > 1.65}
