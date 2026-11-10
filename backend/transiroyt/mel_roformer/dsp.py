# Copyright (c) 2026 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
"""Mel filterbank extracted from mlx-audio 0.6.5 (MIT)."""
import math

import mlx.core as mx


def mel_filters(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0,
    f_max: float | None = None,
    norm: str | None = None,
    mel_scale: str = "htk",
    precise: bool = False,
) -> mx.array:
    """Triangular mel filterbank as an mx.array of shape (n_mels, n_fft // 2 + 1).

    Args:
        precise: If True, compute the filterbank in float64 on the CPU stream
            and cast to float32 before returning. Default float32 path drifts
            ~5e-6 from a torchaudio float64 reference — enough to perturb the
            CTC decode in numerically sensitive models (e.g. granite_speech_nar).
            One-time cost at lru_cache miss; runtime use is unaffected.
    """

    def hz_to_mel(freq, mel_scale="htk"):
        if mel_scale == "htk":
            return 2595.0 * math.log10(1.0 + freq / 700.0)

        # slaney scale
        f_min, f_sp = 0.0, 200.0 / 3
        mels = (freq - f_min) / f_sp
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz) / logstep
        return mels

    def mel_to_hz(mels, mel_scale="htk"):
        if mel_scale == "htk":
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

        # slaney scale
        f_min, f_sp = 0.0, 200.0 / 3
        freqs = f_min + f_sp * mels
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        freqs = mx.where(
            mels >= min_log_mel,
            min_log_hz * mx.exp(logstep * (mels - min_log_mel)),
            freqs,
        )
        return freqs

    f_max = f_max or sample_rate / 2

    def _build(dtype):
        # generate frequency points

        n_freqs = n_fft // 2 + 1
        all_freqs = mx.linspace(0, sample_rate // 2, n_freqs, dtype=dtype)

        # convert frequencies to mel and back to hz

        m_min = hz_to_mel(f_min, mel_scale)
        m_max = hz_to_mel(f_max, mel_scale)
        m_pts = mx.linspace(m_min, m_max, n_mels + 2, dtype=dtype)
        f_pts = mel_to_hz(m_pts, mel_scale)

        # compute slopes for filterbank

        f_diff = f_pts[1:] - f_pts[:-1]
        slopes = mx.expand_dims(f_pts, 0) - mx.expand_dims(all_freqs, 1)

        # calculate overlapping triangular filters

        down_slopes = (-slopes[:, :-2]) / f_diff[:-1]
        up_slopes = slopes[:, 2:] / f_diff[1:]
        filterbank = mx.maximum(
            mx.zeros_like(down_slopes), mx.minimum(down_slopes, up_slopes)
        )

        if norm == "slaney":
            enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
            filterbank *= mx.expand_dims(enorm, 0)

        return filterbank.moveaxis(0, 1)

    if precise:
        # float64 isn't supported on the GPU stream — compute on CPU then cast.
        with mx.stream(mx.cpu):
            return _build(mx.float64).astype(mx.float32)
    return _build(mx.float32)



def stft(
    x,
    n_fft=800,
    hop_length=None,
    win_length=None,
    window: mx.array | str = "hann",
    center=True,
    pad_mode="reflect",
):
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    w = window

    if w.shape[0] < n_fft:
        padding = n_fft - w.shape[0]
        w = mx.pad(w, [(padding // 2, padding - padding // 2)])

    def _pad(x, padding, pad_mode="reflect"):
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        elif pad_mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix])
        else:
            raise ValueError(f"Invalid pad_mode {pad_mode}")

    if center:
        x = _pad(x, n_fft // 2, pad_mode)

    num_frames = 1 + (x.shape[0] - n_fft) // hop_length
    if num_frames <= 0:
        raise ValueError(
            f"Input is too short (length={x.shape[0]}) for n_fft={n_fft} with hop_length={hop_length} and center={center}."
        )

    shape = (num_frames, n_fft)
    strides = (hop_length, 1)
    frames = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(frames * w)


def istft(
    x,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    length=None,
    normalized=False,
):
    """Inverse Short-Time Fourier Transform.

    Args:
        x: Complex STFT output of shape (n_fft // 2 + 1, num_frames)
        hop_length: Hop length between frames (default: win_length // 4)
        win_length: Window length (default: FFT size inferred from the frequency axis)
        window: Window function name or array (default: "hann")
        center: If True, remove center padding (default: True)
        length: Target output length (default: None)
        normalized: If True, divide the overlap-add by the summed squared
            window (least-squares inversion, the division torch.istft always
            performs). If False, divide by the summed window (default: False).
            Note this is unrelated to torch.istft's `normalized` argument,
            which selects FFT scaling.

    Returns:
        Reconstructed time-domain signal
    """
    n_fft = (x.shape[0] - 1) * 2
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 4

    w = window

    if w.shape[0] < n_fft:
        padding = n_fft - w.shape[0]
        w = mx.pad(w, [(padding // 2, padding - padding // 2)])

    num_frames = x.shape[1]
    t = (num_frames - 1) * hop_length + n_fft

    reconstructed = mx.zeros(t)
    window_sum = mx.zeros(t)

    # inverse FFT of each frame
    frames_time = mx.fft.irfft(x, axis=0).transpose(1, 0)

    # get the position in the time-domain signal to add the frame
    frame_offsets = mx.arange(num_frames) * hop_length
    indices = frame_offsets[:, None] + mx.arange(n_fft)
    indices_flat = indices.flatten()

    updates_reconstructed = (frames_time * w).flatten()
    # Use window squared for COLA normalization (matches PyTorch's istft) or just window
    window_norm = (w * w) if normalized else w
    updates_window = mx.tile(window_norm, (num_frames,)).flatten()

    # overlap-add the inverse transformed frame, scaled by the window
    reconstructed = reconstructed.at[indices_flat].add(updates_reconstructed)
    window_sum = window_sum.at[indices_flat].add(updates_window)

    # normalize by the sum of (squared) window values
    # Avoid dividing by zero even on discarded samples: it produces NaN gradients.
    denominator = mx.where(window_sum > 1e-10, window_sum, 1.0)
    reconstructed = reconstructed / denominator

    start = n_fft // 2 if center else 0
    end = start + length if length is not None else t - start
    reconstructed = reconstructed[start:end]
    if length is not None and reconstructed.shape[0] < length:
        reconstructed = mx.pad(reconstructed, [(0, length - reconstructed.shape[0])])

    return reconstructed
