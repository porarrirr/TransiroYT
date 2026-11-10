# Copyright (c) 2026 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""Mel-Band-RoFormer for vocal source separation.

Architecture (Kim Vocal 2, 228M params):
    [B, 2, samples] → STFT → CaC interleave → BandSplit → 6× DualAxis →
    MaskEstimate → complex multiply → iSTFT → [B, 2, samples]

Reference implementations:
    - lucidrains/BS-RoFormer (MIT)
    - ZFTurbo/Music-Source-Separation-Training (MIT)
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import MelRoFormerConfig


class RMSNorm(nn.Module):
    """RMSNorm that matches ZFTurbo's ``F.normalize(x, dim=-1) * sqrt(dim) * gamma``.

    ``mlx.nn.RMSNorm`` uses additive ``eps=1e-5`` which diverges from PyTorch
    ``F.normalize`` (``eps=1e-12``, applied as ``max(||x||, eps)``) when input
    magnitudes are small — which happens at high-frequency STFT bins, so the
    band-split's per-band RMSNorm is where the port silently diverged.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.weight = mx.ones((dim,))

    def __call__(self, x):
        full = x.astype(mx.float32)
        norm = mx.sqrt(mx.sum(full * full, axis=-1, keepdims=True))
        return ((full / mx.maximum(norm, 1e-12)) * self.scale * self.weight).astype(x.dtype)




# ---------- Mel Filterbank ----------


class MelFilterbank:
    """Computes mel-scale frequency band boundaries and gather indices.

    Uses Slaney mel scale (librosa-compatible). Produces binarized triangular
    filters, then computes stereo-interleaved gather indices per band.
    """

    def __init__(self, config: MelRoFormerConfig):
        self.config = config
        fb = self._build_mel_filterbank()
        self.freq_indices, self.band_dims, self.num_bands_per_freq = (
            self._compute_band_info(fb)
        )

    def _build_mel_filterbank(self) -> np.ndarray:
        """Build the ZFTurbo-style binarized mel filterbank.

        Matches ZFTurbo/MSS-Training exactly: builds the Slaney-style
        triangular filterbank via :func:`mlx_audio.dsp.mel_filters`, force-
        assigns the DC bin to band 0 and the Nyquist bin to the last band
        (the underlying triangular filters otherwise leave them zero, which
        breaks the "every freq covered" invariant), then binarizes.
        """
        from .dsp import mel_filters

        sr = self.config.sample_rate
        n_fft = self.config.n_fft
        n_mels = self.config.num_bands

        # dsp.mel_filters returns shape [n_mels, freq_bins] — same orientation
        # as librosa.filters.mel — so this drops in place.
        fb = np.asarray(
            mel_filters(sample_rate=sr, n_fft=n_fft, n_mels=n_mels, mel_scale="slaney")
        ).astype(np.float32)
        fb[0, 0] = 1.0
        fb[-1, -1] = 1.0
        return (fb > 0).astype(np.float32)

    def _compute_band_info(self, fb: np.ndarray):
        """Compute per-band gather indices for stereo CaC interleaving.

        Returns:
            freq_indices: list of np.ndarray — per-band gather indices into CaC spectrum
            band_dims: list of int — per-band input dimensions (for BandSplit projections)
            num_bands_per_freq: np.ndarray [freq_bins*2] — overlap count for scatter averaging
        """
        freq_bins = self.config.freq_bins
        num_bands = self.config.num_bands

        freq_indices = []
        num_bands_per_freq = np.zeros(freq_bins * 2, dtype=np.float32)

        for i in range(num_bands):
            # Find which frequency bins belong to this band
            bin_indices = np.where(fb[i] > 0)[0]
            if len(bin_indices) == 0:
                bin_indices = np.array([i])  # fallback

            # Stereo CaC interleave: each freq bin → 2 entries (L, R)
            cac_indices = []
            for b in bin_indices:
                cac_indices.extend([b * 2, b * 2 + 1])

            freq_indices.append(np.array(cac_indices, dtype=np.int32))
            for idx in cac_indices:
                if idx < len(num_bands_per_freq):
                    num_bands_per_freq[idx] += 1


        band_dims = [len(idx) * 2 for idx in freq_indices]  # ×2 for real/imag
        num_bands_per_freq = np.maximum(num_bands_per_freq, 1.0)

        # MLX does not support indexing with raw numpy arrays — convert once.
        freq_indices = [mx.array(idx) for idx in freq_indices]

        return freq_indices, band_dims, num_bands_per_freq

# ---------- RoFormer Attention ----------


class RoFormerAttention(nn.Module):
    """Multi-head attention with RoPE and per-head sigmoid gates."""

    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.norm = RMSNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_gates = nn.Linear(dim, heads, bias=True)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def prepare(self):
        self.packed_weight = mx.concatenate([self.to_q.weight, self.to_k.weight, self.to_v.weight])
        del self.to_q, self.to_k, self.to_v

    def __call__(self, x):
        batch, frames, _ = x.shape
        h = self.norm(x)
        packed = mx.matmul(h, self.packed_weight.T).reshape(batch, frames, 3, self.heads, self.dim_head)
        q, k, v = [packed[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]
        q = mx.fast.rope(q, dims=self.dim_head, traditional=True, base=10000, scale=1, offset=0)
        k = mx.fast.rope(k, dims=self.dim_head, traditional=True, base=10000, scale=1, offset=0)
        attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(self.dim_head))
        attended *= mx.sigmoid(self.to_gates(h)).transpose(0, 2, 1)[..., None]
        return self.to_out(attended.transpose(0, 2, 1, 3).reshape(batch, frames, -1))


# ---------- Feed-Forward Network ----------


class RoFormerFFN(nn.Module):
    """Feed-forward: RMSNorm → Linear → GELU → Linear.

    Weight key structure matches PyTorch Sequential indices:
        net.0 = RMSNorm, net.1 = Linear(expand), net.4 = Linear(compress)
    """

    def __init__(self, dim: int, ff_mult: int = 4):
        super().__init__()
        ff_dim = dim * ff_mult
        self.net = [
            RMSNorm(dim),  # index 0
            nn.Linear(dim, ff_dim),  # index 1
            None,  # index 2 (GELU activation, no params)
            None,  # index 3 (placeholder)
            nn.Linear(ff_dim, dim),  # index 4
        ]

    def __call__(self, x: mx.array) -> mx.array:
        h = self.net[0](x)  # RMSNorm
        h = self.net[1](h)  # Linear expand
        h = nn.gelu(h)  # GELU
        h = self.net[4](h)  # Linear compress
        return h


# ---------- Transformer Block ----------


class Transformer(nn.Module):
    """Single-axis transformer with RoPE attention + FFN + output RMSNorm."""

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, ff_mult: int):
        super().__init__()
        self.layers = [
            [RoFormerAttention(dim, heads, dim_head), RoFormerFFN(dim, ff_mult)]
            for _ in range(depth)
        ]
        self.norm = RMSNorm(dim)

    def __call__(self, x: mx.array) -> mx.array:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# ---------- BandSplit ----------


class BandSplit(nn.Module):
    """Split spectrogram into mel-scale bands and project each to model dimension."""

    def __init__(self, config: MelRoFormerConfig):
        super().__init__()
        self.filterbank = MelFilterbank(config)
        # Per-band projection: RMSNorm → Linear
        self.to_features = [
            [RMSNorm(bd), nn.Linear(bd, config.dim)] for bd in self.filterbank.band_dims
        ]

    def split(self, x: mx.array) -> mx.array:
        """Split CaC spectrogram into mel bands and project.

        Args:
            x: [B, freq_bins*2, T, 2] (CaC interleaved, real/imag)

        Returns:
            [B, T, num_bands, dim]
        """
        B, F2, T, _ = x.shape
        bands = []

        for indices, proj in zip(self.filterbank.freq_indices, self.to_features, strict=True):
            # Gather frequencies for this band: [B, num_band_freqs, T, 2]
            gathered = x[:, indices, :, :]
            # Flatten last dims: [B, T, num_band_freqs * 2]
            n_freqs = gathered.shape[1]
            band_input = gathered.transpose(0, 2, 1, 3).reshape(B, T, n_freqs * 2)
            # Project: RMSNorm → Linear → [B, T, dim]
            h = proj[0](band_input)  # RMSNorm
            h = proj[1](h)  # Linear
            bands.append(h)

        # Stack bands: [B, T, num_bands, dim]
        return mx.stack(bands, axis=2).astype(mx.float16)

    def merge(self, masks, frequency_count):
        batch, frames, _ = masks[0].shape
        filterbank = self.filterbank
        indices = mx.concatenate(filterbank.freq_indices)
        counts = mx.array(filterbank.num_bands_per_freq).reshape(1, -1, 1, 1)
        bands = [mask.astype(mx.float32).reshape(batch, frames, len(ids), 2).transpose(0, 2, 1, 3)
                 for ids, mask in zip(filterbank.freq_indices, masks, strict=True)]
        output = mx.zeros((batch, frequency_count, frames, 2), dtype=mx.float32)
        return output.at[:, indices, :, :].add(mx.concatenate(bands, axis=1)) / counts


# ---------- Mask Estimator ----------


class MaskEstimator(nn.Module):
    """Estimate complex masks per frequency band via MLP + GLU.

    Per-band MLP follows the ZFTurbo MLP helper:
        Linear(dim → hidden) → Tanh
        [Linear(hidden → hidden) → Tanh] × (mask_estimator_depth − 1)
        Linear(hidden → band_dim × 2)
        GLU

    So with ``mask_estimator_depth=d``, each band has (d + 1) linears.
    """

    def __init__(self, config: MelRoFormerConfig, band_dims: list):
        super().__init__()
        dim = config.dim
        mlp_hidden = config.mlp_hidden
        depth = config.mask_estimator_depth

        self.to_freqs = []
        for bd in band_dims:
            layers = [[nn.Linear(dim, mlp_hidden)]]
            for _ in range(depth - 1):
                layers.append([nn.Linear(mlp_hidden, mlp_hidden)])
            layers.append([nn.Linear(mlp_hidden, bd * 2)])
            self.to_freqs.append(layers)

    def prepare(self):
        bands = self.to_freqs
        self.weights = [mx.stack([band[i][0].weight for band in bands])
                        for i in range(len(bands[0]) - 1)]
        self.biases = [mx.stack([band[i][0].bias for band in bands])[:, None, :]
                       for i in range(len(bands[0]) - 1)]
        self.outputs = [band[-1][0] for band in bands]
        del self.to_freqs

    def __call__(self, x):
        batch, frames, bands, dim = x.shape
        h = x.transpose(2, 0, 1, 3).reshape(bands, batch * frames, dim)
        for weight, bias in zip(self.weights, self.biases, strict=True):
            h = mx.tanh(mx.matmul(h, weight.transpose(0, 2, 1)) + bias)
        result = []
        for i, layer in enumerate(self.outputs):
            out = layer(h[i]).reshape(batch, frames, -1)
            half = out.shape[-1] // 2
            result.append(out[..., :half] * mx.sigmoid(out[..., half:]))
        return result


# ---------- STFT / iSTFT ----------


def stft(audio: mx.array, n_fft: int, hop_length: int, window: mx.array) -> tuple:
    """Short-time Fourier Transform.

    Thin batched/multichannel adapter over :func:`mlx_audio.dsp.stft` which
    operates on 1D input. We loop over the ``[B, channels]`` axes, call the
    shared dsp implementation per signal, then reshape into the
    ``[B, channels, freq_bins, frames]`` layout the BandSplit consumes.

    Args:
        audio: [B, channels, samples]
        n_fft: FFT size
        hop_length: hop between frames
        window: [n_fft] Hann window

    Returns:
        (real, imag) each [B, channels, freq_bins, frames]
    """
    from .dsp import stft as _stft

    B, C, _ = audio.shape

    spectra = []
    for b in range(B):
        for c in range(C):
            spectra.append(
                _stft(
                    audio[b, c],
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=n_fft,
                    window=window,
                    center=True,
                    pad_mode="reflect",
                )
            )
    # Each entry: [num_frames, freq_bins] complex.
    stacked = mx.stack(spectra, axis=0)  # [B*C, num_frames, freq_bins]
    num_frames = stacked.shape[1]

    real = stacked.real.reshape(B, C, num_frames, -1).transpose(0, 1, 3, 2)
    imag = stacked.imag.reshape(B, C, num_frames, -1).transpose(0, 1, 3, 2)
    return real, imag


def istft(
    real: mx.array,
    imag: mx.array,
    n_fft: int,
    hop_length: int,
    window: mx.array,
    length: int,
) -> mx.array:
    """Inverse Short-time Fourier Transform via overlap-add.

    Thin batched/multichannel adapter over :func:`mlx_audio.dsp.istft`
    (which operates on a single ``[freq_bins, num_frames]`` complex
    spectrum). We loop over the ``[B, channels]`` axes and stack.

    Args:
        real: [B, channels, freq_bins, frames]
        imag: [B, channels, freq_bins, frames]
        n_fft: FFT size
        hop_length: hop between frames
        window: [n_fft] Hann window
        length: target output length in samples

    Returns:
        [B, channels, samples]
    """
    from .dsp import istft as _istft

    B, C, _, _ = real.shape

    # Reconstruct complex spectrum and split per-(batch, channel) pair.
    complex_spectra = real + 1j * imag  # [B, C, freq_bins, frames]

    # Note on dsp.istft args: pass ``length=None`` (not the target length)
    # because that branch correctly strips the center-pad; passing the target
    # length skips center-pad removal and shifts the reconstruction by
    # ``win_length // 2``. We trim to ``length`` ourselves below. Use
    # ``normalized=True`` for COLA-style ``window**2`` normalization, matching
    # PyTorch's ``torch.istft`` default.
    waveforms = []
    for b in range(B):
        for c in range(C):
            recon = _istft(
                complex_spectra[b, c],
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                center=True,
                length=None,
                normalized=True,
            )
            # dsp.istft's center-strip can leave the output a few samples
            # short of ``length`` when the chunk size isn't an integer
            # multiple of hop_length (e.g. ZFTurbo's 352800-sample chunks at
            # hop=512). Pad to length with zeros — those tail samples lived
            # in the center-pad region originally and aren't load-bearing.
            if recon.shape[0] < length:
                recon = mx.concatenate(
                    [recon, mx.zeros((length - recon.shape[0],), dtype=recon.dtype)]
                )
            waveforms.append(recon[:length])

    out = mx.stack(waveforms, axis=0).reshape(B, C, length)
    return out


# ---------- Main Model ----------


class MelRoFormer(nn.Module):
    """Mel-Band-RoFormer for vocal source separation.

    Pipeline: STFT → CaC → BandSplit → DualAxisTransformer → MaskEstimate → iSTFT
    """

    def __init__(self, config: MelRoFormerConfig):
        super().__init__()
        self.config = config

        self.band_split = BandSplit(config)

        # 6× dual-axis: [time_transformer, freq_transformer] per depth
        self.layers = [
            [
                Transformer(
                    config.dim, 1, config.heads, config.dim_head, config.ff_mult
                ),
                Transformer(
                    config.dim, 1, config.heads, config.dim_head, config.ff_mult
                ),
            ]
            for _ in range(config.depth)
        ]

        self.mask_estimators = [
            MaskEstimator(config, self.band_split.filterbank.band_dims)
        ]

    def __call__(self, audio: mx.array) -> mx.array:
        """Separate vocals from audio.

        Args:
            audio: [B, 2, samples] stereo audio at 44.1kHz

        Returns:
            [B, 2, samples] separated vocal audio
        """
        original_length = audio.shape[2]
        config = self.config

        # Hann window
        window = mx.array(np.hanning(config.n_fft + 1)[:-1].astype(np.float32))

        # Step 1: STFT → real/imag [B, 2, freq_bins, T]
        stft_real, stft_imag = stft(audio, config.n_fft, config.hop_length, window)
        B = stft_real.shape[0]
        freq_bins = stft_real.shape[2]
        T = stft_real.shape[3]

        # Step 2: CaC interleave — [B, 2, freq_bins, T] → [B, freq_bins*2, T]
        real_interleaved = stft_real.transpose(0, 2, 1, 3).reshape(B, freq_bins * 2, T)
        imag_interleaved = stft_imag.transpose(0, 2, 1, 3).reshape(B, freq_bins * 2, T)

        # Stack real/imag: [B, freq_bins*2, T, 2]
        stft_repr = mx.stack([real_interleaved, imag_interleaved], axis=-1)

        # Step 3: BandSplit → [B, T, num_bands, dim]
        x = self.band_split.split(stft_repr)
        Nb = x.shape[2]
        D = x.shape[3]

        # Step 4: 6× dual-axis transformer
        for time_tf, freq_tf in self.layers:
            # Time attention: [B, T, Nb, D] → [B*Nb, T, D]
            time_in = x.transpose(0, 2, 1, 3).reshape(B * Nb, T, D)
            time_out = time_tf(time_in)
            x = time_out.reshape(B, Nb, T, D).transpose(0, 2, 1, 3)

            # Freq attention: [B, T, Nb, D] → [B*T, Nb, D]
            freq_in = x.reshape(B * T, Nb, D)
            freq_out = freq_tf(freq_in)
            x = freq_out.reshape(B, T, Nb, D)

        # Step 5: Mask estimation → list of [B, T, band_dim]
        masks = self.mask_estimators[0](x)

        # Step 6: Merge masks → [B, freq_bins*2, T, 2]
        full_mask = self.band_split.merge(masks, freq_bins * 2)

        # Step 7: Complex multiply (input × mask)
        input_real = stft_repr[..., 0]  # [B, freq_bins*2, T]
        input_imag = stft_repr[..., 1]
        mask_real = full_mask[..., 0]
        mask_imag = full_mask[..., 1]

        out_real = input_real * mask_real - input_imag * mask_imag
        out_imag = input_real * mask_imag + input_imag * mask_real

        # Step 8: De-interleave → [B, 2, freq_bins, T]
        real_deint = out_real.reshape(B, freq_bins, 2, T).transpose(0, 2, 1, 3)
        imag_deint = out_imag.reshape(B, freq_bins, 2, T).transpose(0, 2, 1, 3)

        # Step 9: iSTFT → [B, 2, samples]
        separated = istft(
            real_deint,
            imag_deint,
            config.n_fft,
            config.hop_length,
            window,
            original_length,
        )

        return separated

    def sanitize(self, weights: dict) -> dict:
        """Sanitize PyTorch weights for MLX.

        Transformations applied (in order, per key):
          1. Split packed ``to_qkv`` projections into separate ``to_q``/``to_k``/``to_v``.
          2. Drop ``rotary_embed.freqs`` — MLX computes RoPE frequencies on-the-fly.
          3. Remap PyTorch ``nn.Sequential`` indices inside the mask estimator's
             per-band MLPs to MLX list indices (linears at PyTorch positions
             0, 2, 4 → MLX positions 0, 1, 2).
          4. Unwrap ``to_out.0.weight`` → ``to_out.weight`` (PyTorch wraps the
             output projection in a ``Sequential(Linear, Dropout)``; MLX uses
             a bare Linear).
          5. Rename any trailing ``.gamma`` → ``.weight`` — the PyTorch RMSNorm
             stores its scale as ``gamma``; MLX's ``nn.RMSNorm`` stores it as
             ``weight``.
        """
        import re

        mask_mlp_re = re.compile(
            r"^(mask_estimators\.\d+\.to_freqs\.\d+)\.0\.(\d+)\.(weight|bias)$"
        )

        new_weights = {}

        for key, value in weights.items():
            if "to_qkv.weight" in key:
                prefix = key.replace("to_qkv.weight", "")
                third = value.shape[0] // 3
                new_weights[f"{prefix}to_q.weight"] = value[:third]
                new_weights[f"{prefix}to_k.weight"] = value[third : 2 * third]
                new_weights[f"{prefix}to_v.weight"] = value[2 * third :]
                continue

            if key.endswith("rotary_embed.freqs"):
                continue

            m = mask_mlp_re.match(key)
            if m:
                prefix, seq_idx, kind = m.groups()
                mlx_idx = int(seq_idx) // 2
                key = f"{prefix}.{mlx_idx}.0.{kind}"

            if key.endswith("to_out.0.weight"):
                key = key[: -len(".0.weight")] + ".weight"

            if key.endswith(".gamma"):
                key = key[: -len(".gamma")] + ".weight"

            new_weights[key] = value

        return new_weights
