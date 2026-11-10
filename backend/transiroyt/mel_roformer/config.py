# Copyright (c) 2026 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""Fixed Kim Mel-Band RoFormer architecture, adapted from mlx-audio 0.6.5."""

from dataclasses import dataclass


@dataclass
class MelRoFormerConfig:
    """Architecture matching the pinned KimberleyJSN weights."""

    # Model architecture
    dim: int = 384
    depth: int = 6  # Number of dual-axis transformer depth levels
    heads: int = 8
    dim_head: int = 64
    num_bands: int = 60
    ff_mult: int = 4
    mlp_expansion_factor: int = 4
    mask_estimator_depth: int = 2

    # STFT parameters
    n_fft: int = 2048
    hop_length: int = 441
    sample_rate: int = 44100

    @property
    def mlp_hidden(self) -> int:
        """MLP hidden dimension in mask estimator."""
        return self.dim * self.mlp_expansion_factor  # 1536

    @property
    def freq_bins(self) -> int:
        """Number of frequency bins from STFT (n_fft/2 + 1)."""
        return self.n_fft // 2 + 1  # 1025
