Adapted from mlx-audio 0.6.5 (MIT), https://pypi.org/project/mlx-audio/0.6.5/ .
Only the Mel RoFormer model, config and mel_filters are included. The general
model downloader, other checkpoint presets and unrelated DSP code are omitted.
Optimizations in model.py and optimized.py reproduce the user-approved 2026-11-10 benchmark.
Model weights are fetched separately from KimberleyJSN/melbandroformer at
ac9b0614ab3cd7f77219e18ba494dfd93956c348 (MIT model card, April 2026).
