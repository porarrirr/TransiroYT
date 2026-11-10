"""ECAPA-TDNN speaker similarity for subtitle-aligned clips. Run with the TTS Python.

Input (stdin JSON): {"audio": 16 kHz mono WAV, "cache": HF cache dir, "download": bool,
"items": [[start, end], ...], "output": float32 matrix path}. Writes the cosine
similarity matrix (row-major float32) and prints per-item level statistics.
"""
from __future__ import annotations

import json
import sys
import time
import traceback

MODEL_REPO = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
SAMPLE_RATE = 16000


def load_model(cache, download):
    import torch
    from huggingface_hub import snapshot_download
    from speechbrain.lobes.features import Fbank
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    from speechbrain.processing.features import InputNormalization

    # The recipe's own YAML resolves an unpinned repository name; build the
    # same modules here and load only the pinned embedding weights.
    directory = snapshot_download(MODEL_REPO, revision=MODEL_REVISION, cache_dir=cache,
                                  allow_patterns=["embedding_model.ckpt", "hyperparams.yaml", "README.md"],
                                  local_files_only=not download)
    model = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072],
                       kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
                       attention_channels=128, lin_neurons=192)
    state = torch.load(f"{directory}/embedding_model.ckpt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return Fbank(n_mels=80), InputNormalization(norm_type="sentence", std_norm=False), model


def main():
    started = time.perf_counter()
    request = json.load(sys.stdin)
    import numpy as np
    import soundfile
    import torch

    features, normalization, model = load_model(request["cache"], request.get("download", False))
    audio, rate = soundfile.read(request["audio"], dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError("speaker audio must be 16 kHz mono")
    vectors, stats = [], []
    with torch.inference_mode():
        for start, end in request["items"]:
            clip = audio[max(0, round(start * rate)):max(0, round(end * rate))]
            if len(clip) < rate * 0.2:
                raise ValueError(f"clip too short for a speaker embedding: {start}-{end}")
            signal = torch.from_numpy(np.ascontiguousarray(clip))[None]
            lengths = torch.ones(1)
            embedding = model(normalization(features(signal), lengths), lengths)[0, 0]
            vectors.append(torch.nn.functional.normalize(embedding, dim=0).numpy())
            peak = float(np.max(np.abs(clip)))
            # Speech level vs the floor between syllables: a music bed or room noise
            # keeps the floor up, and the TTS reproduces it from the reference.
            frame = rate // 50
            frames = len(clip) // frame
            energy = np.mean(clip[:frames * frame].reshape(frames, frame) ** 2, axis=1)
            levels = 10 * np.log10(energy + 1e-10)
            stats.append({"rms": float(np.sqrt(np.mean(clip * clip))), "peak": peak,
                          "clipped": float(np.mean(np.abs(clip) >= 0.999)),
                          "speech_db": float(np.percentile(levels, 90)),
                          "floor_db": float(np.percentile(levels, 10))})
    matrix = np.stack(vectors) @ np.stack(vectors).T
    matrix.astype("<f4").tofile(request["output"])
    print(json.dumps({"ok": True, "count": len(vectors), "stats": stats,
                      "elapsed_seconds": round(time.perf_counter() - started, 3)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)
