"""Kim Mel-Band RoFormer MLX separation using the approved speed-priority settings.

Input JSON: audio (44.1 kHz stereo WAV), model (pinned Kim checkpoint),
vocals and background (output WAV paths). Runs with the audio-worker Python.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def load_model(checkpoint):
    import mlx.core as mx
    from transiroyt.mel_roformer.config import MelRoFormerConfig
    from transiroyt.mel_roformer.model import MelRoFormer
    from transiroyt.mel_roformer.optimized import prepare_model

    converted = checkpoint.with_suffix(".mlx-fp32-v1.safetensors")
    if not converted.exists():
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # Keep FP32 until sanitizing and loading, as in the approved conversion.
        weights = {key: mx.array(value.detach().cpu().float().numpy()) for key, value in state.items()}
        temporary = converted.with_name(f".{converted.name}.tmp.safetensors")
        try:
            mx.save_safetensors(str(temporary), weights)
            temporary.replace(converted)
        finally:
            temporary.unlink(missing_ok=True)
        del state, weights
    model = MelRoFormer(MelRoFormerConfig())
    model.load_weights(list(model.sanitize(mx.load(str(converted))).items()), strict=True)
    model.eval()
    prepare_model(model)
    mx.eval(model.parameters())
    return mx.compile(model)


def main():
    started = time.perf_counter()
    request = json.load(sys.stdin)
    import mlx.core as mx
    import soundfile
    from transiroyt.mel_roformer.separation import SAMPLE_RATE, separate_stream

    model = load_model(Path(request["model"]))
    with soundfile.SoundFile(request["audio"]) as source, \
            soundfile.SoundFile(request["vocals"], "w", SAMPLE_RATE, 2, "FLOAT") as vocals, \
            soundfile.SoundFile(request["background"], "w", SAMPLE_RATE, 2, "FLOAT") as background:
        separate_stream(source, vocals, background, lambda audio: model(mx.array(audio)))
        seconds = source.frames / SAMPLE_RATE
    print(json.dumps({"ok": True, "seconds": seconds,
                      "elapsed_seconds": round(time.perf_counter() - started, 3)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)
