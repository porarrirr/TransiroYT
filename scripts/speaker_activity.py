"""Who speaks when: Nemotron-3-Diarization frame activity for a whole programme. Run with the TTS Python.

Input (stdin JSON): {"audio": 16 kHz mono WAV, "cache": HF cache dir, "download": bool,
"output": float32 matrix path}. Writes the per-speaker activity probabilities, one row
per 10 ms frame and one column per speaker in order of first arrival (row-major
float32), and prints {"ok": true, "frames": n, "speakers": k}.

The model's offline configuration (30.4 s chunks with its speaker cache) is the
checkpoint default; it tracks up to eight speakers over the whole recording.
"""
from __future__ import annotations

import json
import sys
import time
import traceback

MODEL_REPO = "nvidia/Nemotron-3-Diarization"
MODEL_REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
MODEL_FILES = ("config.json", "processor_config.json", "model.safetensors", "README.md")
SAMPLE_RATE = 16000


def main():
    started = time.perf_counter()
    request = json.load(sys.stdin)
    import numpy as np
    import soundfile
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, Nemotron3DiarizationForAudioFrameClassification

    directory = snapshot_download(MODEL_REPO, revision=MODEL_REVISION, cache_dir=request["cache"],
                                  allow_patterns=list(MODEL_FILES), local_files_only=not request.get("download"))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(directory)
    model = Nemotron3DiarizationForAudioFrameClassification.from_pretrained(directory).to(device).eval()
    audio, rate = soundfile.read(request["audio"], dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError("speaker audio must be 16 kHz mono")
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
    with torch.inference_mode():
        activity = model(**inputs).logits.float().sigmoid()[0].cpu().numpy()
    np.ascontiguousarray(activity, dtype="<f4").tofile(request["output"])
    print(json.dumps({"ok": True, "frames": int(activity.shape[0]), "speakers": int(activity.shape[1]),
                      "device": device, "elapsed_seconds": round(time.perf_counter() - started, 3)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # The parent reads stderr for the traceback.
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)
