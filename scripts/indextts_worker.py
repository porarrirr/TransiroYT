"""IndexTTS-2.5 reference-voice worker (Python 3.10-3.11, run by the index-tts environment).

Started by ``backend/transiroyt/reference_conversion.py`` with the model directory
as the working directory: ``infer_v2_5.py`` points ``HF_HUB_CACHE`` at
``./checkpoints/hf_cache`` relative to it, and the pinned weights live in
``checkpoints/``.

stdout is a JSON-lines protocol:
    {"ready": true, "device": "mps"} after the model loads,
    one reply per request {"text", "reference", "output", "lang"}:
    {"ok": true, "seconds": ...} or {"ok": false, "error": ...}.
The model prints to stdout while loading and generating, so stdout is redirected
to stderr before importing it; the protocol keeps a private handle to the
original stdout. SIGTERM ends the process (the parent sends it from close()).
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
import traceback

PROTOCOL = sys.__stdout__
sys.stdout = sys.stderr


def _reply(payload):
    PROTOCOL.write(json.dumps(payload, ensure_ascii=False) + "\n")
    PROTOCOL.flush()


def _terminate(signum, frame):
    raise SystemExit(0)


def main():
    # A SIGTERM raised between model steps stops the worker instead of leaving
    # an orphan holding about 3 GB of MPS memory.
    signal.signal(signal.SIGTERM, _terminate)
    try:
        import torch
        from indextts.infer_v2_5 import IndexTTS2

        model = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints",
                          use_bf16=False, use_qwen_emo=False,
                          device=os.environ.get("INDEXTTS_DEVICE") or None)
    except Exception as exc:  # Report the failure; the parent reads the log for the traceback.
        traceback.print_exc(file=sys.stderr)
        _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    print(f">> IndexTTS-2.5 ready on {model.device}", file=sys.stderr, flush=True)
    _reply({"ready": True, "device": model.device})
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        started = time.monotonic()
        try:
            model.infer(spk_audio_prompt=request["reference"], text=request["text"],
                        lang=request.get("lang", "ja"), output_path=request["output"],
                        verbose=False)
            _reply({"ok": True, "seconds": time.monotonic() - started})
        except Exception as exc:  # A failed take must not stop the resident worker.
            traceback.print_exc(file=sys.stderr)
            _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            # The model frees only the CUDA cache; on MPS the driver memory grew
            # ~0.3 GiB per conversion until the 41st (C782A726) ran out of memory.
            if model.device == "mps":
                torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
