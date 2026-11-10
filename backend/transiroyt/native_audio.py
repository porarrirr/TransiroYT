"""Small C ABI boundary shared by source and bundled Python runtimes."""
import ctypes
import sys
from functools import cache
from pathlib import Path


@cache
def _kernel():
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    try:
        library = ctypes.CDLL(str(Path(__file__).with_name("_audio_native" + suffix)))
    except OSError as error:
        raise RuntimeError("Rust音声ライブラリを読み込めません。scripts/build_native_audio.pyでビルドしてください。") from error
    kernel = library.pcm16_energy
    kernel.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
    kernel.restype = None
    return kernel


def frame_energies(pcm: bytes, frame: int, *, start=0, stop=None, partial=True):
    """Exact squared sums of PCM16LE frames, without copying the input samples."""
    stop = len(pcm) // 2 if stop is None else min(stop, len(pcm) // 2)
    if frame < 1 or start < 0 or len(pcm) % 2:
        raise ValueError("Invalid PCM16 frame bounds")
    count = max(0, stop - start)
    if not partial:
        count -= count % frame
    output = (ctypes.c_uint64 * ((count + frame - 1) // frame))()
    if count:
        pointer = ctypes.cast(ctypes.c_char_p(pcm), ctypes.c_void_p).value + start * 2
        _kernel()(pointer, count, frame, output)
    return list(output)
