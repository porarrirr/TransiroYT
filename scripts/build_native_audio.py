"""Build the dependency-free Rust PCM kernel for development and app bundling."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    target = root / "backend/transiroyt" / ("_audio_native" + suffix)
    source = root / "scripts/native_audio/lib.rs"
    temporary = target.with_suffix(suffix + ".tmp")
    try:
        subprocess.run(["rustc", "--edition=2021", "--crate-type=cdylib", "--crate-name=transiroyt_audio",
                        "-C", "opt-level=3", "-C", "panic=abort", str(source), "-o", str(temporary)], check=True)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


if __name__ == "__main__":
    print(build())
