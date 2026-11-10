# LGPL components: corresponding source and rebuilding

This archive accompanies the TransiroYT 0.5.0 DMG. It contains:

- `dav1d-1.5.3.tar.xz`: AV1 decoder source (BSD-2-Clause), statically linked
  into the shared FFmpeg codec library; original notices are retained.
- `ffmpeg.tar.xz`: unmodified FFmpeg 8.1.1 source, including LGPL-2.1.
- `build_consumer_tools.sh`: the exact FFmpeg configuration/build script.
- `soxr-1.0.0.tar.gz`: Python-SoXR source, including the vendored libsoxr
  source and build system. No separate libsoxr checkout is required.
- `autocommand-2.2.2.tar.gz`: the unmodified pure-Python LGPL-3.0 source,
  only for builds that include this optional dependency. Check the inventory.
- `third-party-sources.lock.json`: versions, download URLs and SHA-256 hashes.
- libsndfile 1.2.2, Ogg 1.3.5, Vorbis 1.3.7, FLAC 1.4.3, Opus 1.4,
  mpg123 1.32.3 and LAME 3.100 source archives. The SoundFile wheel's original
  libsndfile binary matches the git blob in libsndfile-binaries commit
  `a3e6f9769d0c7e91d2d036cf0fdbe5b4bbf18b87`.
- `gcc-13.4.0.tar.xz`: includes libquadmath and its build system. SciPy's
  `__config__.py` records Fortran compiler version 13.4.0. The native input
  hashes and component versions are recorded in `native-manifest.json`.
- `licenses/`: original notices, LGPL/GPL texts, package versions and the
  FFmpeg configuration log (`FFmpeg-config.log`).

No model weights are included. These archives are the actual upstream source
versions used in the application, not a promise to supply source later.

## Build

Use an Apple Silicon Mac, Xcode command line tools, pkg-config, uv and a C/C++ compiler.
The script builds dav1d first using pinned Meson 1.9.1 and Ninja 1.13.0, then
builds FFmpeg with libdav1d and the OS VideoToolbox framework enabled.
For FFmpeg, extract the archive and use the configure flags shown in
`build_consumer_tools.sh`, followed by `make -j4` and `make install`. The script
records the deployment target and intentionally disables GPL, nonfree,
version3 and static libraries. You may choose a different install prefix.
`licenses/FFmpeg-config.log` records the shipped build configuration.

For Python-SoXR, use Python 3.11, pip and a C/C++ toolchain. Extract
`soxr-1.0.0.tar.gz`, then run `python -m pip wheel --no-deps ./soxr-1.0.0`.
Its `pyproject.toml` specifies the build dependencies (including CMake and
nanobind). pip may download these build tools. The archive includes libsoxr's
preferred source and CMake files. Install the resulting wheel into the Python
environment you use to run TransiroYT.

For autocommand, extract its archive and run
`python -m pip wheel --no-deps ./autocommand-2.2.2`, or edit its Python source
directly. It contains no native binary requiring a relink.

For libsndfile and its statically included codecs, use the original
`licenses/supplemental/native/libsndfile-binaries/mac_build.sh` with
`darwin.cmake` from the same directory. Run `sh mac_build.sh arm64` on a Mac
with a C/C++ compiler, CMake, make and autotools. The exact source archives
used by that script are provided above; use those local archives instead
of downloading them again. The upstream script includes extraction and
configuration commands and the final libsndfile link with all codec archives.
This supplies source for the entire combined LGPL library so it can be
rebuilt with modified mpg123 or LAME as well as modified libsndfile.

For libquadmath, extract GCC 13.4.0, create a separate build directory and
configure GCC for your target with Fortran enabled. See GCC's included
`INSTALL` documentation for host prerequisites and configuration. Run
`make all-target-libquadmath` and `make install-target-libquadmath` after
configuring/building the target compiler prerequisites. On macOS use a
Darwin-capable GCC toolchain matching your target; the shipped SciPy build
records GCC 13.4.0. The app's copy is a separate dynamically linked dylib,
so it can be replaced without recompiling TransiroYT or SciPy.

## Replace and run

Work on your own copy. The app's `Contents/Resources/Runtime/bin/ffmpeg` and
`Runtime/lib/*.dylib` are dynamically linked. Replace them with compatible
builds and preserve the bundled `@rpath` / `@loader_path` layout. There is no
static linkage of FFmpeg into TransiroYT. Python-SoXR's extension and its
libsoxr implementation can be rebuilt together from the included source and
installed in `Runtime/python-tts/lib/python3.11/site-packages`. autocommand's
Python files are in the same site-packages directory. The app keeps Python
source alongside hash-checked bytecode; delete obsolete bytecode in your
working copy or regenerate it after editing.
The SoundFile native library is `_soundfile_data/libsndfile_arm64.dylib` in
that site-packages directory; replace it with your rebuilt combined library.
SciPy's separate LGPL library is `scipy/.dylibs/libquadmath.0.dylib`.
Preserve compatible exported symbols, architecture and loader paths.

Changing a signed app invalidates its existing signature. You can run the
backend from a development checkout using your rebuilt Python environment,
or rebuild and sign your modified application with your own signing identity.
The repository README describes the development build; `scripts/build_dmg.py`
describes the distribution build. If redistributing a modified macOS app, use
your own Developer ID and notarization. TransiroYT's private signing key is
not required to build, modify, or run the backend and is not distributed.

TransiroYT imposes no restriction on modification of these LGPL components,
replacement with compatible modified versions, or reverse engineering for
debugging those modifications. Preserve all applicable notices and provide
the matching source when redistributing these components.
