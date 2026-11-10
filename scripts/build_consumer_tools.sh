#!/bin/zsh
# Build portable LGPL/BSD media tools. No Homebrew runtime.
set -euo pipefail
cd "${0:A:h}/.."
build="$PWD/.local/distribution-build"
sources="$build/sources"
prefix="$build/tools"
mkdir -p "$sources" "$prefix/bin" "$prefix/lib" "$prefix/licenses"
fetch() {
  local url="$1" file="$2" checksum="$3"
  [[ -f "$file" ]] || curl -fL --retry 3 "$url" -o "$file"
  print "$checksum  $file" | shasum -a 256 -c -
}
fetch https://ffmpeg.org/releases/ffmpeg-8.1.1.tar.xz "$sources/ffmpeg.tar.xz" b6863adde98898f42602017462871b5f6333e65aec803fdd7a6308639c52edf3
fetch https://download.videolan.org/pub/videolan/dav1d/1.5.3/dav1d-1.5.3.tar.xz "$sources/dav1d-1.5.3.tar.xz" 732010aa5ef461fa93355ed2c6c5fedb48ddc4b74e697eaabe8907eaeb943011
node_version=24.22.1
node_archive="node-v${node_version}-darwin-arm64.tar.gz"
fetch "https://nodejs.org/dist/v${node_version}/${node_archive}" "$sources/$node_archive" 89c682593d76cd44233fa53b1563264e7de9cbb189cad169135c5b4fadfbc0a7
# Refresh cached tools when the pinned security release changes.
if [[ ! -f "$prefix/bin/node" ]] || [[ "$("$prefix/bin/node" --version)" != "v${node_version}" ]]; then
  tar -xf "$sources/$node_archive" -C "$sources"
  cp "$sources/node-v${node_version}-darwin-arm64/bin/node" "$prefix/bin/"
  cp "$sources/node-v${node_version}-darwin-arm64/LICENSE" "$prefix/licenses/Node-and-third-party-licenses.txt"
fi
[[ -d "$sources/ffmpeg-8.1.1" ]] || tar -xf "$sources/ffmpeg.tar.xz" -C "$sources"
export MACOSX_DEPLOYMENT_TARGET=14.0
[[ -d "$sources/dav1d-1.5.3" ]] || tar -xf "$sources/dav1d-1.5.3.tar.xz" -C "$sources"
if [[ ! -f "$prefix/lib/libdav1d.a" ]]; then
  [[ -d "$build/media-build-tools" ]] || uv venv "$build/media-build-tools"
  uv pip install --python "$build/media-build-tools/bin/python" meson==1.9.1 ninja==1.13.0
  export PATH="$build/media-build-tools/bin:$PATH"
  meson setup --reconfigure "$build/dav1d-build" "$sources/dav1d-1.5.3" \
    --prefix="$prefix" --default-library=static --buildtype=release -Denable_tools=false -Denable_tests=false
  meson compile -C "$build/dav1d-build"
  meson install -C "$build/dav1d-build"
fi
cp "$sources/dav1d-1.5.3/COPYING" "$prefix/licenses/dav1d-BSD-2-Clause.txt"
if [[ ! -f "$prefix/bin/ffmpeg" ]] || ! "$prefix/bin/ffmpeg" -hide_banner -protocols 2>/dev/null | /usr/bin/grep -q '  https$' \
   || ! "$prefix/bin/ffmpeg" -hide_banner -encoders 2>/dev/null | /usr/bin/grep ' h264_videotoolbox ' >/dev/null \
   || ! "$prefix/bin/ffmpeg" -hide_banner -decoders 2>/dev/null | /usr/bin/grep ' libdav1d ' >/dev/null; then
  (
    cd "$sources/ffmpeg-8.1.1"
    PKG_CONFIG_PATH="$prefix/lib/pkgconfig" ./configure --prefix="$prefix" --cc=clang --arch=arm64 --target-os=darwin \
      --extra-cflags=-mmacosx-version-min=14.0 --extra-ldflags=-mmacosx-version-min=14.0 \
      --disable-autodetect --disable-gpl --disable-nonfree --disable-version3 \
      --disable-doc --disable-debug --enable-network --disable-iconv \
      --disable-audiotoolbox --enable-videotoolbox --enable-libdav1d --enable-securetransport \
      --disable-ffplay --enable-shared --disable-static
    make -j4
    make install
  )
fi
cp "$sources/ffmpeg-8.1.1/COPYING.LGPLv2.1" "$prefix/licenses/FFmpeg-LGPL-2.1.txt"
# Pin the same Gemma 4-compatible revision used during development. Static linking
# and embedded Metal kernels leave no compiler or Homebrew requirement at runtime.
llama_revision=8ca4060b7e30e76baaa5670ee1ec1b760b7f8552
llama_checksum=68f2759b1cd93f352ade2413c1e123199632a348c46414b05f30efc67913f618
fetch "https://github.com/ggml-org/llama.cpp/archive/$llama_revision.tar.gz" "$sources/llama.cpp-$llama_revision.tar.gz" "$llama_checksum"
[[ -d "$sources/llama.cpp-$llama_revision" ]] || tar -xf "$sources/llama.cpp-$llama_revision.tar.gz" -C "$sources"
llama_source="$sources/llama.cpp-$llama_revision"
if [[ ! -x "$prefix/bin/llama-server" || ! -f "$prefix/licenses/llama.cpp/build-id" ]] \
    || [[ "$(cat "$prefix/licenses/llama.cpp/build-id" 2>/dev/null)" != "$llama_revision-static-metal-no-ui-v2" ]]; then
  cmake -S "$llama_source" -B "$build/llama-build-$llama_revision" \
    -DLLAMA_BUILD_NUMBER=0 -DLLAMA_BUILD_COMMIT=8ca4060b7 \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
    -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
  cmake --build "$build/llama-build-$llama_revision" --config Release --target llama-server -j4
  cp "$build/llama-build-$llama_revision/bin/llama-server" "$prefix/bin/llama-server"
  mkdir -p "$prefix/licenses/llama.cpp"
  cp "$llama_source/LICENSE" "$prefix/licenses/llama.cpp/LICENSE"
  cp -R "$llama_source/licenses" "$prefix/licenses/llama.cpp/"
  cp -R "$llama_source/vendor" "$prefix/licenses/llama.cpp/"
  cp distribution/llama.lock.json "$prefix/licenses/llama.cpp/source.json"
  print "$llama_checksum" > "$prefix/licenses/llama.cpp/source.sha256"
  print "$llama_revision-static-metal-no-ui-v2" > "$prefix/licenses/llama.cpp/build-id"
fi
print "配布用ツール: $prefix"
