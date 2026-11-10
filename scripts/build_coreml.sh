#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."
swift build --package-path apple/Inference -c release --scratch-path "$PWD/.local/CoreMLBuild"
bin_dir=$(swift build --package-path apple/Inference -c release --scratch-path "$PWD/.local/CoreMLBuild" --show-bin-path)
mkdir -p .local/bin
cp "$bin_dir/transiro-coreml" .local/bin/transiro-coreml
codesign --force --sign - .local/bin/transiro-coreml
