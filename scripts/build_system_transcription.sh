#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."
mkdir -p .local
xcrun swiftc -parse-as-library -O -target "$(uname -m)-apple-macos26.0" apple/SystemTranscribe.swift -o .local/system-transcribe
print 'macOS標準文字起こしブリッジをビルドしました（macOS 26以降）。'
