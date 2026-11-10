#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."
/usr/bin/python3 scripts/build_native_audio.py
swift build --package-path apple/Desktop -c release --scratch-path "$PWD/.local/DesktopBuild"
bin_dir=$(swift build --package-path apple/Desktop -c release --scratch-path "$PWD/.local/DesktopBuild" --show-bin-path)
app="$PWD/.local/TransiroYT.app"
# Assemble from scratch so parts removed from the app (such as the former browser extensions) do not linger.
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
./scripts/build_coreml.sh
./scripts/build_system_transcription.sh
cp "$bin_dir/TransiroYT" "$app/Contents/MacOS/TransiroYT"
swift scripts/create_app_icon.swift "$app/Contents/Resources"
/usr/bin/python3 scripts/build_sparkle.py "$app" "$PWD/.local/DesktopBuild/artifacts"
/usr/bin/python3 - "$app/Contents/Info.plist" "$PWD" <<'PY'
import plistlib
import sys

with open(sys.argv[1], "wb") as f:
    plistlib.dump({
        "CFBundleExecutable": "TransiroYT",
        "CFBundleIdentifier": "com.porarri.transiroyt.desktop.development",
        "CFBundleName": "TransiroYT",
        "CFBundleDisplayName": "TransiroYT",
        "CFBundlePackageType": "APPL",
        "CFBundleIconFile": "AppIcon",
        "CFBundleShortVersionString": "0.5.2",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "26.0",
        "NSHighResolutionCapable": True,
        "NSLocalNetworkUsageDescription": "同じWi-FiのiPhone・iPadと接続し、音声生成と音声分析を分担します。",
        "TransiroYTProjectPath": sys.argv[2],
    }, f)
PY
codesign --force --sign - "$app"
print "ビルド完了: $app"
if [[ "${1:-}" == "--open" ]]; then
  open "$app"
fi
