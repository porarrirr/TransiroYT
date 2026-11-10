#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."
work="${TRANSIROYT_MOBILE_BUILD_DIR:-$PWD/.local/work/mobile}"
xcodegen generate --spec apple/Mobile/project.yml
xcodebuild -project apple/Mobile/TransiroYTMobile.xcodeproj -scheme TransiroYTMobile \
  -sdk iphonesimulator -configuration Debug -derivedDataPath "$work/DerivedData" CODE_SIGNING_ALLOWED=NO ARCHS=arm64 build
