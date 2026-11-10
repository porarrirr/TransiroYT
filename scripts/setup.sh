#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."
/usr/bin/python3 scripts/build_native_audio.py
if ! command -v uv >/dev/null; then
  print 'uvを先に導入してください: https://docs.astral.sh/uv/getting-started/installation/'
  exit 1
fi
if ! command -v ffmpeg >/dev/null; then
  print 'ffmpegを先に導入してください: brew install ffmpeg'
  exit 1
fi
uv sync --python 3.12
./scripts/build_system_transcription.sh
if [[ "${1:-}" == "--with-tts" ]]; then
  # The TTS environment (PyTorch, MLX) runs the ECAPA, separation, Whisper and d1 workers.
  revision=89f9d8fbd4d51ea019867ee1197725ede1df13c5
  repo=third_party/Irodori-TTS
  if [[ ! -d "$repo/.git" ]]; then
    mkdir -p third_party
    git clone https://github.com/Aratako/Irodori-TTS.git "$repo"
    git -C "$repo" checkout --detach "$revision"
  elif [[ "$(git -C "$repo" rev-parse HEAD)" != "$revision" ]]; then
    print '既存のIrodori-TTSは別のリビジョンです。変更を保護するため自動変更しません。'
    exit 1
  fi
  # Keep the upstream source/license, but do not bundle its training and web UI environment.
  mkdir -p .local
  if [[ ! -x .local/file-workers/bin/python ]]; then
    uv venv --no-project --python 3.11 .local/file-workers
  fi
  uv pip sync --python .local/file-workers/bin/python --require-hashes distribution/worker-requirements.lock
  # IndexTTS-2.5 rebuilds the file-dubbing reference audio in a separate Python 3.11
  # environment (its dependencies conflict with the TTS environment).
  revision=d9e41aac89fd00b3d71497fddb287b7f24613712
  repo=third_party/index-tts
  if [[ ! -d "$repo/.git" ]]; then
    mkdir -p third_party
    git clone https://github.com/index-tts/index-tts.git "$repo"
    git -C "$repo" checkout --detach "$revision"
  elif [[ "$(git -C "$repo" rev-parse HEAD)" != "$revision" ]]; then
    print '既存のindex-ttsは別のリビジョンです。変更を保護するため自動変更しません。'
    exit 1
  fi
  uv sync --project "$repo" --no-dev --python 3.11
  # The upstream lock pins SciPy 1.16.2 for Python 3.11, but the distribution
  # native manifest reviews the 1.17.0 libquadmath input used by python-tts.
  uv pip install --python "$repo/.venv/bin/python" "scipy==1.17.0"
  # non-editable so the bundled runtime can import indextts after bundle_python
  # removes the .pth files that point at this checkout's absolute path.
  uv pip install --python "$repo/.venv/bin/python" --no-deps "$repo"
fi
print ''
print 'セットアップ完了。scripts/build_desktop.sh で開発用アプリを作成できます。'
