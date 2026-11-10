#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."
export PYTHONPATH="$PWD/backend${PYTHONPATH:+:$PYTHONPATH}"
exec uv run uvicorn transiroyt.app:create_app --factory --host 127.0.0.1 --port 8765
