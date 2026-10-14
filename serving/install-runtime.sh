#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/USER/qwen36-api
mkdir -p "$ROOT" /path/to/LLM/qwen36-api/model /path/to/LLM/qwen36-api/downloads
export UV_NATIVE_TLS=true
DELIVERABLES=/path/to/qwen36-workspace/outputs/qwen36-api
test -x "$ROOT/.venv/bin/python" || uv venv --python 3.12.14 "$ROOT/.venv"
uv pip install --python "$ROOT/.venv/bin/python" -r "$DELIVERABLES/requirements.lock.txt"
uv pip freeze --python "$ROOT/.venv/bin/python" > "$ROOT/requirements.lock.txt"
"$ROOT/.venv/bin/python" --version
"$ROOT/.venv/bin/ft" --version
"$ROOT/.venv/bin/ft" serve --help > "$ROOT/serve-help.txt"
"$ROOT/.venv/bin/ft" bench bw --help > "$ROOT/bench-help.txt"
