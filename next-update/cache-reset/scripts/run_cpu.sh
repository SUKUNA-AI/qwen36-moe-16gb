#!/usr/bin/env bash
# Run a CPU-only python script from this dir with the venv python (read-only; -B: no bytecode anywhere).
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
cd /tmp
exec /home/USER/qwen36-api/.venv/bin/python -B "$@"
