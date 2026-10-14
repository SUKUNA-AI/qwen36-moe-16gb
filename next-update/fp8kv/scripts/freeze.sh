#!/usr/bin/env bash
set -euo pipefail
F=/path/to/qwen36-workspace/work/next-update/fp8kv
cd /tmp
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/USER/qwen36-api/.venv/bin/python -I -B $F/eval-fp8kv.py freeze --out $F/eval-fp8kv-freeze.json --tokenizer /path/to/LLM/qwen36-api/model
/home/USER/qwen36-api/.venv/bin/python -I -B $F/eval-fp8kv.py verify
sha256sum $F/eval-fp8kv-freeze.json $F/eval-fp8kv.py $F/eval-fp8kv-compare.py $F/configs/*.json
