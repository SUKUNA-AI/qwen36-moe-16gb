#!/bin/bash
# CPU tests: the layer-major toy suite + the short-route tests against the patched tree copy.
H=/path/to/qwen36-workspace/work/next-update/short-route
LM=/path/to/qwen36-workspace/work/layer-major
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 QWEN36_ENGINE_METRICS=0
export PYTHONPATH="${FT_ROOT:-$H/tree/python}"
cd $LM
/home/USER/qwen36-api/.venv/bin/python -P $LM/minitest.py "$@"
