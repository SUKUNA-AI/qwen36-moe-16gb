#!/usr/bin/env bash
F=/path/to/qwen36-workspace/work/next-update/fp8kv
cd /tmp && CUDA_VISIBLE_DEVICES= /home/USER/qwen36-api/.venv/bin/python -I -B $F/scripts/predict.py $F/results/predictions.json 2>&1 | tail -95
