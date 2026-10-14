#!/usr/bin/env bash
cd /tmp
CUDA_VISIBLE_DEVICES= /home/USER/qwen36-api/.venv/bin/python -I /path/to/qwen36-workspace/work/next-update/fp8kv/scripts/kbound.py /home/USER/qwen36-api/optimization/reports/kv-scales.json > /path/to/qwen36-workspace/work/next-update/fp8kv/results-kbound.json
echo rc=$?
