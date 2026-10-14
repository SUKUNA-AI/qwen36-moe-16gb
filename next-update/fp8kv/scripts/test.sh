#!/usr/bin/env bash
cd /tmp && CUDA_VISIBLE_DEVICES= /home/USER/qwen36-api/.venv/bin/python -I -B /path/to/qwen36-workspace/work/next-update/fp8kv/tests/test_compare_cpu.py 2>&1 | grep -v '^{"decision"' 
