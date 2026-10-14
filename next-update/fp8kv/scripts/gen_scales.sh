#!/usr/bin/env bash
F=/path/to/qwen36-workspace/work/next-update/fp8kv
cd /tmp && /home/USER/qwen36-api/.venv/bin/python -I $F/scripts/gen_scales.py $F /home/USER/qwen36-api/optimization/reports/kv-scales.json
sha256sum $F/configs/kv-scales-bound.json
