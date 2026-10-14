#!/usr/bin/env bash
cd /tmp
CUDA_VISIBLE_DEVICES= /home/USER/qwen36-api/.venv/bin/python -I /path/to/qwen36-workspace/work/next-update/fp8kv/scripts/vbound.py /home/USER/qwen36-api/optimization/reports/kv-scales.json > /path/to/qwen36-workspace/work/next-update/fp8kv/results-vbound.json; echo rc=$?
/home/USER/qwen36-api/.venv/bin/python -I -c "
import json;d=json.load(open('/path/to/qwen36-workspace/work/next-update/fp8kv/results-vbound.json'))
for l,v in d.items(): print(l, {k:(round(x,3) if isinstance(x,float) else x) for k,x in v.items()})
"
