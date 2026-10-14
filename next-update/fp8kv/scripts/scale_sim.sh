#!/usr/bin/env bash
cd /tmp
CUDA_VISIBLE_DEVICES= /home/USER/qwen36-api/.venv/bin/python -I /path/to/qwen36-workspace/work/next-update/fp8kv/scripts/scale_sim.py /path/to/qwen36-workspace/work/next-update/fp8kv/results-kbound.json > /path/to/qwen36-workspace/work/next-update/fp8kv/results-scale-sim.json; echo rc=$?
/home/USER/qwen36-api/.venv/bin/python -I -c "
import json;d=json.load(open('/path/to/qwen36-workspace/work/next-update/fp8kv/results-scale-sim.json'))
for l,o in d.items():
  print(l, {k:(round(v['rel_l2'],4),v['clipped'],round(v['logit_err_std'],4),round(v['max_abs_k'],1)) for k,v in o.items()})
"
