#!/usr/bin/env bash
cd /tmp
F=/path/to/qwen36-workspace/work/next-update/fp8kv
CUDA_VISIBLE_DEVICES= /home/USER/qwen36-api/.venv/bin/python -I $F/scripts/vsim.py $F/results-vbound.json > $F/results-vsim.json; echo rc=$?
/home/USER/qwen36-api/.venv/bin/python -I -c "
import json;d=json.load(open('$F/results-vsim.json'))
for l,o in d.items(): print(l, {k:(v['rel_l2'],v['clipped'],v['subnormal_frac']) for k,v in o.items()})
"
