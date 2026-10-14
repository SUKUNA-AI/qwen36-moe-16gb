#!/bin/bash
D=/path/to/qwen36-workspace/work/next-update/fla
N=/home/USER/qwen36-api/optimization
cd /tmp
nice -n 19 /home/USER/qwen36-api/.venv/bin/python -I $D/scripts/fla_nsys.py $N/reports/model-final-fla-on-20261014T192523Z.sqlite --label fla-on --out $D/results/nsys-fla-on.json
/home/USER/qwen36-api/.venv/bin/python -I -c "
import json; d=json.load(open('$N/reports/model-final-fla-on-20261014T192523Z.run-receipt.json'))
for r in d['requests']: print({k:r.get(k) for k in ('label','status','prompt_tokens','completion_tokens','cached_tokens','ttft_content_s')})"
