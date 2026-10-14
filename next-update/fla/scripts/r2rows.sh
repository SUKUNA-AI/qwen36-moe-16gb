#!/bin/bash
N=/home/USER/qwen36-api/optimization
tail -12 $N/reports/r2-driver19.log
ls $N/reports | grep -E "r2-ab-" | head
/home/USER/qwen36-api/.venv/bin/python -I - <<'PY'
import json
R='/home/USER/qwen36-api/optimization/reports/'
for r in (1,2,3):
  for a in ('r2-base','r2-attn','main-candidate-r2'):
    try: d=json.load(open(R+f'r2-ab-{r}-{a}-chunk-screen.json'))
    except Exception as e: print(r,a,'missing'); continue
    for x in d.get('rows',[]):
      if x.get('case')=='fresh':
        keys=[k for k in x.keys()]
        print(r,a,x['length'],round(x['ttft_content_s'],3),round(x.get('decode_client_tps',0),1), {k:x[k] for k in x if k in ('prefill_gpu_ms','server_prefill_ms','queue_s','cached_tokens','prompt_tokens')})
print(keys)
PY
