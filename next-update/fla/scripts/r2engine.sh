#!/bin/bash
/home/USER/qwen36-api/.venv/bin/python -I - <<'PY'
import json, statistics as st
R='/home/USER/qwen36-api/optimization/reports/'
arms=('r2-base','r2-attn','main-candidate-r2')
tab={}
for r in (1,2,3):
  for a in arms:
    d=json.load(open(R+f'r2-ab-{r}-{a}-chunk-screen.json'))
    for x in d['rows']:
      if x.get('case')=='fresh':
        e=x.get('engine') or {}
        tab[(a,x['length'],r)]=(x['ttft_content_s'], e.get('prefill_gpu_ms'), e.get('engine_span_s'), e.get('prefill_forward_tokens'))
for L in (16384,65536,120000):
  for a in arms:
    print(L, a, [ (round(tab[(a,L,r)][0],3), tab[(a,L,r)][1] and round(tab[(a,L,r)][1],1)) for r in (1,2,3)])
  for a in arms[1:]:
    tr=[tab[(a,L,r)][1]/tab[('r2-attn' if a!='r2-attn' else 'r2-base',L,r)][1] for r in (1,2,3)]
    print('   prefill_gpu ratio', a, 'vs', 'r2-attn' if a!='r2-attn' else 'r2-base', [round(v,4) for v in tr], 'median', round(st.median(tr),4))
PY
