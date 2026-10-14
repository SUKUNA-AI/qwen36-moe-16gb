#!/usr/bin/env bash
/home/USER/qwen36-api/.venv/bin/python -I - <<'PY'
import json, numpy as np, statistics as st
rows=[json.loads(l) for l in open('/home/USER/qwen36-api/optimization/reports/engine.jsonl') if l.strip()]
data={}
for r in rows:
    if r['expert_slots']!=3164 or r['decode_forwards']<200: continue
    L=r['prefill_forward_tokens']; b=min((93,4125,16413,65564,120029),key=lambda x:abs(x-L))
    if abs(b-L)>50: continue
    d=r['expert_counter_deltas']; n=r['decode_forwards']
    data.setdefault(b,[]).append((d['decode_missing']/(d['decode_active']/320), r['decode_gpu_ms']/n))
X=[];Y=[];G=[]
for b,v in sorted(data.items()):
    m=np.array(v); print(b, 'n',len(v),'median ms/fwd %.2f'%np.median(m[:,1]),'median miss/tok %.1f'%np.median(m[:,0]))
    for x,y in v: X.append(x);Y.append(y);G.append(b)
# pooled within-length regression (demeaned per length)
X=np.array(X);Y=np.array(Y);G=np.array(G)
xd=np.concatenate([X[G==g]-X[G==g].mean() for g in sorted(set(G))]);yd=np.concatenate([Y[G==g]-Y[G==g].mean() for g in sorted(set(G))])
slope=(xd*yd).sum()/(xd*xd).sum(); print('within-length slope ms per (miss/token): %.4f'%slope, 'r=%.2f'%(np.corrcoef(xd,yd)[0,1]))
PY
