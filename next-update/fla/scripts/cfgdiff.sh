#!/bin/bash
cd /home/USER/qwen36-api/optimization/config
/home/USER/qwen36-api/.venv/bin/python -I - <<'PY'
import json
base=json.load(open('main-release.json'))
for f in ('main-candidate-r2','r2-attn','r2-base','profile-final-main-release'):
    d=json.load(open(f+'.json'))
    diff={k:(base['env'].get(k),d['env'].get(k)) for k in set(base['env'])|set(d['env']) if base['env'].get(k)!=d['env'].get(k)}
    print(f, d['chunk'], d['memory_ratio'], d['context'], diff)
PY
