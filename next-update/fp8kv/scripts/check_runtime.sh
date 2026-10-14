#!/usr/bin/env bash
F=/path/to/qwen36-workspace/work/next-update/fp8kv
/home/USER/qwen36-api/.venv/bin/python -I - <<'PY'
import json,hashlib,pathlib
F=pathlib.Path('/path/to/qwen36-workspace/work/next-update/fp8kv')
fz=json.load(open(F/'eval-fp8kv-freeze.json'))
N=pathlib.Path('/home/USER/qwen36-api/optimization/python/freetoken')
for rel,sha in fz['runtime_expectations']['source_files'].items():
    print(rel, sha[:12], hashlib.sha256((N/rel).read_bytes()).hexdigest()==sha)
print('rule', fz['decision_rule']['id'], fz['decision_rule_sha256'][:16], 'profiles', list(fz['profiles']))
PY
