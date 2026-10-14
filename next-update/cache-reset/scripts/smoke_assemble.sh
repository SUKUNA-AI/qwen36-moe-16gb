#!/usr/bin/env bash
# CPU smoke test of validate_v7.py's assembly path on the EXISTING v6 teacher-main-release run (01d9d5cd, no restart).
# Expected: independent recomputation 125 x 0.0 (that run was bit-exact); the unchanged compare-quality.py instead
# picks the later D-run rows (87dd67bd: ru-0, long-4096, long-120000 with header dedup) -> shows the deviation.
set -euo pipefail
here=/path/to/qwen36-workspace/work/next-update/cache-reset
py=/home/USER/qwen36-api/.venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
mkdir -p "$here/cpu-smoke"
"$py" -B -c '
import json,sys
d=json.load(open("/home/USER/qwen36-api/optimization/reports/teacher-main-release.json"))
d["clean_cache"]={"harness":"v6-synthetic","mode":"none","cases":[],"runs":[]}
json.dump(d,open(sys.argv[1],"w"))' "$here/cpu-smoke/synthetic-v6-main-release.json"
cd /tmp
"$py" -B "$here/scripts/validate_v7.py" --profile teacher-main-release --out-dir "$here/cpu-smoke" \
  --assemble-only "$here/cpu-smoke/synthetic-v6-main-release.json" || echo "rc=$?"
