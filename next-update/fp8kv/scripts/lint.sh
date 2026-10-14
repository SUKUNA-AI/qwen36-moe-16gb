#!/usr/bin/env bash
F=/path/to/qwen36-workspace/work/next-update/fp8kv
for s in $F/gpu/*.sh; do bash -n "$s" && echo "syntax ok $(basename $s)"; done
cd /tmp && /home/USER/qwen36-api/.venv/bin/python -I -B -c "
import ast,sys,pathlib
for p in ['$F/gpu/show-quality.py','$F/gpu/run-ids.py','$F/summarize-ab.py','$F/eval-fp8kv.py','$F/eval-fp8kv-compare.py']:
    ast.parse(pathlib.Path(p).read_text()); print('py ok', pathlib.Path(p).name)
"
