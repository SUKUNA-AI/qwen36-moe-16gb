#!/usr/bin/env bash
F=/path/to/qwen36-workspace/work/next-update/fp8kv
T=$(mktemp -d)
cd /tmp && /home/USER/qwen36-api/.venv/bin/python -I -B $F/summarize-ab.py --rounds comb2-ab-1,comb2-ab-2,comb2-ab-3 --base main-layermajor-gdn --fp8 main-combined --out $T/ab-test.json | cut -c1-900
rm -rf $T
