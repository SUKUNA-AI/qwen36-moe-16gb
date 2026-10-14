#!/usr/bin/env bash
F=/path/to/qwen36-workspace/work/next-update/fp8kv
cd /tmp && /home/USER/qwen36-api/.venv/bin/python -I $F/scripts/gen_configs.py $F
N=/home/USER/qwen36-api/optimization
cmp $N/config/main-release.json $F/configs/main-release.json && echo "native main-release.json == workspace copy"
cmp $N/config/long256-release.json /path/to/qwen36-workspace/outputs/qwen36-api/optimization/long256-release.json && echo "native long256-release.json == workspace"
for f in teacher-fixture.json; do ls -la $N/config/$f; done; ls -ld $N/reports/logit-reference
