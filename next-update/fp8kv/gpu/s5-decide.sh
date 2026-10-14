#!/usr/bin/env bash
# S5 (CPU): the pre-registered quality decision (eval-fp8kv-compare.py, rule fp8kv-quality-rule-1). Run ids come from
# the reports, not chosen by hand. Speed/VRAM summaries were written by s3/s4 (results/ab-*.json).
set -uo pipefail
source "$(dirname "$0")/common.sh"
IDS=$("$PY" -I "$F/gpu/run-ids.py") || die "run ids"
note "S5 fp8 run ids $IDS"
cd /tmp && "$PY" -I -B "$F/eval-fp8kv-compare.py" --quality quality-main-release-fp8kv \
  --baseline fp8kv-eval-release --candidate fp8kv-eval-fp8 \
  --functional-baseline functional-main-release --functional-candidate functional-main-release-fp8kv \
  --fp8-run-ids "$IDS" --out "$F/results/decision-quality.json" | tee -a "$LOG"
