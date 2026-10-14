#!/usr/bin/env bash
# S4 (GPU, OPTIONAL, 2 lock sessions: ~11-12 + ~5 min): the 256K profile with FP8 KV (2987 expert slots instead of 1481).
# Quality numerics are identical to main-release-fp8kv (same scales/kernels); this measures speed/VRAM at 120K/250K
# and the 250K functional fixture (baseline: final-f1-l256-a1-functional.json, long-facts-250000 PASS, TTFT 80.4 s).
set -uo pipefail
source "$(dirname "$0")/common.sh"
export QWEN36_BENCH_WARMUP_S=60
cd "$N"
retry75 "S4 l256 A/B" "$PY" "$N/screen-chunk.py" --name fp8kv-l256-ab-1 --profiles long256-release,long256-release-fp8kv \
  --lengths 120000,250000 --output 512 >> "$LOG" 2>&1
"$PY" -I "$F/summarize-ab.py" --rounds fp8kv-l256-ab-1 --base long256-release --fp8 long256-release-fp8kv \
  --out "$F/results/ab-long256.json" | tee -a "$LOG"
note "S4 functional long256-release-fp8kv @250000"
flock -w 3600 -E 75 "$LOCK" bash "$F/gpu/functional-session.sh" long256-release-fp8kv functional-long256-release-fp8kv-250k 250000 >> "$LOG" 2>&1
note "S4 functional rc=$?"
