#!/usr/bin/env bash
# S3 (GPU, 3 lock sessions: ~9, ~9, ~13 min): rotating paired decode/TTFT A/B (AB, BA, AB+hr), sustained 60 s warm-up,
# fresh 16K/64K/120K inputs, exactly 512 generated tokens. Round 3 adds the headroom arm main-release-fp8kv-hr
# (same FP8 numerics, memory_ratio .665 -> ~3167 slots, ~+1.25 GiB free). screen-chunk.py takes the GPU lock itself
# (non-blocking, exit 75 when busy): retry75, never wrap it in flock. No profiler.
set -uo pipefail
source "$(dirname "$0")/common.sh"
export QWEN36_BENCH_WARMUP_S=60
cd "$N"
retry75 "S3 round1" "$PY" "$N/screen-chunk.py" --name fp8kv-ab-1 --profiles main-release,main-release-fp8kv \
  --lengths 16384,65536,120000 --output 512 >> "$LOG" 2>&1
retry75 "S3 round2" "$PY" "$N/screen-chunk.py" --name fp8kv-ab-2 --profiles main-release-fp8kv,main-release \
  --lengths 16384,65536,120000 --output 512 >> "$LOG" 2>&1
retry75 "S3 round3" "$PY" "$N/screen-chunk.py" --name fp8kv-ab-3 --profiles main-release,main-release-fp8kv,main-release-fp8kv-hr \
  --lengths 16384,65536,120000 --output 512 >> "$LOG" 2>&1
"$PY" -I "$F/summarize-ab.py" --rounds fp8kv-ab-1,fp8kv-ab-2,fp8kv-ab-3 --base main-release --fp8 main-release-fp8kv \
  --extra main-release-fp8kv-hr --out "$F/results/ab-main128.json" | tee -a "$LOG"
