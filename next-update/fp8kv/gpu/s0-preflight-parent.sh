#!/usr/bin/env bash
# S0 (CPU, PARENT ONLY: writes $N/config; the allowlist patch must already be applied + synced). No GPU, no lock.
set -euo pipefail
source "$(dirname "$0")/common.sh"
note "S0 start"
units_idle || die "a model unit is active"
# 1) configs + scales (byte copies, mode 600). Copies under outputs/ are optional but must then be byte-identical.
for f in main-release-fp8kv main-release-fp8kv-hr long256-release-fp8kv kv-scales-bound; do
  install -m 600 "$F/configs/$f.json" "$N/config/$f.json"
done
sha256sum "$N/config/main-release-fp8kv.json" "$N/config/main-release-fp8kv-hr.json" \
  "$N/config/long256-release-fp8kv.json" "$N/config/kv-scales-bound.json" | tee -a "$LOG"
sha256sum "$N/config/kv-scales-bound.json" | grep -q '^e9888b62c30f04e355902bad68ae908bc2912d143a2c26738974e7026d0b1de0 ' || die "scales sha"
cmp "$N/config/main-release.json" "$F/configs/main-release.json" || die "native main-release.json changed since the plan was frozen"
cmp "$N/config/long256-release.json" "$O/long256-release.json" || die "native long256-release.json differs from workspace"
# 2) allowlist (allowlist-fp8kv.patch on work/FreeToken-opt, then the usual sync-v5 flow) -> verify in native
grep -q "'teacher-main-release-fp8kv'" "$N/python/freetoken/engine/qualification.py" || die "allowlist name missing in native qualification.py"
# 3) FP8-KV code path present in the native tree (unchanged since 2026-10-13)
sha256sum "$N/python/freetoken/kvcache/mha_pool.py" "$N/python/freetoken/kernel/triton/fp8_kv.py" \
  "$N/python/freetoken/attention/fi.py" | tee -a "$LOG"
# 4) FlashInfer 0.6.18.post1 sm_120f e4m3 JIT modules already built (no compile expected at start)
ls -d /home/USER/.cache/flashinfer/0.6.18.post1/120f/cached_ops/batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_* \
  /home/USER/.cache/flashinfer/0.6.18.post1/120f/cached_ops/batch_decode_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_* | tee -a "$LOG"
# 5) eval freeze still valid (bench.py, eval-v1 sources, 3 runtime files unchanged) + decision self-test
(cd /tmp && CUDA_VISIBLE_DEVICES= "$PY" -I -B "$F/eval-fp8kv.py" verify) | tee -a "$LOG" | grep -q '"ok": true' || die "eval freeze invalid"
(cd /tmp && CUDA_VISIBLE_DEVICES= "$PY" -I -B "$F/tests/test_compare_cpu.py") | tail -1 | tee -a "$LOG" | grep -q '"ok": true' || die "compare self-test"
note "S0 OK"
