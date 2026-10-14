#!/usr/bin/env bash
# Phase-3 GPU driver for the synclean-refill re-evaluation (pre-registered in prereg.json).
#   bash phase3-driver.sh [refill] [synclean]        (default: refill synclean; RUN=a1 names the reports)
#   PRECHECK_ONLY=1 RUN=pc bash phase3-driver.sh     CPU-only: check configs/allowlist/sync, no phase wait, no GPU
# Launch as a TRACKED background task (never nohup inside a short-lived wsl call).
# Per feature: preconditions (CPU) -> teacher (qualify.py, allowlisted name) -> 3 rounds x 2 outputs of paired,
# rotating screen sessions (base main-release vs candidate, warm-up 60 s) -> evaluate.py (decision per prereg).
# Each GPU session: blocking flock -w 3600 on the shared lock, <= 20 min, experiment unit stopped on exit.
# Resumable: finished sessions (summary report exists) and a recorded QUALIFY_PASS are skipped on rerun.
set -uo pipefail
W=/path/to/qwen36-workspace
D=$W/work/next-update/synclean-refill
O=$W/outputs/qwen36-api/optimization
N=/home/USER/qwen36-api/optimization
PY=/home/USER/qwen36-api/.venv/bin/python
LOCK=/home/USER/qwen36-api/run/gpu-exclusive.lock
PHASE=/home/USER/qwen36-api/run/phase2.done
RUN=${RUN:-a1}
mkdir -p "$D/results"
LOG=$D/results/driver-$RUN.log
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
features=("$@"); [ ${#features[@]} -eq 0 ] && features=(refill synclean)
export QWEN36_BENCH_WARMUP_S=60
SYNCLEAN_FILES="attention/fi.py attention/linear.py engine/layer_major.py kernel/cuda_gdn.py kernel/fla/index.py kernel/fla/utils.py moe/offload_cache.py sync_lean.py"

note "driver start run=$RUN features=${features[*]} prereg.json sha256 $(sha256sum "$D/prereg.json" | cut -c1-16)"
if [ "${PRECHECK_ONLY:-0}" != 1 ]; then until [ -e "$PHASE" ]; do sleep 60; done; note "phase gate: $PHASE present"; fi

gpu() {  # gpu <label> <session.sh args...>: blocking lock wait, retried while the wait times out (rc 75)
  local label=$1; shift; local rc t=0
  while :; do
    flock -w 3600 -E 75 "$LOCK" bash "$D/session.sh" "$@" >> "$LOG" 2>&1; rc=$?
    [ "$rc" -ne 75 ] && break
    t=$((t + 1)); [ "$t" -ge 6 ] && break
    note "$label: lock wait timed out, retry $t"
  done
  note "$label rc=$rc"; return "$rc"
}

qcheck() {  # teacher gate from reports/quality-<candidate>.json (same keys as the parent drivers)
  "$PY" -I - "$N/reports/quality-$1.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
def find(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key: return v
            r = find(v, key)
            if r is not None: return r
    elif isinstance(o, list):
        for v in o:
            r = find(v, key)
            if r is not None: return r
    return None
out = {k: find(d, k) for k in ('expected', 'observed', 'max_abs', 'forced_targets_complete', 'new_functional_failures', 'ppl_gate_pass')}
print(json.dumps(out))
sys.exit(0 if out['max_abs'] == 0 and not out['new_functional_failures'] and out['forced_targets_complete'] is not False else 3)
PYEOF
}

precheck() {  # precheck <feature> <candidate>
  local feat=$1 cand=$2 ok=0
  [ -f "$N/config/$cand.json" ] || { note "PRECHECK $feat: $N/config/$cand.json missing (parent: sync configs/$cand.json)"; ok=1; }
  [ -f "$N/config/$cand.json" ] && ! cmp -s "$N/config/$cand.json" "$D/configs/$cand.json" && { note "PRECHECK $feat: native $cand.json differs from configs/$cand.json"; ok=1; }
  for p in main-release "$cand"; do
    if [ -f "$O/$p.json" ] && ! cmp -s "$O/$p.json" "$N/config/$p.json"; then note "PRECHECK $feat: workspace $p.json differs from native"; ok=1; fi
  done
  grep -q "'teacher-$cand'" "$N/python/freetoken/engine/qualification.py" || { note "PRECHECK $feat: allowlist name teacher-$cand missing in native qualification.py"; ok=1; }
  case "$feat" in
    refill)
      grep -q 'QWEN36_LAYER_MAJOR_REFILL_POLICY' "$N/python/freetoken/moe/slot_refill.py" 2>/dev/null || { note "PRECHECK refill: slot_refill.py missing in native"; ok=1; } ;;
    synclean)
      [ -f "$N/python/freetoken/sync_lean.py" ] || { note "PRECHECK synclean: sync_lean.py not synced"; ok=1; }
      grep -q 'def seed_empty_sequence' "$N/python/freetoken/kernel/cuda_gdn.py" || { note "PRECHECK synclean: cuda_gdn.py without seed_empty_sequence"; ok=1; }
      grep -q '_pooled(meta)' "$N/python/freetoken/engine/layer_major.py" || { note "PRECHECK synclean: layer_major.py without the grouped-path rebase"; ok=1; }
      for f in $SYNCLEAN_FILES; do
        cmp -s "$N/python/freetoken/$f" "$D/tree/python/freetoken/$f" || note "PRECHECK synclean (info): native $f != rebased tree copy (other synced changes?) $(sha256sum "$N/python/freetoken/$f" 2>/dev/null | cut -c1-12)"
      done ;;
  esac
  return $ok
}

for feat in "${features[@]}"; do
  case "$feat" in
    refill)   cand=main-release-refill-restore; LENS=16384,65536,120000; OUTS="1024 512" ;;
    synclean) cand=main-release-synclean;       LENS=16384,120000;       OUTS="512 128" ;;
    *) note "unknown feature $feat"; continue ;;
  esac
  note "== feature $feat candidate $cand"
  if ! precheck "$feat" "$cand"; then note "NOT_RUN $feat: preconditions not met"; continue; fi
  if [ "${PRECHECK_ONLY:-0}" = 1 ]; then note "PRECHECK $feat OK"; continue; fi
  # 1. teacher 0-diff (skipped when this run already recorded a pass)
  if grep -q "QUALIFY_PASS $cand" "$LOG"; then
    note "teacher $cand already passed in this run"
  else
    t0=$(date +%s)
    gpu "teacher $cand" teacher "$cand"
    q=$N/reports/quality-$cand.json
    if [ ! -f "$q" ] || [ "$(stat -c %Y "$q")" -lt "$t0" ]; then
      note "QUALIFY_FAIL $cand (no fresh quality-$cand.json) -> REJECT $feat (A/B skipped)"; continue
    fi
    if qcheck "$cand" | tee -a "$LOG"; then note "QUALIFY_PASS $cand"; else note "QUALIFY_FAIL $cand -> REJECT $feat (A/B skipped)"; continue; fi
  fi
  # 2. paired rotating A/B: 3 rounds x outputs; one output per session, arm order alternates
  for r in 1 2 3; do
    j=0
    for out in $OUTS; do
      if [ $(((r + j) % 2)) -eq 1 ]; then P=main-release,$cand; else P=$cand,main-release; fi
      name=srf-$RUN-$feat-r$r-o$out
      if [ -f "$N/reports/$name-chunk-screen.json" ]; then
        note "skip $name (summary exists)"
      else
        gpu "screen $name [$P]" screen --name "$name" --profiles "$P" --lengths "$LENS" --output "$out"
      fi
      "$PY" -I "$D/evaluate.py" --run "$RUN" --feature "$feat" --peek "$name" 2>&1 | tee -a "$LOG"
      j=$((j + 1))
    done
  done
  # 3. decision per prereg.json
  "$PY" -I "$D/evaluate.py" --run "$RUN" --feature "$feat" 2>&1 | tee -a "$LOG"
done
note "driver done"
