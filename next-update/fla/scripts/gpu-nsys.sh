#!/usr/bin/env bash
# FLA root cause, GPU phase 2: two Nsight Systems captures (release config, FLA on / off) of a
# fresh 16K and a fresh 120K layer-major prefill. Diagnostic only (no speed claims).
# Usage (WSL): bash gpu-nsys.sh            -> waits for phase1.done, then queues on the GPU lock
#   inner mode (under flock): bash gpu-nsys.sh --inner
set -uo pipefail
umask 077
W=/path/to/qwen36-workspace
D=$W/work/next-update/fla
N=/home/USER/qwen36-api/optimization
PY=/home/USER/qwen36-api/.venv/bin/python
LOCK=/home/USER/qwen36-api/run/gpu-exclusive.lock
MARK=/home/USER/qwen36-api/run/phase1.done
LOG=$D/results/gpu-nsys.log
ARMS=${ARMS:-"on off"}
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

if [ "${1:-}" != "--inner" ]; then
  while [ ! -e "$MARK" ]; do sleep 60; done
  # 75 = lock wait timeout, 73 = a foreign unit was still active when we got the lock: re-queue
  for attempt in 1 2 3 4 5 6; do
    log "phase1.done present; queueing on the GPU lock (attempt $attempt)"
    flock -w 3600 -E 75 "$LOCK" bash "$0" --inner; rc=$?
    log "session rc=$rc"
    case $rc in 75|73) sleep 5;; *) break;; esac
  done
  exit $rc
fi

# ---- inside the GPU lock ----
SESSION_END=$(( $(date +%s) + 19 * 60 ))
MYINV=""
cleanup() {
  if [ -n "$MYINV" ] && [ "$(systemctl --user show -p InvocationID --value qwen36-experiment 2>/dev/null)" = "$MYINV" ]; then
    log "trap: stopping my experiment unit ($MYINV)"
    systemctl --user stop qwen36-experiment || true
  fi
}
trap cleanup EXIT INT TERM
if systemctl --user is-active --quiet qwen36-inference; then log "REFUSE: qwen36-inference active"; exit 73; fi
# the previous lock holder may still be stopping its unit: give it up to 90 s
for i in $(seq 1 45); do systemctl --user is-active --quiet qwen36-experiment || break; sleep 2; done
if systemctl --user is-active --quiet qwen36-experiment; then log "REFUSE: qwen36-experiment active"; exit 73; fi
for f in profile-request.py bench.py; do
  a=$(sha256sum < "$N/$f" | cut -c1-12); b=$(sha256sum < "$W/outputs/qwen36-api/optimization/$f" | cut -c1-12)
  log "native $f $a workspace $b"
done
grep -q 'profile-final-\*' "$N/start-experiment.sh" || { log "start-experiment.sh does not route profile-final-*"; exit 78; }

for arm in $ARMS; do
  prof=profile-final-fla-$arm
  now=$(date +%s)
  if [ $(( SESSION_END - now )) -lt 420 ]; then log "skip $arm: less than 7 min left in the 19-min session"; continue; fi
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,clocks.sm --format=csv,noheader | sed "s/^/[$arm pre-start gpu] /" | tee -a "$LOG"
  log "start $prof"
  bash "$N/start-experiment.sh" "$prof" >> "$LOG" 2>&1 || { log "start failed"; continue; }
  sleep 2
  MYINV=$(systemctl --user show -p InvocationID --value qwen36-experiment)
  log "unit invocation $MYINV"
  # /v1/models answers before the engine is up (first attempt got 503); wait for /health ready
  timeout 450 "$PY" "$N/check-experiment.py" 420 >> "$LOG" 2>&1 || {
    log "engine not ready: stopping"; systemctl --user stop qwen36-experiment || true; MYINV=""; continue; }
  log "engine ready"
  left=$(( SESSION_END - $(date +%s) - 30 ))
  timeout "$left" "$PY" "$N/profile-request.py" run --profile "$prof" --wait-ready 420 --wait-exit 360 >> "$LOG" 2>&1
  log "profile-request rc=$?"
  # nsys stop-shutdown ends the unit after the capture window; cap the wait
  while systemctl --user is-active --quiet qwen36-experiment && [ "$(date +%s)" -lt $(( SESSION_END - 20 )) ]; do sleep 5; done
  if systemctl --user is-active --quiet qwen36-experiment; then
    log "unit still active at the session cap: stopping it"
    systemctl --user stop qwen36-experiment || true
  fi
  MYINV=""
  base=$("$PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_base"])' "$N/reports/profile-final-current.json")
  log "report base $base"
  ls -la "$N/reports/$base".* 2>&1 | tee -a "$LOG"
  echo "$base" > "$D/results/nsys-base-$arm.txt"
done
log "GPU part done"
exit 0
