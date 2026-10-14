# Shared variables for the FP8-KV GPU plan (source me). Production stays stopped; experiments only.
W=/path/to/qwen36-workspace
F=$W/work/next-update/fp8kv
O=$W/outputs/qwen36-api/optimization
N=/home/USER/qwen36-api/optimization
PY=/home/USER/qwen36-api/.venv/bin/python
LOCK=/home/USER/qwen36-api/run/gpu-exclusive.lock
LOG=$F/results/gpu-plan.log
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
die() { note "STOP: $*"; exit 1; }
units_idle() { ! systemctl --user is-active --quiet qwen36-inference && ! systemctl --user is-active --quiet qwen36-experiment; }
# Tools that take the GPU lock themselves (screen-chunk.py): retry while busy (75) or a unit is active (73).
retry75() {
  local label=$1; shift; local t=0 rc
  while :; do "$@"; rc=$?; { [ "$rc" -ne 75 ] && [ "$rc" -ne 73 ]; } && break; t=$((t+1)); [ "$t" -ge 1200 ] && break; sleep 3; done
  note "$label rc=$rc"; return "$rc"
}
