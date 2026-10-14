#!/usr/bin/env bash
# ONE GPU session (<= 20 min). Run ONLY as: flock -w 3600 -E 75 /home/USER/qwen36-api/run/gpu-exclusive.lock bash session.sh ...
#   session.sh teacher <candidate>                 qualify.py (teacher + functional + compare)
#   session.sh screen --name N --profiles A,B --lengths L --output O    paired screen (QWEN36_BENCH_WARMUP_S from env)
# Never touches qwen36-inference; stops qwen36-experiment on exit (while the lock is held only this session starts it).
set -uo pipefail
D=/path/to/qwen36-workspace/work/next-update/synclean-refill
N=/home/USER/qwen36-api/optimization
PY=/home/USER/qwen36-api/.venv/bin/python
started=0
cleanup() {
  if [ "$started" = 1 ] && systemctl --user is-active --quiet qwen36-experiment; then
    systemctl --user stop qwen36-experiment
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
systemctl --user is-active --quiet qwen36-inference && { echo "REFUSED: production unit active"; exit 73; }
systemctl --user is-active --quiet qwen36-experiment && { echo "REFUSED: an experiment unit is already active"; exit 73; }
cd "$N" || exit 1
kind=$1; shift
started=1
echo "$(date -u +%FT%TZ) session $kind $* (warmup ${QWEN36_BENCH_WARMUP_S:-0}s)"
case "$kind" in
  teacher) timeout -k 30 1170 "$PY" "$N/qualify.py" --candidate "$1"; rc=$? ;;
  screen)  timeout -k 30 1170 "$PY" "$D/screen-wrapper.py" "$@"; rc=$? ;;
  *) echo "unknown session kind $kind"; rc=2 ;;
esac
echo "$(date -u +%FT%TZ) session $kind rc=$rc"
exit $rc
