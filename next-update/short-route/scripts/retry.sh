#!/bin/bash
# Re-queue a session on flock wait timeout (exit 75), per parent instruction. Usage: retry.sh SESSION ARMS
HERE=/path/to/qwen36-workspace/work/next-update/short-route
while true; do
  bash $HERE/scripts/gpu-run.sh "$1" "$2"; rc=$?
  [ $rc -eq 75 ] || exit $rc
  echo "lock wait timed out (75) at $(date -Is); re-queueing $1"
done
