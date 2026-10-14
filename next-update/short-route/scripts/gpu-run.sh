#!/bin/bash
# GPU entry: phase gate (phase1.done), then the GPU lock (blocking, 1 h max) and gpu-session.sh.
# Usage: gpu-run.sh SESSION ARMS
set -uo pipefail
HERE=/path/to/qwen36-workspace/work/next-update/short-route
RUN=/home/USER/qwen36-api/run
until [ -e "$RUN/phase1.done" ]; do sleep 60; done
echo "phase marker seen $(date -Is); waiting for the GPU lock"
flock -w 3600 -E 75 "$RUN/gpu-exclusive.lock" bash "$HERE/scripts/gpu-session.sh" "$1" "$2"
rc=$?
echo "gpu-session exit $rc at $(date -Is)"
exit $rc
