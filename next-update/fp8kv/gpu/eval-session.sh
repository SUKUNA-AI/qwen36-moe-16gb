#!/usr/bin/env bash
# One eval-v1 (FP8-KV freeze) session for PROFILE, run under the GPU lock held by the caller (flock).
# Stops the experiment unit it started on every exit path.
set -uo pipefail
source "$(dirname "$0")/common.sh"
PROFILE=$1; NAME=$2
units_idle || die "a model unit is active"
trap 'systemctl --user stop qwen36-experiment 2>/dev/null' EXIT
bash "$N/start-experiment.sh" "$PROFILE" || die "start $PROFILE"
"$PY" "$N/check-experiment.py" 420 || die "readiness $PROFILE"
cd /tmp && "$PY" -I -B "$F/eval-fp8kv.py" run --name "$NAME" --expect-profile "$PROFILE"
