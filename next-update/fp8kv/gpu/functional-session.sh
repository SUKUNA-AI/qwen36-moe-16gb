#!/usr/bin/env bash
# One functional session (5 fixtures; long-facts at LONG tokens) for PROFILE under the caller's GPU lock.
set -uo pipefail
source "$(dirname "$0")/common.sh"
PROFILE=$1; NAME=$2; LONG=$3
units_idle || die "a model unit is active"
trap 'systemctl --user stop qwen36-experiment 2>/dev/null' EXIT
bash "$N/start-experiment.sh" "$PROFILE" || die "start $PROFILE"
"$PY" "$N/check-experiment.py" 420 || die "readiness $PROFILE"
cd "$N" && "$PY" "$N/quality.py" --name "$NAME" --long "$LONG"
