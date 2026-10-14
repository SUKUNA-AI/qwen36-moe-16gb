#!/usr/bin/env bash
# S1 (GPU, one lock session, ~10-12 min): teacher forcing (20 cases / 943 tokens) + 5 functional fixtures +
# compare-quality vs teacher-bf128 (= release reference). qualify.py is the project's standard path; it stops the
# experiment unit in its own finally. It writes teacher-main-release-fp8kv.json into outputs/ (parent harness behaviour).
set -uo pipefail
source "$(dirname "$0")/common.sh"
units_idle || die "a model unit is active"
note "S1 qualify main-release-fp8kv"
cd "$N" && flock -w 3600 -E 75 "$LOCK" "$PY" "$N/qualify.py" --candidate main-release-fp8kv >> "$LOG" 2>&1
note "S1 rc=$?"
"$PY" -I "$F/gpu/show-quality.py" | tee -a "$LOG"
