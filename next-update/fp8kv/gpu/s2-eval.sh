#!/usr/bin/env bash
# S2 (GPU, 2 lock sessions of ~8-9 min each): eval-v1 (60 cases) on the release and on FP8 KV, same build, same freeze.
# Protocol: the first VALID run of each profile counts; an INVALID run may be repeated under a NEW name only.
set -uo pipefail
source "$(dirname "$0")/common.sh"
for spec in main-release:fp8kv-eval-release main-release-fp8kv:fp8kv-eval-fp8; do
  P=${spec%%:*}; NAME=${spec##*:}
  note "S2 eval $P -> $NAME"
  flock -w 3600 -E 75 "$LOCK" bash "$F/gpu/eval-session.sh" "$P" "$NAME" >> "$LOG" 2>&1
  note "S2 $P rc=$?"
done
