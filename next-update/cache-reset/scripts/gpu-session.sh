#!/usr/bin/env bash
# Phase-1 GPU validation of the harness-v7 clean-cache teacher step. Two separate GPU sessions (one per profile),
# each under the shared GPU-exclusive lock (blocking wait up to 1 h), each <= 19 min.
#   MSYS_NO_PATHCONV=1 wsl -d archlinux -u USER -- bash /path/to/qwen36-workspace/work/next-update/cache-reset/scripts/gpu-session.sh [main|long256|both]
set -uo pipefail
here=/path/to/qwen36-workspace/work/next-update/cache-reset
lock=/home/USER/qwen36-api/run/gpu-exclusive.lock
mkdir -p "$here/gpu-results"
which="${1:-both}"
overall=0
for p in teacher-main-release teacher-long256-release; do
  case "$which:$p" in both:*|main:teacher-main-release|long256:teacher-long256-release) ;; *) continue;; esac
  log="$here/gpu-results/${p}.log"
  echo "waiting for GPU lock $(date -u +%FT%TZ) ($p)" | tee "$log"
  flock -w 3600 -E 75 "$lock" bash "$here/scripts/gpu-inner.sh" "$p" >> "$log" 2>&1
  rc=$?
  echo "$p rc=$rc $(date -u +%FT%TZ)" | tee -a "$log"
  [ "$rc" = 0 ] || overall=$rc
done
exit "$overall"
