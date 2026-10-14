#!/usr/bin/env bash
# One GPU session (run UNDER the GPU-exclusive flock by gpu-session.sh): harness-v7 teacher validation of ONE
# allowlisted release teacher profile. Max 19 min (hard timeout), experiment unit stopped on exit.
set -euo pipefail
profile="$1"
here=/path/to/qwen36-workspace/work/next-update/cache-reset
py=/home/USER/qwen36-api/.venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
started=0
cleanup() { if [ "$started" = 1 ]; then systemctl --user stop qwen36-experiment >/dev/null 2>&1 || true; fi; }
trap cleanup EXIT
if systemctl --user is-active --quiet qwen36-inference.service; then
  echo 'qwen36-inference (production) is active; refusing' >&2; exit 73
fi
if systemctl --user is-active --quiet qwen36-experiment.service; then
  echo 'qwen36-experiment is already active (not ours); refusing to touch it' >&2; exit 73
fi
echo "session start $(date -u +%FT%TZ) profile=$profile"
started=1
cd /tmp
rc=0
timeout -k 30 1140 "$py" -B "$here/scripts/validate_v7.py" --profile "$profile" --out-dir "$here/gpu-results" || rc=$?
echo "session end $(date -u +%FT%TZ) rc=$rc"
exit "$rc"
