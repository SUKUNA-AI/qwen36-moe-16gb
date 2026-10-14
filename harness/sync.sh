#!/usr/bin/env bash
set -euo pipefail
root=/home/USER/qwen36-api/optimization
workspace='/path/to/qwen36-workspace'
for unit in qwen36-inference qwen36-experiment; do
  if systemctl --user is-active --quiet "$unit"; then
    echo "Refusing source replacement while $unit is active" >&2
    exit 73
  fi
done
cp -a "$workspace/work/FreeToken-opt/python/freetoken/." "$root/python/freetoken/"
cp "$workspace/outputs/qwen36-api/"*.py "$root/"
cp "$workspace/outputs/qwen36-api/optimization/"*.py "$root/"
chmod 700 "$root" "$root/reports"
chmod 600 "$root/"*.py
