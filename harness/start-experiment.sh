#!/usr/bin/env bash
# Start ONE experiment unit for an explicit profile (harness V6).
# V6 (P-1/P-2): launchers/scripts are delivered only by sync-v5.sh; nothing is copied
# wholesale from the workspace any more. Only the started profile's own config is
# refreshed from the workspace by its exact name (qualify.py writes teacher-<cand>.json
# there); a native-only config (e.g. profile-final-*-ncu<k>) is used as is.
set -euo pipefail
root=/home/USER/qwen36-api/optimization
workspace='/path/to/qwen36-workspace/outputs/qwen36-api/optimization'
root=${QWEN36_DATA_ROOT:-$root}
workspace=${QWEN36_HARNESS_SOURCE:-$workspace}
profile="${1:-baseline64}"
if ! [[ "$profile" =~ ^[a-z0-9][a-z0-9._-]{0,120}$ ]]; then
  echo "Invalid profile name: $profile" >&2
  exit 64
fi
if systemctl --user is-active --quiet qwen36-inference.service; then
  echo 'Production is active; stop it explicitly before experimenting' >&2
  exit 73
fi
if systemctl --user is-active --quiet qwen36-experiment.service; then
  echo 'An experiment is already active' >&2
  exit 73
fi
source_config="$workspace/$profile.json"
target_config="$root/config/$profile.json"
if [ -f "$source_config" ] && [ ! -L "$source_config" ]; then
  if ! cmp -s -- "$source_config" "$target_config"; then
    staged=$(mktemp "$root/config/.start-experiment.XXXXXX")
    cp -- "$source_config" "$staged"
    chmod 600 -- "$staged"
    mv -f -T -- "$staged" "$target_config"
  fi
elif [ ! -f "$target_config" ] || [ -L "$target_config" ]; then
  echo "Profile config missing in workspace and native: $profile.json" >&2
  exit 66
fi
launch=run-experiment.sh
case "$profile" in profile64|profile-cuda128|profile-main128|profile-final-*) launch=profile.sh;; esac
systemd-run --user --unit=qwen36-experiment --collect \
  --property=WorkingDirectory="$root" --property=MemoryHigh=32G \
  --property=MemoryMax=36G --property=MemorySwapMax=0 \
  --property=UMask=0077 --property=TimeoutStopSec=30 \
  /bin/bash "$root/$launch" "$profile"
