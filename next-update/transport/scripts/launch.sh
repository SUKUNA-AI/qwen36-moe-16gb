#!/usr/bin/env bash
# Start ONE qwen36-experiment unit for an S4 arm config (config/s4-ARM.json). Call only while
# holding the GPU lock; refuses when production or another experiment is active.
# Usage: launch.sh ARM [plain|nsys] [OUT_PREFIX]
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$HOME/qwen36-api/optimization
profile="s4-$1"
mode=${2:-plain}
[[ "$profile" =~ ^s4-[a-z0-9-]{1,40}$ ]] || { echo "bad arm $1" >&2; exit 64; }
[ -f "$root/config/$profile.json" ] || { echo "missing $root/config/$profile.json" >&2; exit 66; }
if systemctl --user is-active --quiet qwen36-inference.service; then echo 'production active' >&2; exit 73; fi
if systemctl --user is-active --quiet qwen36-experiment.service; then echo 'experiment active' >&2; exit 73; fi
systemd-run --user --unit=qwen36-experiment --collect \
  --property=WorkingDirectory="$root" --property=MemoryHigh=32G \
  --property=MemoryMax=36G --property=MemorySwapMax=0 \
  --property=UMask=0077 --property=TimeoutStopSec=60 \
  /bin/bash "$here/unit.sh" "$profile" "$mode" "${3:-/dev/null}"
