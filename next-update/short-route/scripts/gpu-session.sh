#!/bin/bash
# Runs INSIDE the GPU lock. Installs the short-route arm configs (new files, removed on exit), runs
# driver.py (<= 19 min) and always stops an experiment unit it started.
# Usage: gpu-session.sh SESSION ARMS   (e.g. s1 rdp)
set -uo pipefail
HERE=/path/to/qwen36-workspace/work/next-update/short-route
N=/home/USER/qwen36-api/optimization
py=/home/USER/qwen36-api/.venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cleanup() {
  if systemctl --user is-active --quiet qwen36-experiment && \
     systemctl --user show -p ExecStart qwen36-experiment | grep -q 'shortroute-'; then
    systemctl --user stop qwen36-experiment
  fi
  rm -f "$N/config/shortroute-r.json" "$N/config/shortroute-d.json" "$N/config/shortroute-p.json"
}
trap cleanup EXIT INT TERM
echo "GPU lock held $(date -Is)"
if systemctl --user is-active --quiet qwen36-inference.service; then echo 'production active; refusing' >&2; exit 73; fi
if systemctl --user is-active --quiet qwen36-experiment.service; then echo 'an experiment is active under our lock; refusing' >&2; exit 74; fi
# The shim mirrors the patch against these exact sources: refuse if native differs from the workspace tree.
for f in engine/layer_major.py moe/offload_cache.py scheduler/layer_major.py; do
  a=$(sha256sum < "$N/python/freetoken/$f" | cut -c1-64); b=$(sha256sum < "$HERE/orig/$f" | cut -c1-64)
  [ "$a" = "$b" ] || { echo "native $f differs from orig copy; refusing" >&2; exit 78; }
done
for a in r d p; do install -m 600 "$HERE/configs/shortroute-$a.json" "$N/config/shortroute-$a.json"; done
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,clocks.sm,temperature.gpu --format=csv,noheader
cd /tmp
timeout 1140 "$py" -B "$HERE/scripts/driver.py" "$1" "$2"
echo "driver exit $? $(date -Is)"
