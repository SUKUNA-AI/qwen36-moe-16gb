#!/usr/bin/env bash
# Runs INSIDE the GPU lock: installs the S4 arm configs (removed on exit), runs driver.py with the
# given arguments and always stops an experiment unit it started.
# Usage: session.sh DRIVER_ARGS...
set -uo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../../.." && pwd)
native=$HOME/qwen36-api/optimization
py=$HOME/qwen36-api/.venv/bin/python
cleanup() {
  if systemctl --user is-active --quiet qwen36-experiment && \
     systemctl --user show -p ExecStart qwen36-experiment | grep -q 's4-'; then
    systemctl --user stop qwen36-experiment
  fi
  "$py" -I "$here/make_configs.py" "$repo" "$native/config" --remove
}
trap cleanup EXIT INT TERM
: "${QWEN36_MODEL_DIR:?set QWEN36_MODEL_DIR to the pinned checkpoint directory}"
export QWEN36_MODEL_DIR
if systemctl --user is-active --quiet qwen36-inference.service; then echo 'production active; refusing' >&2; exit 73; fi
if systemctl --user is-active --quiet qwen36-experiment.service; then echo 'an experiment is active under our lock; refusing' >&2; exit 74; fi
"$py" -I "$here/make_configs.py" "$repo" "$native/config"
cd /tmp
QWEN36_DATA_ROOT=$native "$py" -B "$here/driver.py" "$@"
echo "driver exit $? $(date -Is)"
