#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python=${QWEN36_CPU_PYTHON:-python3}
export CUDA_VISIBLE_DEVICES=''
exec "$python" "$here/cpu_checks.py" "$@"
