#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/USER/qwen36-api
export CUDA_HOME="$ROOT/cuda-13.4"
export PATH="$CUDA_HOME/bin:$ROOT/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/wsl/lib:${LIBRARY_PATH:-}"
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 NVCC_CCBIN=/usr/bin/g++-15 MAX_JOBS=4
"$ROOT/.venv/bin/python" /path/to/qwen36-workspace/outputs/qwen36-api/check-kernels.py
