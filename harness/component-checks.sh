#!/usr/bin/env bash
set -euo pipefail
root=/home/USER/qwen36-api
export CUDA_HOME="$root/cuda-13.4"
export PATH="$CUDA_HOME/bin:$root/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:/usr/lib/wsl/lib:${LIBRARY_PATH:-}"
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15
export NVCC_CCBIN=/usr/bin/g++-15 MAX_JOBS=4 OMP_NUM_THREADS=16
export PYTHONPATH="$root/optimization/python:$root/optimization"
export QWEN36_DATA_ROOT="$root/optimization"
cd "$QWEN36_DATA_ROOT"
python test-kv.py
python test-spec-gpu.py
python test-mtp.py
python test-complement.py --actual --name complement-actual
QWEN36_COMPLEMENT_TRANSPORT=cuda QWEN36_COMPLEMENT_LIBRARY="$root/optimization/build/libqwen36_complement.so" \
    python test-complement.py --actual --name complement-actual-cuda
