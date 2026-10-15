#!/usr/bin/env bash
# Body of the qwen36-experiment unit for an S4 transport arm: run-experiment.sh's environment,
# with the S4 source overlay first on PYTHONPATH; "nsys" wraps it in a cudaProfilerApi capture.
# Usage (from launch.sh only): unit.sh PROFILE plain|nsys OUT_PREFIX
set -euo pipefail
root=$HOME/qwen36-api
overlay=$HOME/.cache/qwen36-cloud-review/s4-overlay/python
export CUDA_HOME="$root/cuda-13.4"
export PATH="$CUDA_HOME/bin:$root/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:/usr/lib/wsl/lib:${LIBRARY_PATH:-}"
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15
export NVCC_CCBIN=/usr/bin/g++-15 MAX_JOBS=4 OMP_NUM_THREADS=16
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LOG_LEVEL=INFO
unset MALLOC_ARENA_MAX MALLOC_ARENA_TEST MALLOC_MMAP_THRESHOLD_ MALLOC_TRIM_THRESHOLD_ MALLOC_TOP_PAD_ MALLOC_MMAP_MAX_
export GLIBC_TUNABLES=glibc.malloc.arena_max=2:glibc.malloc.mmap_threshold=1048576:glibc.malloc.trim_threshold=4194304
export PYTHONPATH="$overlay"
export QWEN36_DATA_ROOT="$root/optimization"
unset FREETOKEN_API_LOG_DIR
cd "$QWEN36_DATA_ROOT"
if [ "$2" = nsys ]; then
  exec "$root/optimization/tools/target-linux-x64/nsys" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --kill=sigterm \
    --cuda-graph-trace=node --wait=primary --export=sqlite --force-overwrite=false --output="$3" \
    "$root/.venv/bin/python" "$QWEN36_DATA_ROOT/experiment.py" "$1"
fi
exec "$root/.venv/bin/python" "$QWEN36_DATA_ROOT/experiment.py" "$1"
