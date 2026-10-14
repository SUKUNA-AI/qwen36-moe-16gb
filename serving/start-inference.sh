#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/USER/qwen36-api
test -f /path/to/LLM/qwen36-api/model/download-receipt.json || { echo 'Verified checkpoint download receipt is missing'; exit 75; }
export CUDA_HOME="$ROOT/cuda-13.4"
export PATH="$CUDA_HOME/bin:$ROOT/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:/usr/lib/wsl/lib:${LIBRARY_PATH:-}"
export OMP_NUM_THREADS=16
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15
export NVCC_CCBIN=/usr/bin/g++-15 MAX_JOBS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LOG_LEVEL=INFO
# glibc malloc tunables (release-prep): bound host-process arena retention seen in the 2 h soak
# (stepwise anon growth on fresh 65K/120K requests). Exported before the interpreter starts: ld.so
# reads GLIBC_TUNABLES at exec and every child process inherits it. Legacy MALLOC_* aliases from an
# inherited environment are removed so the value below is the only allocator configuration.
unset MALLOC_ARENA_MAX MALLOC_ARENA_TEST MALLOC_MMAP_THRESHOLD_ MALLOC_TRIM_THRESHOLD_ MALLOC_TOP_PAD_ MALLOC_MMAP_MAX_
export GLIBC_TUNABLES=glibc.malloc.arena_max=2:glibc.malloc.mmap_threshold=1048576:glibc.malloc.trim_threshold=4194304
unset FREETOKEN_API_LOG_DIR
cd "$ROOT"
export PYTHONPATH="$ROOT/release/python"
exec "$ROOT/.venv/bin/python" "$ROOT/profile-runtime.py"
