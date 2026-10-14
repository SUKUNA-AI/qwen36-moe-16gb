#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/USER/qwen36-api
export CUDA_HOME="$ROOT/cuda-13.4"
export NVCC_CCBIN=/usr/bin/g++-15
mkdir -p "$ROOT/work"
cat > "$ROOT/work/toolchain-probe.cu" <<'CU'
#include <cuda_runtime.h>
__global__ void probe(int* value) { if (threadIdx.x == 0) *value = 1; }
CU
"$CUDA_HOME/bin/nvcc" -ccbin /usr/bin/g++-15 -arch=sm_120 -std=c++17 -Xcompiler=-fPIC -c "$ROOT/work/toolchain-probe.cu" -o "$ROOT/work/toolchain-probe.o"
/usr/bin/g++-15 -shared "$ROOT/work/toolchain-probe.o" -L"$CUDA_HOME/lib" -L"$CUDA_HOME/lib64" -lcudart -o "$ROOT/work/toolchain-probe.so"
echo TOOLCHAIN_COMPILE_LINK_PASS
