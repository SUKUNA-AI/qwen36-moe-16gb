#!/usr/bin/env bash
set -euo pipefail
root=/home/USER/qwen36-api/optimization
cuda=/home/USER/qwen36-api/cuda-13.4
src="$root/python/freetoken/kernel/csrc/qwen36_complement.cu"
mkdir -p "$root/build"
"$cuda/bin/nvcc" -ccbin /usr/bin/g++-15 -std=c++17 -O3 -shared -Xcompiler=-fPIC \
  --gpu-architecture=sm_120 --cudart=shared "$src" -o "$root/build/libqwen36_complement.so"
sha256sum "$src" "$root/build/libqwen36_complement.so" > "$root/build/complement.sha256"
chmod 700 "$root/build"
chmod 600 "$root/build/"*
