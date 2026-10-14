#!/usr/bin/env bash
# Build the fused full-attention output gate (sigmoid -> BF16 multiply -> static FP8 quant).
# CPU only. Output: build/libqwen_attention_gate.so (the archive build was named
# libqwen_gate_fp8.so and installed under the released name libqwen_attention_gate.so).
#   CUDA_HOME=<CUDA 13.4 toolkit>  (default /usr/local/cuda)
#   CUDA_CCBIN=<host compiler>     (default g++-15; the released build used GCC 15.3)
# Note: target is -arch=sm_120 (not sm_120a), no --use_fast_math; keep the source file name,
# it is part of the anonymous-namespace symbol names.
set -euo pipefail
umask 077
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CUDA=${CUDA_HOME:-/usr/local/cuda}
CCBIN=${CUDA_CCBIN:-g++-15}
mkdir -p "$HERE/build"
"$CUDA/bin/nvcc" -std=c++17 -O3 --shared -arch=sm_120 -ccbin "$CCBIN" \
  -Xcompiler=-fPIC -Xptxas=-v -lineinfo --cudart=shared \
  -Xlinker -rpath -Xlinker "$CUDA/lib64" \
  "$HERE/attention_gate_fp8.cu" -o "$HERE/build/libqwen_attention_gate.so" \
  2> "$HERE/build/build.log"
( cd "$HERE" && sha256sum attention_gate_fp8.cu build/libqwen_attention_gate.so ) > "$HERE/build/sha256.txt"
cat "$HERE/build/sha256.txt"
