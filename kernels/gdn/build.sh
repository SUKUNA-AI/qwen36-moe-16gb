#!/usr/bin/env bash
# Build libqwen36_gdn.so (ABI 2, sm_120a) + cubin/PTX/SASS dumps. CPU only (no GPU needed).
#   CUDA_HOME=<CUDA 13.4 toolkit>   (default /usr/local/cuda)
#   CUDA_CCBIN=<host compiler>      (default g++-15; the released build used GCC 15.3)
#   BUILD_OUT=<dir>                 output directory (default ./build)
#   NO_PTX=1                        skip the PTX dump
#   EXTRA_NVCC="-DQWEN_FLA_NEGCTRL=1|2|3"  negative-control builds (must FAIL fla/test_fla_fused.py)
set -euo pipefail
SRC=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
CCBIN=${CUDA_CCBIN:-g++-15}
OUT=${BUILD_OUT:-$SRC/build}
# Low CPU priority (A/B timing runs on the same host are sensitive to CPU contention); serial nvcc.
NVCC="nice -n 19 $CUDA_HOME/bin/nvcc"
CUOBJDUMP=${CUOBJDUMP:-$CUDA_HOME/bin/cuobjdump}
mkdir -p "$OUT"; chmod 700 "$OUT"
# qwen36_gdn.cu includes "qwen36_gdn_fla.cuh" from its own directory: flatten both into OUT.
tr -d '\r' < "$SRC/qwen36_gdn.cu" > "$OUT/qwen36_gdn.cu"
tr -d '\r' < "$SRC/fla/qwen36_gdn_fla.cuh" > "$OUT/qwen36_gdn_fla.cuh"
cd "$OUT"
# No fast-math: precise expf/log1pf/div.rn like the PyTorch / sgl_kernel builds (no FTZ).
FLAGS=(-std=c++17 -O3 -gencode arch=compute_120a,code=sm_120a -Xptxas -v
       --ftz=false --prec-div=true --prec-sqrt=true --fmad=true -lineinfo
       -ccbin "$CCBIN" -Xcompiler -fPIC,-O2 ${EXTRA_NVCC:-})
$NVCC "${FLAGS[@]}" -shared -o libqwen36_gdn.so qwen36_gdn.cu 2> build.log || { cat build.log; exit 1; }
$NVCC "${FLAGS[@]}" -cubin -o qwen36_gdn.cubin qwen36_gdn.cu 2>> build.log
if [ "${NO_PTX:-0}" != 1 ]; then
  $NVCC -std=c++17 -O3 -gencode arch=compute_120a,code=sm_120a --ftz=false --prec-div=true \
    --prec-sqrt=true -ccbin "$CCBIN" -ptx -o qwen36_gdn.ptx qwen36_gdn.cu
fi
nice -n 19 "$CUOBJDUMP" -sass qwen36_gdn.cubin > qwen36_gdn.sass
grep -E "registers|spill|smem|Compiling entry" build.log | sed 's/^ptxas info *: //' > resource-usage.txt || true
sha256sum libqwen36_gdn.so qwen36_gdn.cu qwen36_gdn_fla.cuh qwen36_gdn.cubin > SHA256SUMS
chmod 600 "$OUT"/*
echo "built $(sha256sum libqwen36_gdn.so | cut -c1-16)"
head -60 resource-usage.txt
