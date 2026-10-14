#!/usr/bin/env bash
# Build the bit-exact NVFP4 MoE prefill library (v4, fp32 globals) + cubin/PTX/SASS dumps.
# CPU only; does not touch the GPU.
#   CUDA_HOME=<CUDA 13.4 toolkit>  (default /usr/local/cuda)
#   CUDA_CCBIN=<host compiler>     (default g++-15; the released build used GCC 15.3)
#   ./build.sh [OUT_DIR]           (default ./build)
set -euo pipefail
umask 022
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CUDA=${CUDA_HOME:-/usr/local/cuda}
CCBIN=${CUDA_CCBIN:-g++-15}
NVDISASM=${NVDISASM:-$CUDA/bin/nvdisasm}
SRC="$HERE/src/qwen36_nvfp4_bitexact.cu"
OUT="${1:-$HERE/build}"
mkdir -p "$OUT"
FLAGS=(-std=c++17 -O3 --shared -Xcompiler=-fPIC -gencode=arch=compute_120a,code=sm_120a
       -ccbin "$CCBIN" -lineinfo -Xptxas=-v --cudart=shared
       -Xlinker -rpath -Xlinker "$CUDA/lib64")
printf '%s\n' "${FLAGS[@]}" > "$OUT/build-flags.txt"
"$CUDA/bin/nvcc" "${FLAGS[@]}" "$SRC" -o "$OUT/libqwen_nvfp4_bitexact.so" 2> "$OUT/build.log" || { cat "$OUT/build.log"; exit 1; }
"$CUDA/bin/nvcc" -std=c++17 -O3 -gencode=arch=compute_120a,code=sm_120a -ccbin "$CCBIN" -lineinfo \
   -cubin -o "$OUT/qwen36_nvfp4_bitexact.cubin" "$SRC"
"$CUDA/bin/nvcc" -std=c++17 -O3 -gencode=arch=compute_120a,code=compute_120a -ccbin "$CCBIN" \
   -ptx -o "$OUT/qwen36_nvfp4_bitexact.ptx" "$SRC"
"$NVDISASM" -c -g "$OUT/qwen36_nvfp4_bitexact.cubin" > "$OUT/qwen36_nvfp4_bitexact.sass"
"$CUDA/bin/nvcc" --version | tail -2 > "$OUT/nvcc-version.txt"
"$CCBIN" --version | head -1 >> "$OUT/nvcc-version.txt"
( cd "$HERE" && sha256sum "src/qwen36_nvfp4_bitexact.cu" ) > "$OUT/build-sha256.txt"
( cd "$OUT" && sha256sum libqwen_nvfp4_bitexact.so qwen36_nvfp4_bitexact.cubin ) >> "$OUT/build-sha256.txt"
awk '/Compiling entry function/{match($0,/_Z[A-Za-z0-9_]+/); f=substr($0,RSTART,RLENGTH)} /spill stores/{s=$0} /Used [0-9]+ registers/{match($0,/Used [0-9]+ registers/); print f, substr($0,RSTART,RLENGTH), s}' "$OUT/build.log" | sed -E "s/ +/ /g" > "$OUT/regs.txt" || true
cut -c1-260 "$OUT/regs.txt"
cat "$OUT/build-sha256.txt"
