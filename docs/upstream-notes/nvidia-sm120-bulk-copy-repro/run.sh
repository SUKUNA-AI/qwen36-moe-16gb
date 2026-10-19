#!/usr/bin/env bash
# Build and run both variants of bulk_stack_repro.cu on an sm_120 GPU.
# Usage: CUDA=/usr/local/cuda bash run.sh [out_dir]
set -euo pipefail
CUDA=${CUDA:-${CUDA_HOME:-/usr/local/cuda}}
CUOBJDUMP=${CUOBJDUMP:-$CUDA/bin/cuobjdump}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$(mktemp -d)}
mkdir -p "$OUT"
"$CUDA/bin/nvcc" --version | tail -1
for v in 1 0; do
  name=$([ "$v" = 1 ] && echo cluster || echo cta)
  "$CUDA/bin/nvcc" -O3 -arch=sm_120 -DDST_CLUSTER=$v -Xptxas -v "$HERE/bulk_stack_repro.cu" -o "$OUT/repro_$name" 2> "$OUT/ptxas_$name.txt"
  "$CUOBJDUMP" -sass "$OUT/repro_$name" > "$OUT/sass_$name.txt"
  "$CUOBJDUMP" -elf "$OUT/repro_$name" > "$OUT/elf_$name.txt"
  echo "== $name"
  grep -E "stack frame|registers" "$OUT/ptxas_$name.txt" | sed 's/^ *//'
  echo "SASS: CALL.ABS=$(grep -c 'CALL.ABS' "$OUT/sass_$name.txt" || true) UBLKCP=$(grep -c 'UBLKCP' "$OUT/sass_$name.txt" || true) syscall_refs=$(grep -c '__cuda_syscall' "$OUT/elf_$name.txt" || true)"
  "$OUT/repro_$name"
done
