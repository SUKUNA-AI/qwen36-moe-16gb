#!/usr/bin/env bash
# Usage: scan.sh <cccl_root> <arch e.g. 120> <out_dir> [test-glob]
set -u
ROOT=$1; ARCH=$2; OUT=$3; GLOB=${4:-*}
N=$HOME/qwen36-api/cuda-13.4/bin/nvcc
OBJ=$HOME/qwen36-api/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump
T=$ROOT/libcudacxx/test/libcudacxx/cuda/ptx
mkdir -p "$OUT"
for f in $T/ptx.$GLOB.compile.pass.cpp; do
  b=$(basename "$f" .compile.pass.cpp)
  s=$(date +%s%N)
  if ! $N -std=c++17 -x cu -arch=sm_$ARCH -cubin -O3 -w -I "$ROOT/libcudacxx/include" -I "$T" -I "$ROOT/libcudacxx/test/support" "$f" -o "$OUT/$b.cubin" 2> "$OUT/$b.err"; then
    echo "$b COMPILE_FAIL $(head -c 200 "$OUT/$b.err" | tr '\n' ' ')"; continue
  fi
  e=$(date +%s%N)
  $OBJ -sass "$OUT/$b.cubin" 2>/dev/null > "$OUT/$b.sass"
  $OBJ -elf "$OUT/$b.cubin" 2>/dev/null | grep -oE '__cuda_syscall[A-Za-z0-9_]*' | sort -u > "$OUT/$b.syscalls"
  nf=$(grep -c 'Function :' "$OUT/$b.sass")
  bad=$(awk '/Function :/{fn=$3} /CALL\.ABS/{print fn}' "$OUT/$b.sass" | sort -u | wc -l)
  printf '%s secs=%.1f functions=%s with_CALL.ABS=%s syscalls=%s\n' "$b" "$(awk -v a=$s -v b=$e "BEGIN{print (b-a)/1e9}")" "$nf" "$bad" "$(tr '\n' ',' < "$OUT/$b.syscalls")"
done
