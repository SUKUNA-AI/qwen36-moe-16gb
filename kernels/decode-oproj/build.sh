#!/usr/bin/env bash
# CPU-only build of the decode o-projection library (ABI v1) + SASS evidence. No GPU use.
#   src/qwen_oproj_gemv.cu  ->  build/libqwen_oproj_gemv.so
# Refuses to overwrite a pinned build (FORCE_REBUILD=1), CRLF sources, and any SASS that would raise
# the context stack (CALL.ABS / __cuda_syscall) or uses .shared::cluster (ARCHITECTURE.md 5.3).
set -euo pipefail
umask 022
W=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CUDA=${CUDA_HOME:-/usr/local/cuda}
CCBIN=${CUDA_CCBIN:-g++-15}
TB=${CUDA_BIN_TOOLS:-$CUDA/bin}   # cuobjdump (e.g. the one shipped with Triton)
SRC="$W/src/qwen_oproj_gemv.cu"
OUT=${BUILD_OUT:-"$W/build"}
LIB="$OUT/libqwen_oproj_gemv.so"
mkdir -p "$OUT"
# nvcc output is not byte-reproducible (per-compilation module id); cuda_oproj_gemv.py pins the
# sha256 of THIS artifact: never overwrite it by accident.
if [ -e "$LIB" ] && [ "${FORCE_REBUILD:-0}" != 1 ]; then
  echo "$LIB exists and is pinned by sha256 in cuda_oproj_gemv.py; set FORCE_REBUILD=1 to rebuild" \
       "(then update LIBRARY_SHA256 and re-run every test)" >&2
  exit 3
fi
if grep -q $'\r' "$SRC"; then echo "source has CR line endings; refusing" >&2; exit 1; fi
SRC_SHA=$(sha256sum "$SRC" | cut -c1-64)
FLAGS=(-std=c++17 -O3 -gencode=arch=compute_120a,code=sm_120a -ccbin "$CCBIN" -Xcompiler=-fPIC -Xptxas=-v -lineinfo --cudart=static
       "-DQWEN_OPROJ_SOURCE_SHA256=\"$SRC_SHA\"")
{
  echo "nvcc: $("$CUDA/bin/nvcc" --version | tail -1)"
  echo "flags: ${FLAGS[*]} --shared"
  echo "source_sha256: $SRC_SHA"
} > "$OUT/build-info.txt"
"$CUDA/bin/nvcc" "${FLAGS[@]}" --shared "$SRC" -o "$LIB" 2> "$OUT/build.log" || { cat "$OUT/build.log"; exit 1; }
"$CUDA/bin/nvcc" "${FLAGS[@]}" -ptx "$SRC" -o "$OUT/qwen_oproj_gemv.ptx"
"$TB/cuobjdump" -sass "$LIB" > "$OUT/libqwen_oproj_gemv.sass"
echo "== spills/registers"
grep -E "spill|registers" "$OUT/build.log" | sort | uniq -c
fail=0
if grep -nE "CALL\.ABS|__cuda_syscall" "$OUT/libqwen_oproj_gemv.sass"; then echo "SASS has CALL.ABS/__cuda_syscall" >&2; fail=1; fi
if grep -n "shared::cluster" "$OUT/qwen_oproj_gemv.ptx"; then echo "PTX uses .shared::cluster" >&2; fail=1; fi
if grep -nE " [1-9][0-9]* bytes spill" "$OUT/build.log"; then echo "register spills" >&2; fail=1; fi
[ "$fail" = 0 ] || exit 2
echo "sass_checks: no CALL.ABS, no __cuda_syscall, no .shared::cluster, no spills" >> "$OUT/build-info.txt"
echo "== QMMA per kernel"
awk '/Function :/{f=$3} /QMMA/{n[f]++} END{for (k in n) print n[k], k}' "$OUT/libqwen_oproj_gemv.sass"
echo "== exported symbols"
nm -D --defined-only "$LIB" | awk '{print $3}' | tee "$OUT/exports.txt"
echo "== NEEDED"
readelf -d "$LIB" | grep NEEDED
sha256sum "$SRC" "$LIB" | sed "s#$W/##" | tee "$OUT/sha256.txt"
