#!/usr/bin/env bash
# CPU-only build of the production FP8 GEMV library (ABI v2) + SASS evidence. No GPU use.
#   src/qwen_fp8_gemv.cu  ->  build/libqwen_fp8_gemv.so
# Flags are the prior tested build's flags plus -DQWEN_FP8_GEMV_PRODUCTION (analysis-only entry
# points compiled out) and -DQWEN_FP8_GEMV_SOURCE_SHA256 (ABI probe value).
set -euo pipefail
umask 022
W=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CUDA=${CUDA_HOME:-/usr/local/cuda}
CCBIN=${CUDA_CCBIN:-g++-15}
TB=${CUDA_BIN_TOOLS:-$CUDA/bin}   # cuobjdump
# Optional: SASS dump of an earlier tested build to compare the row-wise kernels against
# (the release build was compared with the v1 analysis build: identical row-wise SASS).
PRIOR_SASS=${PRIOR_SASS:-}
SRC="$W/src/qwen_fp8_gemv.cu"
OUT="$W/build"
mkdir -p "$OUT"
# nvcc output is not byte-reproducible (per-compilation module id), and cuda_fp8_gemv.py pins the
# sha256 of THIS artifact: never overwrite it by accident.
if [ -e "$OUT/libqwen_fp8_gemv.so" ] && [ "${FORCE_REBUILD:-0}" != 1 ]; then
  echo "build/libqwen_fp8_gemv.so exists and is pinned by sha256 in cuda_fp8_gemv.py; set FORCE_REBUILD=1" \
       "to rebuild (then update LIBRARY_SHA256 and re-run every test)" >&2
  exit 3
fi
if grep -q $'\r' "$SRC"; then echo "source has CR line endings; refusing" >&2; exit 1; fi
SRC_SHA=$(sha256sum "$SRC" | cut -c1-64)
FLAGS=(-std=c++17 -O3 -arch=sm_120a -ccbin "$CCBIN" -Xcompiler=-fPIC -Xptxas=-v -lineinfo --cudart=static
       -DQWEN_FP8_GEMV_PRODUCTION "-DQWEN_FP8_GEMV_SOURCE_SHA256=\"$SRC_SHA\"")
echo "nvcc: $("$CUDA/bin/nvcc" --version | tail -1)" > "$OUT/build-info.txt"
echo "flags: ${FLAGS[*]} --shared" >> "$OUT/build-info.txt"
echo "source_sha256: $SRC_SHA" >> "$OUT/build-info.txt"
"$CUDA/bin/nvcc" "${FLAGS[@]}" --shared "$SRC" -o "$OUT/libqwen_fp8_gemv.so" 2> "$OUT/build.log" \
  || { cat "$OUT/build.log"; exit 1; }
"$TB/cuobjdump" -sass "$OUT/libqwen_fp8_gemv.so" > "$OUT/libqwen_fp8_gemv.sass"
echo "== spills/registers (rowwise kernels)"
grep -E "spill|registers" "$OUT/build.log" | sort | uniq -c
# Per-kernel SASS identity against the prior tested library (same arithmetic must give same SASS).
if [ -n "$PRIOR_SASS" ]; then
python3 - "$PRIOR_SASS" "$OUT/libqwen_fp8_gemv.sass" > "$OUT/sass-compare.txt" <<'EOF'
import re, sys
def functions(path):
    text = open(path).read()
    # cuobjdump: "Function : <mangled>" blocks; keep the sm_120a ELF only (the sm_120 stub is a trap)
    out, cur, arch = {}, None, None
    for line in text.splitlines():
        m = re.search(r"arch = (sm_\w+)", line)
        if m:
            arch = m.group(1)
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = (arch, m.group(1)); out[cur] = []
            continue
        if cur and re.match(r"\s*/\*[0-9a-f]{4}\*/", line):
            # instruction text (address, opcode, operands, scheduling annotation) without the
            # encoding comment; whitespace normalized (cuobjdump pads columns per dump)
            out[cur].append(" ".join(re.sub(r"/\*\s*0x[0-9a-f]+\s*\*/", "", line).split()))
    return out
old, new = functions(sys.argv[1]), functions(sys.argv[2])
rowwise = sorted(k for k in new if "gemv_rowwise_kernel" in k[1])
print("rowwise kernels in new build:", len(rowwise))
same = 0
for k in rowwise:
    if k not in old:
        print("NEW-ONLY", k); continue
    ok = old[k] == new[k]
    same += ok
    qmma = sum("QMMA" in s for s in new[k])
    print(("IDENTICAL " if ok else "DIFFERENT ") + k[0] + " " + k[1] + f" ({len(new[k])} instr, {qmma} QMMA)")
print("removed (analysis-only) functions:", sorted(f for (a, f) in old if (a, f) not in new))
print(f"SUMMARY identical={same}/{len(rowwise)}")
EOF
cat "$OUT/sass-compare.txt"
fi
echo "== exported symbols"
nm -D --defined-only "$OUT/libqwen_fp8_gemv.so" | awk '{print $3}' | tee "$OUT/exports.txt"
echo "== NEEDED"
readelf -d "$OUT/libqwen_fp8_gemv.so" | grep NEEDED
sha256sum "$SRC" "$OUT/libqwen_fp8_gemv.so" | sed "s#$W/##" | tee "$OUT/sha256.txt"
