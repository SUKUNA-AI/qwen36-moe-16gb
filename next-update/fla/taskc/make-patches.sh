#!/bin/bash
# Task C: .shared::cluster -> .shared::cta for the 1-D bulk global->shared copies (CRLF preserved).
set -euo pipefail
W=/path/to/qwen36-workspace/work
T=$W/next-update/fla/taskc
cd "$W"
rm -rf "$T/b"
FILES=(cuda-attention/src/qfa_prefill*.cu cuda-decode-prefetch/src/dp_common.cuh pcie-transfer/pcie_bench.cu)
for f in "${FILES[@]}"; do
  mkdir -p "$T/b/$(dirname "$f")"
  sed 's/cp\.async\.bulk\.shared::cluster\.global\.mbarrier::complete_tx::bytes/cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes/' "$f" > "$T/b/$f"
  cmp -s "$f" "$T/b/$f" && { echo "UNCHANGED $f"; exit 1; }
done
mk() { local out=$1; shift; : > "$T/$out"; for f in "$@"; do diff -u --label "a/$f" --label "b/$f" "$f" "$T/b/$f" >> "$T/$out" || true; done; }
mk qfa-prefill-shared-cta.patch cuda-attention/src/qfa_prefill*.cu
mk dp_common-shared-cta.patch cuda-decode-prefetch/src/dp_common.cuh
mk pcie_bench-shared-cta.patch pcie-transfer/pcie_bench.cu
wc -l "$T"/*.patch
git apply --check -p1 "$T/qfa-prefill-shared-cta.patch" "$T/dp_common-shared-cta.patch" "$T/pcie_bench-shared-cta.patch" && echo APPLY_CHECK_OK
# remaining non-comment shared::cluster bulk copies in the patched copies
grep -n "shared::cluster" -r "$T/b" | grep -v "^\s*//" || echo "no shared::cluster left in patched copies"
sha256sum "$T"/*.patch
