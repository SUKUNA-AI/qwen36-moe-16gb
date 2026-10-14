#!/bin/bash
# Task C verification, compile only (no GPU): before/after cubins, look for the driver syscall.
set -uo pipefail
W=/path/to/qwen36-workspace/work
T=$W/next-update/fla/taskc
CUDA=/home/USER/qwen36-api/cuda-13.4
NVCC="nice -n 19 $CUDA/bin/nvcc"
CUOBJ=$CUDA/bin/cuobjdump
[ -x "$CUOBJ" ] || CUOBJ=/home/USER/qwen36-api/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump
OUT=/tmp/fla-taskc; rm -rf "$OUT"; mkdir -p "$OUT/inc"; chmod 700 "$OUT"
ln -sfn /home/USER/qwen36-api/.venv/lib/python3.12/site-packages/nvidia/cu13/include/nvtx3 "$OUT/inc/nvtx3"
"$CUDA/bin/nvcc" --version | tail -2
build() {  # side name srcdir relpath flags...
  local side=$1 name=$2 dir=$3 rel=$4; shift 4
  mkdir -p "$OUT/$side"
  $NVCC -ccbin /usr/bin/g++-15 "$@" -I"$OUT/inc" -cubin -o "$OUT/$side/$name.cubin" "$dir/$rel" > "$OUT/$side/$name.log" 2>&1 || { echo "BUILD_FAIL $side $name"; tail -5 "$OUT/$side/$name.log"; return; }
  "$CUOBJ" -elf "$OUT/$side/$name.cubin" > "$OUT/$side/$name.elf" 2>&1
  "$CUOBJ" -sass "$OUT/$side/$name.cubin" > "$OUT/$side/$name.sass" 2>&1
  local sys=$(grep -c "__cuda_syscall" "$OUT/$side/$name.elf"); local call=$(grep -c "CALL.ABS" "$OUT/$side/$name.sass")
  local ublk=$(grep -c "UBLKCP" "$OUT/$side/$name.sass")
  local stack=$(grep -o "[0-9]* bytes stack frame" "$OUT/$side/$name.log" | sort -u | tr '\n' ' ')
  printf "%-7s %-28s syscall_refs=%-3s CALL.ABS=%-3s UBLKCP=%-3s %s\n" "$side" "$name" "$sys" "$call" "$ublk" "$stack"
}
QFA=(-gencode=arch=compute_120a,code=sm_120a -O3 -use_fast_math -std=c++17 -lineinfo -Xptxas -v -DQFA_STAGES=5)
for side in before after; do
  if [ $side = before ]; then D=$W; else D=$T/b; fi
  for f in $W/cuda-attention/src/qfa_prefill*.cu; do
    n=$(basename "$f" .cu)
    build $side "$n" "$D" "cuda-attention/src/$n.cu" "${QFA[@]}" &
  done
  wait
  # dp_common.cuh is only used through decode_prefetch_bench.cu (dp_kernels.cuh includes it)
  mkdir -p "$OUT/dp-$side"; cp $W/cuda-decode-prefetch/src/decode_prefetch_bench.cu $W/cuda-decode-prefetch/src/dp_kernels.cuh "$OUT/dp-$side/"
  cp "$D/cuda-decode-prefetch/src/dp_common.cuh" "$OUT/dp-$side/"
  build $side decode_prefetch_bench "$OUT/dp-$side" decode_prefetch_bench.cu -std=c++17 -O3 -lineinfo --gpu-architecture=sm_120 -Xptxas=-v &
  build $side pcie_bench "$D" pcie-transfer/pcie_bench.cu -std=c++17 -O3 -lineinfo -arch=sm_120a -Xptxas=-v &
  wait
done
echo "artifacts: $OUT"
