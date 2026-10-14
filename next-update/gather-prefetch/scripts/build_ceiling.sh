#!/bin/bash
set -euo pipefail
D=/path/to/qwen36-workspace/work/next-update/gather-prefetch
CUDA=/home/USER/qwen36-api/cuda-13.4
mkdir -p $D/build
cd $D/build
$CUDA/bin/nvcc -ccbin /usr/bin/g++-15 -std=c++20 -O3 -arch=sm_120a -Xptxas=-v -lineinfo $D/src/ceiling_bench.cu -o ceiling_bench.new 2> build.log || { grep -i -E "error|warning" build.log | head -40; exit 1; }
grep -c "Compiling entry" build.log
grep -i -E "spill|stack" build.log | sort | uniq -c | sort -rn | head -5
grep -i "syscall\|CALL" build.log | head -3 || true
mv -f ceiling_bench.new ceiling_bench
sha256sum $D/src/ceiling_bench.cu ceiling_bench > ceiling.sha256
echo BUILD_OK
