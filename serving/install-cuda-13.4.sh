#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/USER/qwen36-api
mkdir -p "$ROOT/cuda-13.4" /path/to/LLM/qwen36-api/downloads
while read -r relative sha; do
  archive="/path/to/LLM/qwen36-api/downloads/${relative##*/}"
  if [ ! -f "$archive" ]; then
    curl --fail --location --retry 3 --output "$archive" "https://developer.download.nvidia.com/compute/cuda/redist/$relative"
  fi
  printf '%s  %s\n' "$sha" "$archive" | sha256sum -c -
  tar -xf "$archive" --strip-components=1 -C "$ROOT/cuda-13.4"
done <<'COMPONENTS'
cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-13.4.59-archive.tar.xz 0c08d1df80b5d0bd081778d392446ab50a6a54b047c0136b39c8dbfdf2cfed3f
cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.4.49-archive.tar.xz ec3df9f91560db23a064c6b395d4fde17b2893f5bf954c6bd193dd03a44de522
cccl/linux-x86_64/cccl-linux-x86_64-13.3.4.2.1-archive.tar.xz 7c973e70d8f8d732f801fe2297c9cb5a541f22824fa6ad64ac4f0e183f9c4397
cuda_crt/linux-x86_64/cuda_crt-linux-x86_64-13.4.59-archive.tar.xz 0a62d47788ffc1ec9d85c248d61a0941b8f0c47c6eeba9a26e27c01955094d3c
libnvvm/linux-x86_64/libnvvm-linux-x86_64-13.4.59-archive.tar.xz 8966b24692ef664b77a3d3411a0c82b4960a2d6a179e0281dac0752c5c1518c7
libcurand/linux-x86_64/libcurand-linux-x86_64-10.4.4.49-archive.tar.xz f08512d67cf25de22026638127a1635280e594b03a92f0592d33acfacad4c7fc
COMPONENTS
test -e "$ROOT/cuda-13.4/lib64" || ln -s lib "$ROOT/cuda-13.4/lib64"
"$ROOT/cuda-13.4/bin/nvcc" --version
