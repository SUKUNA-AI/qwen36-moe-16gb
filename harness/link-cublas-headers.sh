#!/usr/bin/env bash
set -euo pipefail
root=/home/USER/qwen36-api/optimization
source=/home/USER/qwen36-api/.venv/lib/python3.12/site-packages/nvidia/cu13/include
mkdir -p "$root/include"
for header in "$source"/cublas*.h; do
  target="$root/include/$(basename "$header")"
  if [ -e "$target" ]; then test "$(readlink "$target")" = "$header"; else ln -s "$header" "$target"; fi
done
