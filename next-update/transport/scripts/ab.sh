#!/usr/bin/env bash
# Rotating A/B: each (round, group) is one GPU-lock session (gpu.sh), so other sessions can take
# the GPU between groups. Odd rounds reverse the arm order. Usage: ab.sh TAG ROUNDS GROUP...
# GROUP is a comma list of arms, e.g. t1,c0p,c0e,c0pe
set -uo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu=${S4_GPU_SH:?path of the lock wrapper}
tag=$1 rounds=$2; shift 2
for ((r = 0; r < rounds; r++)); do
  for group in "$@"; do
    arms=$group
    if ((r % 2)); then arms=$(echo "$group" | tr ',' '\n' | tac | paste -sd,); fi
    echo "=== round $r arms $arms $(date -Is)"
    bash "$gpu" 3000 bash "$here/session.sh" --arms "$arms" --round "$r" --tag "$tag" --reps 4 --warmup 20 ${S4_AB_EXTRA:-}
  done
done
