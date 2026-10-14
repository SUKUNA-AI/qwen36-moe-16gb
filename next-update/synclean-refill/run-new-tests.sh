#!/bin/bash
D=/path/to/qwen36-workspace/work/next-update/synclean-refill
HS=/path/to/qwen36-workspace/work/host-sync
FT=${FT:-$D/tree/python}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
FT_ROOT=$FT timeout -k 10 1500 bash $HS/run-tests.sh "${@:-$D/tests}" 2>&1 | grep -v "INFO .*layer-major prefill uid" | tail -${TAILN:-60}
case "$FT" in "$D"/*) find "$FT" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null;; esac  # never clean a foreign/native tree
