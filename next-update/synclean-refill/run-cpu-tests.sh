#!/bin/bash
# CPU-only regression of the rebased sync-lean tree copy (never the workspace/native trees).
# Usage: run-cpu-tests.sh [tree python dir]  (default: this dir's tree/python)
D=/path/to/qwen36-workspace/work/next-update/synclean-refill
FT=${1:-$D/tree/python}
HS=/path/to/qwen36-workspace/work/host-sync
LM=/path/to/qwen36-workspace/work/layer-major
case "$FT" in "$D"/*) find "$FT" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null;; esac  # never clean a foreign/native tree
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "== host-sync suites (sync_lean, layer_major, refill) on $FT"
FT_ROOT=$FT timeout -k 10 1500 bash $HS/run-tests.sh tests/sync_lean tests/layer_major tests/refill 2>&1 | grep -v "INFO .*layer-major prefill uid" | grep -E "^FAIL|^ERROR|^skip|^== |Error:" | tail -40
echo "== layer-major suites (attnbatch, layer_major, refill) on $FT"
cd $LM && FT_ROOT=$FT timeout -k 10 1500 nice -n 19 ionice -c3 bash $LM/run-tests.sh tests/attnbatch tests/layer_major tests/refill 2>&1 | grep -v "INFO .*layer-major prefill uid" | grep -E "^FAIL|^ERROR|^skip|^== |Error:" | tail -40
if [ -d $D/tests ]; then
  echo "== new grouped-lean tests"
  cd $HS && FT_ROOT=$FT timeout -k 10 1500 bash $HS/run-tests.sh $D/tests 2>&1 | grep -v "INFO .*layer-major prefill uid" | tail -40
fi
case "$FT" in "$D"/*) find "$FT" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null;; esac  # never clean a foreign/native tree
