#!/bin/bash
# Dry-run the sync-lean patches on a COPY of work/FreeToken-opt (never touches the original).
set -uo pipefail
W=/path/to/qwen36-workspace
D=$W/work/next-update/synclean-refill
H=$W/work/host-sync/patch
N=/home/USER/qwen36-api/optimization
T=$D/tree
rm -rf "$T"; mkdir -p "$T/python"
cp -a $W/work/FreeToken-opt/python/freetoken "$T/python/"
find "$T" -name __pycache__ -type d -prune -exec rm -rf {} +
echo "== workspace vs native for touched files"
for f in attention/fi.py attention/linear.py kernel/fla/index.py kernel/fla/utils.py moe/offload_cache.py engine/layer_major.py scheduler/layer_major.py moe/layer_major.py moe/slot_refill.py engine/engine.py models/qwen3_5_moe/model.py; do
  a=$(sha256sum "$T/python/freetoken/$f" | cut -c1-12); b=$(sha256sum "$N/python/freetoken/$f" 2>/dev/null | cut -c1-12)
  echo "$f ws=$a native=$b $([ "$a" = "$b" ] && echo SAME || echo DIFF)"
done
cd "$T"
for P in sync-lean.patch sync-lean-fi-plans.patch; do
  echo "== $P git apply --check"; git apply --check -p1 "$H/$P" 2>&1 | head -20
  echo "== $P patch --dry-run"; patch -p1 --dry-run < "$H/$P" 2>&1 | head -30
  echo "== apply for real to the copy (to allow P2 on top)"; patch -p1 --no-backup-if-mismatch -s < "$H/$P" 2>&1 | head; 
done
find "$T" -name "*.rej" -o -name "*.orig" | head
