#!/bin/bash
# Control tree: P1 + P2 applied with fuzz only (no grouped-path rebase), to show what the new tests detect.
W=/path/to/qwen36-workspace
D=$W/work/next-update/synclean-refill
T=$D/tree-fuzz
rm -rf "$T"; mkdir -p "$T/python"
cp -a $W/work/FreeToken-opt/python/freetoken "$T/python/"
find "$T" -name __pycache__ -type d -prune -exec rm -rf {} +
cd "$T" && patch -s -p1 < $W/work/host-sync/patch/sync-lean.patch && patch -s -p1 < $W/work/host-sync/patch/sync-lean-fi-plans.patch && echo "fuzz tree ready"
