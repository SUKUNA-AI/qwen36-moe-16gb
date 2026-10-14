#!/bin/bash
H=/path/to/qwen36-workspace/work/next-update/short-route
cd $H
: > shortroute.patch
for f in engine/layer_major.py moe/offload_cache.py scheduler/layer_major.py; do
  diff -u --label a/python/freetoken/$f --label b/python/freetoken/$f orig/$f patch/$f >> shortroute.patch
done
grep -c $'\r' shortroute.patch
