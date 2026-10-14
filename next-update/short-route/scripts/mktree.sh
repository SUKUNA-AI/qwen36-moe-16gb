#!/bin/bash
set -e
H=/path/to/qwen36-workspace/work/next-update/short-route
W=/path/to/qwen36-workspace/work/FreeToken-opt
rm -rf $H/tree; mkdir -p $H/tree/python
cp -r $W/python/freetoken $H/tree/python/
find $H/tree -name __pycache__ -type d -prune -exec rm -rf {} +
cd $H/tree
patch -p1 --dry-run < $H/shortroute.patch
patch -p1 < $H/shortroute.patch
for f in engine/layer_major.py moe/offload_cache.py scheduler/layer_major.py; do cmp python/freetoken/$f $H/patch/$f && echo "same $f"; done
find . -name '*.orig' -o -name '*.rej'
sha256sum $H/shortroute.patch
