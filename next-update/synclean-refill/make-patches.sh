#!/bin/bash
# Build the deliverable patches from this dir's tree copy vs the CURRENT work/FreeToken-opt (pristine).
set -euo pipefail
W=/path/to/qwen36-workspace
D=$W/work/next-update/synclean-refill
FT=$W/work/FreeToken-opt/python/freetoken
T=$D/tree/python/freetoken
FILES="attention/fi.py attention/linear.py engine/layer_major.py kernel/cuda_gdn.py kernel/fla/index.py kernel/fla/utils.py moe/offload_cache.py sync_lean.py"
out=$D/synclean-release.patch; : > "$out"
for f in $FILES; do
  a=$FT/$f; [ -f "$a" ] || a=/dev/null
  diff -uN --label "a/python/freetoken/$f" --label "b/python/freetoken/$f" "$a" "$T/$f" >> "$out" || true
done
# every other file of the copy must equal the pristine tree
diff -rq -x __pycache__ "$FT" "$T" | grep -v -E "$(echo $FILES | sed 's/ /|/g')" && { echo "UNEXPECTED extra differences"; exit 1; } || true
# allowlist names for the teacher runs (engine/qualification.py)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cp "$FT/engine/qualification.py" "$tmp/q.py"
python3 - "$tmp/q.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p, encoding='utf-8', newline='').read()
old = "'teacher-long256-combined'"
assert s.count(old) == 1
s = s.replace(old, old + ",'teacher-main-release-synclean','teacher-main-release-refill-restore'")
open(p, 'w', encoding='utf-8', newline='').write(s)
PY
diff -u --label a/python/freetoken/engine/qualification.py --label b/python/freetoken/engine/qualification.py "$FT/engine/qualification.py" "$tmp/q.py" > $D/allowlist.patch || true
# verify both apply cleanly to a fresh copy and reproduce the tree
c=$tmp/check; mkdir -p $c/python; cp -a $W/work/FreeToken-opt/python/freetoken $c/python/; find $c -name __pycache__ -type d -prune -exec rm -rf {} +
cd $c && git apply --check -p1 $D/synclean-release.patch && git apply -p1 $D/synclean-release.patch && git apply --check -p1 $D/allowlist.patch && git apply -p1 $D/allowlist.patch
for f in $FILES; do cmp -s $c/python/freetoken/$f $T/$f || { echo "MISMATCH $f"; exit 1; }; done
echo "patches OK: $(sha256sum $D/synclean-release.patch | cut -c1-12) synclean-release.patch, $(sha256sum $D/allowlist.patch | cut -c1-12) allowlist.patch"
for f in $FILES; do echo "after $f $(sha256sum $T/$f | cut -c1-12) (before $( [ -f $FT/$f ] && sha256sum $FT/$f | cut -c1-12 || echo new))"; done
echo "after engine/qualification.py $(sha256sum $c/python/freetoken/engine/qualification.py | cut -c1-12) (before $(sha256sum $FT/engine/qualification.py | cut -c1-12))"
/home/USER/qwen36-api/.venv/bin/python -I -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]; print('syntax OK')" $(for f in $FILES; do echo $T/$f; done) $c/python/freetoken/engine/qualification.py
