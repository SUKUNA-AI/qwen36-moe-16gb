#!/usr/bin/env bash
# Optional pre-steps for final-gates.sh (DRAFT; CPU only, no GPU, no units). Default is --check (reports only);
# --apply performs the writes the parent decided on. Writes (only with --apply):
#   P1  release-tooling patches 01-08 -> outputs/ (patch -p1 from the workspace root, verified against after_sha256;
#       09/profiles.json is NOT applied: final-install.sh writes profiles.json with the attention-batch key + flags)
#   P2  teacher allowlist 'teacher-main-release','teacher-long256-release' -> work/FreeToken-opt engine/qualification.py
#   P3  (--gather in) tuned decode-gather module as a manifest-bound venv prebuilt: copy the private JIT build whose
#       directory equals the CURRENT recipe digest (shared::cta fix) into the venv kernel cache and merge the
#       manifest_entry.py record into work/FreeToken-opt/python/freetoken/kernel/kernel_cache_manifest.json
# Must run AFTER r3-driver21 finished and BEFORE final-gates.sh (all three change frozen/synced bytes).
# Usage: final-prep.sh --gather in|out [--apply]
set -uo pipefail
W=/path/to/qwen36-workspace
FT=$W/work/FreeToken-opt
PY=/home/USER/qwen36-api/.venv/bin/python
RT=$W/work/agents/release-tooling
H=$W/work/agents/final-gates
CACHE=/home/USER/qwen36-api/.venv/lib/python3.12/site-packages/freetoken_kernel_cache/jit_cache
MOD=freetoken__decode_gather_tma_8_3_16384
PRIV=/home/USER/.cache/freetoken/provenance-jit/$MOD
QF=$FT/python/freetoken/engine/qualification.py
MANIFEST=$FT/python/freetoken/kernel/kernel_cache_manifest.json
SCR=${FINAL_PREP_SCRATCH:-/tmp}   # scratch for the provenance inventory output
GATHER= APPLY=0
while [ $# -gt 0 ]; do
  case "$1" in --gather) GATHER=${2:-}; shift 2;; --apply) APPLY=1; shift;; *) echo "unknown argument $1" >&2; exit 2;; esac
done
case "$GATHER" in in|out) ;; *) echo '--gather in|out required' >&2; exit 2;; esac
note() { echo "$(date -u +%FT%TZ) [prep] $*"; }
fail=0
systemctl --user is-active --quiet qwen36-experiment && { note "STOP: qwen36-experiment active (r3 or an agent still running)"; exit 1; }
systemctl --user is-active --quiet qwen36-inference && { note "STOP: qwen36-inference active"; exit 1; }

# P1 release-tooling patches 01-08
while read -r path before after patch; do
  now=$(sha256sum < "$W/$path" | cut -c1-64)
  if [ "$now" = "$after" ]; then note "P1 $path: applied"; continue; fi
  [ "$now" = "$before" ] || { note "P1 $path: UNKNOWN state ${now:0:12} (neither before ${before:0:12} nor after ${after:0:12})"; fail=1; continue; }
  if [ "$APPLY" = 0 ]; then note "P1 $path: pending ($patch)"; continue; fi
  ( cd "$W" && patch -p1 --dry-run < "$W/$patch" > /dev/null && patch -p1 < "$W/$patch" > /dev/null ) || { note "P1 $path: patch failed"; fail=1; continue; }
  [ "$(sha256sum < "$W/$path" | cut -c1-64)" = "$after" ] && note "P1 $path: patched (after ${after:0:12})" || { note "P1 $path: wrong bytes after patch"; fail=1; }
done < <("$PY" -I -c 'import json,sys
for f in json.load(open(sys.argv[1]))["files"]:
    if not f["path"].endswith("profiles.json"): print(f["path"], f["before_sha256"], f["after_sha256"], f["patch"])' "$RT/result.json")

# P2 allowlist (CRLF/LF preserving, same idiom as r2/r3 drivers)
"$PY" -I - "$QF" "$APPLY" <<'PY' || fail=1
import sys
p, apply = sys.argv[1], sys.argv[2] == '1'
s = open(p, encoding='utf-8', newline='').read()
need = [n for n in ('teacher-main-release', 'teacher-long256-release') if "'%s'" % n not in s]
if not need:
    print('P2 allowlist: both names present'); sys.exit(0)
anchor = "'teacher-main-combined',"
assert s.count(anchor) == 1, 'allowlist anchor not unique'
if not apply:
    print('P2 allowlist: pending ' + ','.join(need)); sys.exit(0)
open(p, 'w', encoding='utf-8', newline='').write(s.replace(anchor, anchor + ''.join("'%s'," % n for n in need)))
print('P2 allowlist: added ' + ','.join(need))
PY

# P3 decode-gather prebuilt (only when the parent decided gather is in)
if [ "$GATHER" = in ]; then
  ENVX=(env CUDA_HOME=/home/USER/qwen36-api/cuda-13.4 PATH=/home/USER/qwen36-api/cuda-13.4/bin:$PATH CC=/usr/bin/gcc-15
        CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15 NVCC_CCBIN=/usr/bin/g++-15 CUDA_VISIBLE_DEVICES= TVM_FFI_CUDA_ARCH_LIST=12.0
        PYTHONPATH=$FT/python)
  # Already done by the parent (any build path)? A venv prebuilt bound by the current manifest is all that matters.
  if [ -f "$CACHE/$MOD/$MOD.so" ] && grep -q "\"$MOD\"" "$MANIFEST"; then
    CUDA_VISIBLE_DEVICES= "$PY" -I "$FT/python/freetoken/kernel/kernel_provenance.py" inventory --kernel-dir "$FT/python/freetoken/kernel" \
      --cache-dir "$CACHE" > $SCR/final-prep-provenance.json 2>&1
    if "$PY" -I "$H/final-helpers.py" provenance $SCR/final-prep-provenance.json --require-module "$MOD" --modules-only; then
      note "P3 already done: $MOD is a manifest-bound venv prebuilt for the current recipe digest"
      note "prep done apply=$APPLY fail=$fail; next: bash $H/final-gates.sh --gather $GATHER --dry-run, then without --dry-run"
      exit "$fail"
    fi
    note "P3 venv prebuilt/manifest present but NOT prebuilt for the current sources (stale digest?): see $SCR/final-prep-provenance.json"
  fi
  any=$(ls -1 "$PRIV"/*/"$MOD.so" 2>/dev/null | head -1)
  [ -n "$any" ] || any=$(ls -1 "$CACHE/$MOD/$MOD.so" 2>/dev/null | head -1)
  if [ -z "$any" ] && [ "$APPLY" = 1 ]; then  # CPU-only private build from the workspace tree (r2/r3 prebuild idiom)
    ( cd /tmp && "${ENVX[@]}" timeout 600 "$PY" -c 'from freetoken.kernel.decode_gather import _jit_decode_gather_module as m, TUNED_DECODE_GATHER as t; m(t); print("gather JIT built")' )
    any=$(ls -1 "$PRIV"/*/"$MOD.so" 2>/dev/null | head -1)
  fi
  if [ -z "$any" ]; then note "P3 no gather build found (private or cache)"; fail=1
  else
    entry=$(cd /tmp && "${ENVX[@]}" "$PY" "$W/work/wf-gather/scripts/manifest_entry.py" --so "$any" 2>/dev/null)
    digest=$("$PY" -I -c 'import json,sys; d=json.loads(sys.argv[1]); print(d[sys.argv[2]]["recipe_digest"])' "$entry" "$MOD" 2>/dev/null)
    src=$PRIV/${digest:0:24}/$MOD.so dst=$CACHE/$MOD/$MOD.so
    note "P3 current recipe digest ${digest:0:24}; private build $( [ -f "$src" ] && echo present || echo MISSING )"
    if [ -z "$digest" ] || [ ! -f "$src" ]; then note "P3 no private build for the current digest (rebuild needed)"; fail=1
    elif [ -f "$dst" ] && ! cmp -s "$src" "$dst"; then note "P3 venv cache holds a DIFFERENT $MOD.so: refusing to overwrite"; fail=1
    elif [ "$APPLY" = 0 ]; then note "P3 pending: copy $src -> $dst and merge the manifest entry"
    else
      mkdir -p "$(dirname "$dst")" && { [ -f "$dst" ] || cp -p "$src" "$dst"; } && chmod 644 "$dst"
      entry=$(cd /tmp && "${ENVX[@]}" "$PY" "$W/work/wf-gather/scripts/manifest_entry.py" --so "$dst")
      "$PY" -I - "$MANIFEST" "$MOD" "$entry" <<'PY' || fail=1
import json, sys
path, mod, entry = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])[sys.argv[2]]
d = json.load(open(path))
old = d['modules'].get(mod)
if old is not None:
    assert old == entry, 'manifest already binds a different ' + mod
    print('P3 manifest: entry already present'); sys.exit(0)
d['modules'][mod] = entry
open(path, 'w', newline='\n').write(json.dumps(d, indent=1, sort_keys=True) + '\n')
print('P3 manifest: entry merged (so_sha256 %s)' % entry['so_sha256'][:16])
PY
    fi
  fi
  if [ "$APPLY" = 1 ]; then
    # inventory (not check): the workspace tree has no native-only extension .so; modules are what P3 changes
    CUDA_VISIBLE_DEVICES= "$PY" -I "$FT/python/freetoken/kernel/kernel_provenance.py" inventory --kernel-dir "$FT/python/freetoken/kernel" \
      --cache-dir "$CACHE" > $SCR/final-prep-provenance.json 2>&1
    "$PY" -I "$H/final-helpers.py" provenance $SCR/final-prep-provenance.json --require-module "$MOD" --modules-only || fail=1
  fi
else
  note "P3 skipped (--gather out): decode_gather stays off; no venv/manifest change"
fi
note "prep done apply=$APPLY fail=$fail; next: bash $H/final-gates.sh --gather $GATHER --dry-run, then without --dry-run"
exit "$fail"
