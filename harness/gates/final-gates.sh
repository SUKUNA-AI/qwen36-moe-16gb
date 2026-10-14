#!/usr/bin/env bash
# Final release gates A-E (DRAFT for the parent; nothing here touches production units, Caddy or the release tree).
# Final candidate main-release = r2-attn (layer-major prefill + CUDA GDN ABI-2 a8a806ed + FP8 GEMV 685c78cf + bit-exact
# CUDA NVFP4 prefill + QWEN36_LAYER_MAJOR_ATTN_BATCH=2) [+ tuned decode gather + TVM_FFI_CUDA_ARCH_LIST=12.0 if
# --gather in] + FREETOKEN_DISABLE_JIT=1; long256-release = the same at 262144 / .73. FLA, refill, sync-lean excluded.
# Order (C runs AFTER D: the long-context gates' identity check and the soak's --expected-identity are bound to the D
# matrix manifest, which only exists once the matrix has started):
#   A  CPU   preconditions, configs, run-config->profile mapping, freeze check (documented exceptions only), sync 'final',
#            native CPU suites + 4 release-tooling suites, kernel provenance (+ gather prebuilt), tunables contract [HARD]
#   B  GPU   qualify.py main-release, long256-release: teacher 0-diff + functional + PPL                            [HARD]
#   D  GPU   tool pilot (bf128) -> matrix bf128,main-release (5 pairs, output 128) -> compare-ab                     [gate]
#   C  GPU   C0 main128 context gates 131072/120000, L4 long256 262144/250000, F1 baseline64-final 65536/60000      [gate]
#   E  GPU   soak pilot 30 s (all case types) -> 7200 s soak on main-release (GLIBC_TUNABLES pre/post) -> analysis   [gate]
#   Z  CPU   summary + decision draft (reports/final-gates-RUN-{summary,decision-draft}.json)
# Every GPU step runs under `flock -w 3600 -E 75 LOCK` with retry on 75/73 (r3-driver21 idiom). Tools that take the
# lock themselves (matrix, tool-pilot, compare-ab) run through parent-lockwrap.py (parent-screen.py idea).
# Usage: final-gates.sh --gather in|out [--run-id f1] [--sync-label final] [--dry-run] [--skip-c0] [--soak-anyway]
#   --sync-label   only if the workspace legitimately changed after a 'final' sync (new label = new sync manifest;
#                  combine with a new --run-id so every receipt binds to the new bytes)
#   --dry-run      CPU only, read-only: preconditions, derived configs (temp dir), mapping, freeze check, sync --dry-run,
#                  plan. Writes nothing in outputs/, work/FreeToken-opt/ or native.
#   re-runs        steps already PASS for this run id are skipped (reports/final-gates-RUN.steps, append-only)
# Exit: 0 main128 releasable (all its gates passed), 1 otherwise / stopped, 2 usage.
set -uo pipefail
W=/path/to/qwen36-workspace
O=$W/outputs/qwen36-api/optimization
FT=$W/work/FreeToken-opt
N=/home/USER/qwen36-api/optimization
PY=/home/USER/qwen36-api/.venv/bin/python
LOCK=/home/USER/qwen36-api/run/gpu-exclusive.lock
H=$W/work/agents/final-gates
HELP=$H/final-helpers.py
WRAP=$H/parent-lockwrap.py
RT=$W/work/agents/release-tooling
L256=$W/work/agents/long256-r1b
TUN=$W/work/agents/release-prep/patches/new-check-glibc-tunables.py
CACHE=/home/USER/qwen36-api/.venv/lib/python3.12/site-packages/freetoken_kernel_cache/jit_cache
DRAFT=$RT/staging/profiles-draft.json
DRAFT_SHA=dce3a145ce03082f045d472a2f7032d4ccc9e818fbc3768bd890cddb0940e1de
GATHER_MOD=freetoken__decode_gather_tma_8_3_16384
KL_SHA=93e766db34450f9ae414d63d58e09ee80ffdd5f898620742657a588c909bf100
TUNABLES='glibc.malloc.arena_max=2:glibc.malloc.mmap_threshold=1048576:glibc.malloc.trim_threshold=4194304'
LOG=$N/reports/final-gates.log
EXTRA=(--extra-config main-release.json --extra-config long256-release.json)
# make-freeze --check MISMATCH lines allowed: "path workspace12" (and workspace == sync-freeze). Release-tooling
# freeze_exceptions_to_document (patches 01-03 applied) + the exceptions already documented in r3-driver21.sh.
ALLOW_FREEZE='outputs/qwen36-api/profile-runtime.py 0f30ee09c751
outputs/qwen36-api/optimization/release.py 6d8328c3d48e
outputs/qwen36-api/optimization/make-qualification.py 6398c923d3a7
outputs/qwen36-api/optimization/run-experiment.sh a06b0b29a178
outputs/qwen36-api/optimization/bench.py 204d28710ef0
outputs/qwen36-api/optimization/screen-chunk.py 798a2728f941'
# The four ctypes libraries the run configs load (= the bytes release.py stages; r2 run receipts carry the same sha256).
declare -A LIBSHA=(
  [$N/private-cuda-gdn/pinned/libqwen36_gdn-a8a806eda3a90a65.so]=a8a806eda3a90a65c45b3ab02d8c7beaacb507aa9c1f380a0f255fa99ba067a4
  [$N/private-fp8gemv/libqwen_fp8_gemv-685c78cf89e60d02.so]=685c78cf89e60d0205f0db3ac2040203dec27e510a1e543cb7a7d3097414093b
  [$N/private-libraries/libqwen_nvfp4_bitexact.so]=b7fea22750256c2a34d45ce512bae62027165fa211bb2581003af96055225e4b
  [$N/private-libraries/libqwen_attention_gate.so]=f654f87a0a3f621fa51324fb00482ef51084d26432b1ac12b7710c9a5877f55e)
libsha_ok() { [ "$(sha256sum < "$1" | cut -c1-64)" = "$2" ]; }
# 4 release-tooling suites (staged copies applied to outputs by patches 05-08) and their minimum test counts.
SUITES='test-profile-runtime-cpu.py:88 test-release-qualification-cpu.py:56 test-release-cpu.py:13 test-export-release-evidence-cpu.py:13'

RUN=f1 GATHER= MODE=run SKIP_C0=0 SOAK_ANYWAY=0 SL=final
while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) RUN=${2:-}; shift 2;;
    --sync-label) SL=${2:-}; shift 2;;
    --gather) GATHER=${2:-}; shift 2;;
    --dry-run) MODE=dry-run; shift;;
    --skip-c0) SKIP_C0=1; shift;;
    --soak-anyway) SOAK_ANYWAY=1; shift;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
[[ "$RUN" =~ ^[a-z0-9]{1,8}$ ]] || { echo 'bad --run-id (want [a-z0-9]{1,8})' >&2; exit 2; }
[[ "$SL" =~ ^final[a-z0-9-]{0,20}$ ]] || { echo 'bad --sync-label (want final[a-z0-9-]*, default final)' >&2; exit 2; }
case "$GATHER" in in|out) ;; *) echo '--gather in|out is required (parent decision on the tuned decode gather)' >&2; exit 2;; esac
T=final-$RUN
STEPS=$N/reports/final-gates-$RUN.steps
MAN=$N/reports/$T-m-manifest.json
FREF=$N/reports/functional-main-release.json
FAILS=0

note() { local line; line="$(date -u +%FT%TZ) [$RUN] $*"; echo "$line"; [ "$MODE" = run ] && echo "$line" >> "$LOG"; return 0; }
logged() { if [ "$MODE" = run ]; then "$@" >> "$LOG" 2>&1; else "$@"; fi; }
record() {  # record STEP STATUS DETAIL  (append-only; the last line of a step wins)
  note "VERDICT $1 $2 ${3:-}"
  [ "$MODE" = run ] && echo "$1 $2 ${3:-}" >> "$STEPS"
  return 0
}
status_of() { [ -f "$STEPS" ] && awk -v s="$1" '$1==s{v=$2} END{print v}' "$STEPS"; }
passed() { [ "$(status_of "$1")" = PASS ]; }
detail_of() {  # detail_of STEP KEY -> value of KEY=... in the last line of STEP
  [ -f "$STEPS" ] || return 0
  awk -v s="$1" -v k="$2=" '$1==s{line=$0} END{n=split(line,a," "); for(i=3;i<=n;i++) if(index(a[i],k)==1){print substr(a[i],length(k)+1)}}' "$STEPS"
}
attempt() { local n=0; [ -f "$STEPS" ] && n=$(awk -v s="$1" '$1==s{c++} END{print c+0}' "$STEPS"); echo $((n+1)); }
check() {  # check LABEL cmd...  (dry-run: report and continue; run: hard stop)
  local label=$1; shift
  if "$@"; then note "ok: $label"; return 0; fi
  FAILS=$((FAILS+1))
  if [ "$MODE" = run ]; then stop "precondition: $label"; else note "DRY-RUN FAIL: $label"; fi
  return 1
}
retry() {  # r3-driver21 idiom: retry only while the lock wait timed out (75) or a model unit was active (73)
  local label=$1; shift; local t=0 rc
  while :; do "$@"; rc=$?; { [ "$rc" -ne 75 ] && [ "$rc" -ne 73 ]; } && break; t=$((t+1)); [ "$t" -ge 1200 ] && break; sleep 3; done
  note "$label rc=$rc"; return "$rc"
}
lockrun() { flock -w 3600 -E 75 "$LOCK" "$@"; }
jget() { "$PY" -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" "$1" "$2" 2>/dev/null || echo MISSING; }
summary() {
  [ "$MODE" = run ] || return 0
  "$PY" -I "$HELP" summary --steps "$STEPS" --run "$RUN" --gather "$GATHER" --reports "$N/reports" \
    --out "$N/reports/final-gates-$RUN-summary.json" --decision-out "$N/reports/final-gates-$RUN-decision-draft.json" \
    --release-prep-draft "$W/work/agents/release-prep/decision-draft.json" 2>&1 | tee -a "$LOG"
}
stop() { note "STOP: $*"; [ "$MODE" = run ] && echo "STOP FAIL $(echo "$*" | tr ' ' '_')" >> "$STEPS"; summary; exit 1; }
units_inactive() { ! systemctl --user is-active --quiet qwen36-inference && ! systemctl --user is-active --quiet qwen36-experiment; }
trap 'note "interrupted (signal)"; exit 130' INT TERM

cd "$N" || exit 1
note "final-gates start mode=$MODE gather=$GATHER sync_label=$SL skip_c0=$SKIP_C0 soak_anyway=$SOAK_ANYWAY"
if [ "$MODE" = run ]; then
  prev=$(awk '$1=="CONFIG"{v=$2} END{print v}' "$STEPS" 2>/dev/null)
  [ -n "$prev" ] && [ "$prev" != "gather=$GATHER" ] && { note "STOP: run $RUN was started with $prev; use a new --run-id"; exit 2; }
  [ -z "$prev" ] && echo "CONFIG gather=$GATHER" >> "$STEPS"
fi

# ============================================================================================ A (CPU, hard)
if [ "$MODE" = run ] && passed A; then note "A already PASS (skip)"; else
A0=$SECONDS
check "units qwen36-inference/qwen36-experiment inactive" units_inactive
check "release-tooling patches 01-08 applied (after_sha256)" "$PY" -I - "$RT/result.json" "$W" <<'PYEOF'
import hashlib, json, pathlib, sys
d = json.load(open(sys.argv[1])); W = pathlib.Path(sys.argv[2]); bad = []
for f in d['files']:
    if f['path'].endswith('profiles.json'):
        continue  # patch 09 is replaced by final-install.sh (profiles.json with attention batch + qualified flags)
    now = hashlib.sha256((W / f['path']).read_bytes()).hexdigest()
    if now != f['after_sha256']:
        bad.append(f"{f['path']}: {now[:12]} (before {f['before_sha256'][:12]}, want {f['after_sha256'][:12]})")
print('\n'.join(bad) if bad else 'release-tooling patches 01-08: all 8 files at after_sha256')
sys.exit(3 if bad else 0)
PYEOF
for name in teacher-main-release teacher-long256-release; do
  check "allowlist $name in workspace engine/qualification.py" grep -q "'$name'" "$FT/python/freetoken/engine/qualification.py"
done
for lib in "${!LIBSHA[@]}"; do
  check "library $(basename "$lib") sha256 ${LIBSHA[$lib]:0:16}" libsha_ok "$lib" "${LIBSHA[$lib]}"
done
check "profiles draft sha256 ${DRAFT_SHA:0:12}" libsha_ok "$DRAFT" "$DRAFT_SHA"
for f in "$L256/run-long256-gates-r1b.sh" "$L256/source-identity-check-r1b.py" "$L256/evaluate-gates-r1b.py" \
         "$L256/budget-calc-r1b.py" "$W/work/agents/long256-plan/gpu-mem-sampler.sh" "$TUN" "$HELP" "$WRAP" \
         "$O/baseline64-final.json" "$O/bf128.json" "$W/work/agents/harness-v6/make-freeze.py"; do
  check "present $(basename "$f")" test -f "$f"
done
if [ "$GATHER" = in ]; then
  check "decode_gather.cuh has no .shared::cluster bulk copy" bash -c \
    '! grep -v -E "^[[:space:]]*//" "$1" | grep -q "shared::cluster"' _ "$FT/python/freetoken/kernel/csrc/jit/decode_gather.cuh"
  check "kernel_cache_manifest.json binds $GATHER_MOD" grep -q "\"$GATHER_MOD\"" "$FT/python/freetoken/kernel/kernel_cache_manifest.json"
  check "venv prebuilt $GATHER_MOD.so present" test -f "$CACHE/$GATHER_MOD/$GATHER_MOD.so"
fi
# A1 configs (create O_EXCL or verify identical)
if [ "$MODE" = run ]; then CFGDIR=$O; check "configs main-release/long256-release" logged "$PY" -I "$HELP" configs --gather "$GATHER" --opt "$O" --write
else
  CFGDIR=$(mktemp -d "${FINAL_GATES_SCRATCH:-/tmp}/final-gates-dry.XXXXXX"); note "dry-run scratch (kept for review): $CFGDIR"
  cp "$O/r2-attn.json" "$CFGDIR/"; [ -f "$O/main-candidate-r3.json" ] && cp "$O/main-candidate-r3.json" "$CFGDIR/"
  for c in main-release long256-release; do [ -f "$O/$c.json" ] && cp "$O/$c.json" "$CFGDIR/"; done
  check "configs (derived into $CFGDIR)" "$PY" -I "$HELP" configs --gather "$GATHER" --opt "$CFGDIR" --write
fi
# A2 run config -> production profile mapping (staged validator + staged make-qualification, CPU)
VAL=$W/outputs/qwen36-api/profile-runtime.py; MQ=$O/make-qualification.py
if [ "$MODE" = dry-run ] && [ "$(sha256sum < "$VAL" | cut -c1-12)" != 0f30ee09c751 ]; then
  VAL=$RT/staging/qwen36-api/profile-runtime.py; MQ=$RT/staging/qwen36-api/optimization/make-qualification.py
  note "dry-run: patches not applied yet; mapping uses the release-tooling staging copies"
fi
check "run configs map exactly onto the release profiles" logged "$PY" -I "$HELP" mapping --gather "$GATHER" --validator "$VAL" \
  --mq "$MQ" --draft "$DRAFT" --main "$CFGDIR/main-release.json" --long "$CFGDIR/long256-release.json" --b64 "$O/baseline64-final.json"
# A3 freeze + check (documented exceptions only), A4 sync 'final'
if [ "$MODE" = run ] && [ -f "$N/reports/sync-v5-$SL.json" ]; then
  note "sync '$SL' already applied (reports/sync-v5-$SL.json): verify only"
else
  if [ "$MODE" = run ]; then FREEZE=$O/sync-$SL-freeze.sha256; FCHK=$N/reports/final-freeze-check-$RUN.txt
  else FREEZE=$CFGDIR/sync-$SL-freeze.sha256; FCHK=$CFGDIR/freeze-check.txt; fi
  FX=("${EXTRA[@]}")
  if [ "$MODE" = dry-run ] && { [ ! -f "$O/main-release.json" ] || [ ! -f "$O/long256-release.json" ]; }; then
    FX=(); note "dry-run: main-release/long256-release.json not in outputs/ yet: freeze printed WITHOUT --extra-config (real run adds 2 config lines)"
  fi
  bash "$O/sync-v5.sh" --print-freeze "${FX[@]}" > "$FREEZE"; prc=$?
  check "sync-v5 --print-freeze rc=$prc" test "$prc" = 0
  python3 "$W/work/agents/harness-v6/make-freeze.py" --check "$FREEZE" > "$FCHK" 2>&1
  grep -E '^(MISMATCH|MUTABLE|RUNTIME|check:)' "$FCHK" | while read -r l; do note "freeze: $l"; done
  UNEXPECTED=$(ALLOW_FREEZE="$ALLOW_FREEZE" awk 'BEGIN{n=split(ENVIRON["ALLOW_FREEZE"],a,"\n"); for(i=1;i<=n;i++) ok[a[i]]=1}
    $1=="MISMATCH"{p=$2; sub(/:$/,"",p); if(!((p" "$6) in ok) || $6!=$8) print}' "$FCHK")
  check "freeze: no MISMATCH beyond the 6 documented exceptions${UNEXPECTED:+ (unexpected: $UNEXPECTED)}" test -z "$UNEXPECTED"
  check "freeze: runtime tree present in sync freeze" grep -q '^RUNTIME ' "$FCHK"
  if [ "$MODE" = run ]; then
    retry "sync dry-run" lockrun bash "$O/sync-v5.sh" --dry-run --label "$SL" --freeze "$FREEZE" "${EXTRA[@]}" >> "$LOG" 2>&1 || stop "sync dry-run"
    retry "sync apply" lockrun bash "$O/sync-v5.sh" --apply --label "$SL" --freeze "$FREEZE" "${EXTRA[@]}" >> "$LOG" 2>&1 || stop "sync apply"
  else
    bash "$O/sync-v5.sh" --dry-run --label "$SL" --freeze "$FREEZE" "${FX[@]}" > "$CFGDIR/sync-dry-run.txt" 2>&1; src=$?
    note "dry-run: sync-v5 --dry-run rc=$src (75 = GPU lock busy now; 65/66 = freeze/source problem) -> $CFGDIR/sync-dry-run.txt"
    tail -5 "$CFGDIR/sync-dry-run.txt"
  fi
fi
if [ "$MODE" = run ]; then
  bash "$O/sync-v5.sh" --verify --label "$SL" >> "$LOG" 2>&1 || stop "sync verify (drift since sync '$SL')"
  for c in main-release long256-release baseline64-final bf128; do
    check "native config/$c.json == workspace" cmp -s "$O/$c.json" "$N/config/$c.json"
  done
  for name in teacher-main-release teacher-long256-release; do
    check "allowlist $name in native engine/qualification.py" grep -q "'$name'" "$N/python/freetoken/engine/qualification.py"
  done
  for lib in "${!LIBSHA[@]}"; do check "post-sync library $(basename "$lib") sha256" libsha_ok "$lib" "${LIBSHA[$lib]}"; done
  # A5 native CPU suites + the 4 release-tooling suites (run-tests.sh idiom: from /tmp, -I -B, nice/ionice)
  ( cd "$N" && CUDA_VISIBLE_DEVICES= "$PY" test-paired-contract-cpu.py ) > "$N/reports/$T-cpu-paired-contract.log" 2>&1; rc=$?
  # Pre-existing layout failures (identical at R1/R1b: 57 run, failures=1, errors=1): two tests look for run configs in
  # $N/ instead of $N/config/. Accepted only when the failing set is exactly these two; anything else stops.
  pc_bad=$(grep -E '^(FAIL|ERROR): test_' "$N/reports/$T-cpu-paired-contract.log" | sed -E 's/ \(.*//' | sort | tr '\n' ' ')
  pc_known='ERROR: test_screen_configs_differ_only_in_chunk_and_ratio FAIL: test_no_blanket_copy_and_qualify_flow_compatible '
  [ "$rc" != 0 ] && [ "$pc_bad" = "$pc_known" ] && note "test-paired-contract-cpu: only the 2 known pre-existing config-layout failures"
  check "test-paired-contract-cpu rc=$rc ($(tail -1 "$N/reports/$T-cpu-paired-contract.log" | cut -c1-80))" test "$rc" = 0 -o "$pc_bad" = "$pc_known"
  ( cd "$N" && CUDA_VISIBLE_DEVICES= "$PY" test-soak-cpu.py ) > "$N/reports/$T-cpu-soak.log" 2>&1; rc=$?
  check "test-soak-cpu rc=$rc ($(tail -1 "$N/reports/$T-cpu-soak.log" | cut -c1-80))" test "$rc" = 0
  QWEN36_GUARD_UNDER_TEST="$N/guard.py" bash "$O/CPU-checks.sh" > "$N/reports/cpu-checks-$T.log" 2>&1; rc=$?
  check "CPU-checks.sh (guard v2 suites) rc=$rc" test "$rc" = 0
  for s in $SUITES; do
    suite=${s%%:*}; min=${s##*:}; out=$N/reports/$T-${suite%.py}.log
    ( cd /tmp && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 nice -n 19 ionice -c3 "$PY" -I -B "$O/$suite" -v ) > "$out" 2>&1; rc=$?
    ran=$(grep -oE '^Ran [0-9]+ tests' "$out" | grep -oE '[0-9]+' | tail -1)
    check "$suite rc=$rc ran=${ran:-0} (>= $min)" test "$rc" = 0 -a "${ran:-0}" -ge "$min"
  done
  # A6 kernel provenance of the synced tree + venv cache (release requires every module prebuilt)
  CUDA_VISIBLE_DEVICES= "$PY" -I "$N/python/freetoken/kernel/kernel_provenance.py" check --kernel-dir "$N/python/freetoken/kernel" \
    --cache-dir "$CACHE" --require-extension _pinned_tensor > "$N/reports/kernel-provenance-$T.json" 2>&1; rc=$?
  check "kernel_provenance check rc=$rc" test "$rc" = 0
  if [ "$GATHER" = in ]; then REQ=(--require-module "$GATHER_MOD"); else REQ=(); fi
  check "kernel provenance: all prebuilt${REQ:+ incl. $GATHER_MOD}" logged "$PY" -I "$HELP" provenance "$N/reports/kernel-provenance-$T.json" "${REQ[@]}"
  # A7 allocator tunables contract of the synced launcher + checker self-test
  check "native run-experiment.sh exports GLIBC_TUNABLES" grep -qxF "export GLIBC_TUNABLES=$TUNABLES" "$N/run-experiment.sh"
  check "check-glibc-tunables --selftest" logged "$PY" -I "$TUN" --selftest
  record A PASS "t=$((SECONDS-A0))s sync=$SL gather=$GATHER"
else
  note "dry-run: A5 suites / A6 provenance / A7 tunables run only after the sync (real mode)"
  CUDA_VISIBLE_DEVICES= "$PY" -I "$N/python/freetoken/kernel/kernel_provenance.py" check --kernel-dir "$N/python/freetoken/kernel" \
    --cache-dir "$CACHE" --require-extension _pinned_tensor > "$CFGDIR/provenance-now.json" 2>&1
  if [ "$GATHER" = in ]; then "$PY" -I "$HELP" provenance "$CFGDIR/provenance-now.json" --require-module "$GATHER_MOD"
  else "$PY" -I "$HELP" provenance "$CFGDIR/provenance-now.json"; fi
  note "dry-run: provenance of the CURRENT native tree rc=$? (informational before the final sync)"
fi
fi

if [ "$MODE" = dry-run ]; then
  cat <<EOF
plan (real mode, run id $RUN, gather $GATHER; GPU steps under flock -w 3600 -E 75 $LOCK):
  B  $PY $N/qualify.py --candidate main-release        -> qcheck quality-main-release.json, extract $T-qrun-*
     $PY $N/qualify.py --candidate long256-release     -> qcheck quality-long256-release.json
  D  parent-lockwrap.py tool-pilot --profile bf128 --repeats 3 --name $T-tool-a1
     parent-lockwrap.py matrix --profiles bf128,main-release --pairs 5 --name $T-m --output 128 --tool-pilot $T-tool-a1
     parent-lockwrap.py compare-ab --name $T-m --candidate main-release --baseline bf128 --output 128
  C  run-long256-gates-r1b.sh --profile main-release --context 131072 --length 120000 --expected-long-prompt 120067 \\
       --expected-boundary-prompt 130396 --identity-manifest $MAN --identity-arm main-release --functional-reference $FREF --tag $T-c128-a1
     run-long256-gates-r1b.sh --profile long256-release --context 262144 --length 250000 --identity-manifest $MAN \\
       --identity-arm main-release --functional-reference $FREF --tag $T-l256-a1
     run-long256-gates-r1b.sh --profile baseline64-final --context 65536 --length 60000 --identity-manifest $MAN --identity-arm bf128 \\
       --functional-reference $FREF --known-limitation-case long-facts-60000 --known-limitation-sha $KL_SHA --tag $T-b64-a1
  E  start main-release; check-glibc-tunables; soak.py --name soak-pilot-$T-a1 --seconds 30 ...; soak-analysis (pilot_view)
     start main-release; check-glibc-tunables pre; soak.py --name soak-$T-a1 (7200 s); check-glibc-tunables post; soak-analysis
  Z  final-helpers.py summary -> reports/final-gates-$RUN-summary.json, reports/final-gates-$RUN-decision-draft.json
dry-run precondition failures: $FAILS
EOF
  exit $(( FAILS > 0 ))
fi

# ============================================================================================ B (GPU, hard)
qualify_one() {  # qualify_one CANDIDATE
  local c=$1 t0=$SECONDS rc q
  if passed "B.$c" || [ "$(status_of "B.$c")" = PASS_EQUIV_R1B256 ]; then note "B.$c already $(status_of "B.$c") (skip)"; return 0; fi
  units_inactive || note "a model unit is active: qualify waits on the lock / unit"
  retry "qualify $c" lockrun "$PY" "$N/qualify.py" --candidate "$c" >> "$LOG" 2>&1; rc=$?
  "$PY" -I "$HELP" qcheck "$N/reports/quality-$c.json" 2>&1 | tee -a "$LOG"; q=${PIPESTATUS[0]}
  journalctl --user -u qwen36-experiment --since "-40 min" --no-pager -o cat 2>/dev/null \
    | grep -i -E 'groups of|cuda.gdn|fp8|gather|DISABLE_JIT|jit|Traceback' | tail -12 | cut -c1-240 | tee -a "$LOG"
  "$PY" "$N/check-candidate-metrics.py" --candidate "$c" >> "$LOG" 2>&1; note "check-candidate-metrics $c rc=$? (informational)"
  local st=PASS
  # R1b equivalence was a historical f1 decision (docs/METHODOLOGY.md section 1.2).
  # New runs require the complete run-bound byte-exact gate; no equivalence override.
  if [ "$rc" != 0 ] || [ "$q" != 0 ]; then record "B.$c" FAIL "qualify_rc=$rc qcheck_rc=$q t=$((SECONDS-t0))s"; stop "teacher/quality gate failed for $c"; fi
  "$PY" -I "$HELP" extract-run --report "$N/reports/teacher-$c.json" --profile "teacher-$c" --config "$O/$c.json" --teacher \
    --out "$N/reports/$T-qrun-teacher-$c.json" >> "$LOG" 2>&1 || { record "B.$c" FAIL "teacher run receipt"; stop "teacher run receipt of $c does not match its config"; }
  "$PY" -I "$HELP" extract-run --report "$N/reports/functional-$c.json" --profile "$c" --config "$O/$c.json" \
    --out "$N/reports/$T-qrun-functional-$c.json" >> "$LOG" 2>&1 || { record "B.$c" FAIL "functional run receipt"; stop "functional run receipt of $c does not match its config"; }
  record "B.$c" "$st" "t=$((SECONDS-t0))s quality=quality-$c.json"
}
qualify_one main-release
qualify_one long256-release

# ============================================================================================ D (GPU)
t0=$SECONDS
if passed D.pilot; then PILOT=$(detail_of D.pilot name); note "D.pilot already PASS ($PILOT)"; else
  PILOT=$T-tool-a$(attempt D.pilot)
  retry "tool-pilot $PILOT" lockrun "$PY" -I "$WRAP" tool-pilot --profile bf128 --repeats 3 --name "$PILOT" >> "$LOG" 2>&1; rc=$?
  st=$(jget "$N/reports/$PILOT-tool-pilot.json" "d.get('status')"); rst=$(jget "$N/reports/$PILOT-tool-pilot-runner.json" "d.get('status')")
  if [ "$rc" = 0 ] && [ "$st" = PASS ] && [ "$rst" = PASS ]; then record D.pilot PASS "name=$PILOT t=$((SECONDS-t0))s"
  else record D.pilot FAIL "name=$PILOT rc=$rc status=$st runner=$rst"; stop "tool pilot failed: no matrix, so no identity reference for C/E"; fi
fi
t0=$SECONDS
if passed D.matrix || [ "$(status_of D.matrix)" = INVALID ]; then note "D.matrix already recorded ($(status_of D.matrix))"; else
  for try in 1 2; do
    retry "matrix $T-m (try $try)" lockrun "$PY" -I "$WRAP" matrix --profiles bf128,main-release --pairs 5 --name "$T-m" --output 128 \
      --tool-pilot "$PILOT" >> "$N/reports/$T-m.console.log" 2>&1; rc=$?
    [ "$rc" = 1 ] || break
    fstep=$(jget "$N/reports/$T-m-matrix.json" "[r for r in d if r.get('status')=='failed'][-1].get('failed_step')")
    case "$fstep" in preflight|start|readiness) note "matrix arm failed at $fstep (resumable under the same manifest): one resume";; *) break;; esac
  done
  jget "$N/reports/$T-m-matrix.json" "[(r['profile'], r['repeat'], r.get('status'), r.get('arm_verdict'), r.get('failed_step'), round(((r.get('vram') or {}).get('min_gpu_free_bytes') or 0)/2**30, 3), round(((r.get('vram') or {}).get('foreign_gpu_used_bytes_pre_start') or 0)/2**20)) for r in d]" | tee -a "$LOG"
  case $rc in
    0) record D.matrix PASS "rc=0 name=$T-m t=$((SECONDS-t0))s";;
    3) record D.matrix INVALID "rc=3 arm(s)_below_VRAM_floor name=$T-m";;
    *) record D.matrix FAIL "rc=$rc name=$T-m";;
  esac
fi
if [ -f "$MAN" ]; then record D.manifest PASS "sha=$(sha256sum < "$MAN" | cut -c1-16)"
else stop "matrix manifest $MAN missing: C (identity) and E (--expected-identity) have no reference"; fi
CMP=$N/reports/$T-m-main-release-vs-bf128-comparison.json
if [ "$(status_of D.matrix)" = FAIL ]; then record D.compare NOT_RUN "matrix_incomplete"
elif [ -n "$(status_of D.compare)" ] && [ "$(status_of D.compare)" != ERROR ]; then note "D.compare already recorded ($(status_of D.compare))"
else
  rc=exists; [ -f "$CMP" ] || { retry "compare-ab main-release/bf128" lockrun "$PY" -I "$WRAP" compare-ab --name "$T-m" --candidate main-release \
    --baseline bf128 --output 128 >> "$LOG" 2>&1; rc=$?; }
  v=$(jget "$CMP" "d.get('verdict')"); ok=$(jget "$CMP" "d.get('contract_pass') is True and d.get('five_pairs') is True")
  jget "$CMP" "{k: d.get(k) for k in ('verdict','verdict_reasons','contract_pass','five_pairs','no_regression_over_3pct_full_request_975','whole_request_gain_at_least_5pct_bonferroni','performance_gate_pass','output_content','failures')}" | tee -a "$LOG"
  reasons=$(jget "$CMP" "','.join(d.get('verdict_reasons') or [])"); diffs=$(jget "$CMP" "'%s/%s' % (d['output_content']['differs'], d['output_content']['compared'])")
  case "$v:$ok" in
    PASS:True|INCONCLUSIVE:True) record D.compare "$v" "rc=$rc reasons=${reasons:-none} outputs_differ=$diffs";;
    FAIL:*|INVALID:*) record D.compare "$v" "rc=$rc reasons=${reasons:-none} contract=$ok";;
    *) record D.compare ERROR "rc=$rc verdict=$v contract=$ok";;
  esac
fi

# ============================================================================================ C (GPU)
gates() {  # gates STEP TAGBASE ARGS...   (run-long256-gates-r1b.sh exit: 0 PASS*, 1 FAIL, 2 usage, 73/75 busy, 76 predicted floor breach, 77 identity)
  local step=$1 base=$2 tag rc v cls mf t0=$SECONDS; shift 2
  if passed "$step" || [ "$(status_of "$step")" = PASS_WITH_KNOWN_LIMITATION ]; then note "$step already $(status_of "$step") (skip)"; return 0; fi
  tag=$base-a$(attempt "$step")
  retry "$step $tag" lockrun bash "$L256/run-long256-gates-r1b.sh" "$@" --identity-manifest "$MAN" --tag "$tag" >> "$LOG" 2>&1; rc=$?
  v=$(jget "$N/reports/$tag-evaluation.json" "d['verdict']"); cls=$(jget "$N/reports/$tag-evaluation.json" "d.get('vram_reserve_class')")
  mf=$(jget "$N/reports/$tag-evaluation.json" "d.get('min_gpu_free_GiB')")
  jget "$N/reports/$tag-evaluation.json" "{k: d.get(k) for k in ('verdict','failed_checks','not_run','warnings','vram_reserve_class','min_gpu_free_GiB')}" | tee -a "$LOG"
  case $rc in
    0) record "$step" "$v" "tag=$tag class=$cls min_free_GiB=$mf t=$((SECONDS-t0))s";;
    77) record "$step" FAIL "tag=$tag identity_mismatch"; stop "$step: identity differs from $MAN";;
    76) record "$step" NOT_RUN "tag=$tag predicted_VRAM_floor_breach(76)_nothing_started";;
    *) record "$step" FAIL "tag=$tag rc=$rc verdict=$v class=$cls";;
  esac
}
[ -f "$FREF" ] || stop "missing functional reference $FREF"
if [ "$SKIP_C0" = 1 ]; then record C.c0 SKIPPED_BY_PARENT "--skip-c0"
else gates C.c0 "$T-c128" --profile main-release --context 131072 --length 120000 --expected-long-prompt 120067 \
  --expected-boundary-prompt 130396 --identity-arm main-release --functional-reference "$FREF"; fi
gates C.l4 "$T-l256" --profile long256-release --context 262144 --length 250000 --identity-arm main-release --functional-reference "$FREF"
gates C.f1 "$T-b64" --profile baseline64-final --context 65536 --length 60000 --identity-arm bf128 --functional-reference "$FREF" \
  --known-limitation-case long-facts-60000 --known-limitation-sha "$KL_SHA"

# ============================================================================================ E (GPU)
soak_run() {  # soak_run NAME SECONDS_ARG  -> rc of the locked start+tunables+soak block
  local name=$1 secs=$2 post=
  [ -z "$secs" ] && post="; $PY -I $TUN --unit qwen36-experiment --output $N/reports/$name-tunables-post.json"
  retry "soak $name" lockrun bash -c "set -e; ! systemctl --user is-active --quiet qwen36-experiment || exit 73
    trap 'systemctl --user stop qwen36-experiment' EXIT
    bash $N/start-experiment.sh main-release; $PY $N/check-experiment.py 300
    $PY -I $TUN --unit qwen36-experiment --output $N/reports/$name-tunables-pre.json
    cd $N && $PY $N/soak.py --name $name $secs --profile main-release --expected-identity $MAN$post" >> "$LOG" 2>&1
}
blockers=""
for s in A B.main-release D.pilot D.matrix D.compare C.c0; do
  case "$s:$(status_of "$s")" in A:PASS|B.main-release:PASS|D.pilot:PASS|D.matrix:PASS|D.compare:PASS|D.compare:INCONCLUSIVE|C.c0:PASS|C.c0:SKIPPED_BY_PARENT) ;;
    *) blockers="$blockers $s=$(status_of "$s")";; esac
done
if [ -n "$blockers" ] && [ "$SOAK_ANYWAY" != 1 ]; then
  record E.pilot NOT_RUN "main128_already_blocked:${blockers// /,}(use_--soak-anyway)"; record E.soak NOT_RUN "see_E.pilot"
else
  [ -n "$blockers" ] && note "soak runs although main128 is blocked (--soak-anyway):$blockers"
  if passed E.pilot; then note "E.pilot already PASS"; else
    t0=$SECONDS; name=soak-pilot-$T-a$(attempt E.pilot)
    soak_run "$name" "--seconds 30"; rc=$?
    "$PY" "$N/soak-analysis.py" --name "$name" --profile main-release --expected-identity "$MAN" > "$N/reports/$name-analysis.stdout" 2>&1; arc=$?
    pv=$(jget "$N/reports/$name-analysis.json" "(d.get('pilot_view') or {}).get('all_case_types_passed')")
    jget "$N/reports/$name-analysis.json" "{'pilot_view': d.get('pilot_view'), 'verdict': d.get('verdict')}" | cut -c1-1500 | tee -a "$LOG"
    tun=$(jget "$N/reports/$name-tunables-pre.json" "d.get('verdict')")
    if [ "$pv" = True ] && [ "$tun" = PASS ]; then record E.pilot PASS "name=$name rc=$rc analysis_rc=$arc tunables=$tun t=$((SECONDS-t0))s"
    else record E.pilot FAIL "name=$name rc=$rc all_case_types_passed=$pv tunables=$tun"; fi
  fi
  if ! passed E.pilot; then record E.soak NOT_RUN "pilot_not_passed"
  elif passed E.soak; then note "E.soak already PASS"
  else
    t0=$SECONDS; name=soak-$T-a$(attempt E.soak)
    soak_run "$name" ""; rc=$?
    "$PY" "$N/soak-analysis.py" --name "$name" --profile main-release --expected-identity "$MAN" > "$N/reports/$name-analysis.stdout" 2>&1; arc=$?
    A=$N/reports/$name-analysis.json
    jget "$A" "{k: d.get(k) for k in ('verdict', 'gate_reasons', 'failure_reasons', 'inconclusive_reasons', 'two_hour_execution_gate_pass', 'elapsed_s', 'requests')}" | cut -c1-3000 | tee -a "$LOG"
    v=$(jget "$A" "d.get('verdict')"); two=$(jget "$A" "d.get('two_hour_execution_gate_pass')")
    tpre=$(jget "$N/reports/$name-tunables-pre.json" "d.get('verdict')"); tpost=$(jget "$N/reports/$name-tunables-post.json" "d.get('verdict')")
    if [ "$v" = PASS ] && [ "$two" = True ] && [ "$tpre" = PASS ] && [ "$tpost" = PASS ]; then
      record E.soak PASS "name=$name rc=$rc analysis_rc=$arc tunables=$tpre/$tpost t=$((SECONDS-t0))s"
    else record E.soak "$( [ "$v" = INCONCLUSIVE ] && echo INCONCLUSIVE || echo FAIL )" "name=$name rc=$rc verdict=$v two_hour=$two tunables=$tpre/$tpost"; fi
  fi
fi

# ============================================================================================ Z (CPU)
summary
rel=$(jget "$N/reports/final-gates-$RUN-summary.json" "d['releasable']")
note "final-gates done releasable=$rel (review reports/final-gates-$RUN-summary.json and the decision draft, then final-install.sh)"
[ "$rel" = True ]
