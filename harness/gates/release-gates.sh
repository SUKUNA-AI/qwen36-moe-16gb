#!/usr/bin/env bash
# Release gates for the next main128 candidate (schema-2 evidence). final-gates.sh stays the archived f1 driver.
#   A  CPU  units idle, overlay == this checkout, candidate config/allowlist, run-bound REF/FB, fresh report names,
#           kernel provenance, full CPU suite                                                              [hard]
#   B  GPU  qualify.py --name --reference-report --functional-baseline: teacher 0-diff + bound functional  [hard]
#   D  GPU  tool pilot -> matrix BASELINE,CANDIDATE -> compare-ab (--release) or median screen (default)
#   C  GPU  run-context-gates.sh CANDIDATE 131072 120000 NAME -> context-check (bound to B's teacher run)
#   E  GPU  soak pilot 30 s -> soak (1800 s default, 7200 s --release) -> soak-analysis (VRAM floor 0.8 GiB)
#   Z  CPU  summary reports/release-gates-RUN-summary.json
# Site paths come from config/local.env (QWEN36_ROOT, QWEN36_NATIVE). Every report name carries the run id.
# Usage: release-gates.sh --candidate C --reference-report REF --functional-baseline FB [--baseline main-release]
#        [--pilot-profile bf128] [--run-id ID] [--release] [--dry-run] [--env FILE] [--cpu-python P] [--soak-anyway]
#   default    screen: 3 A/B pairs, 30 min soak; never releasable
#   --release  5 pairs + compare-ab, 2 h soak
#   --dry-run  CPU only, read-only: the A checks that need no writes, then the exact GPU commands
#   re-runs    steps already PASS for this run id are skipped (reports/release-gates-RUN.steps, append-only)
# Exit: 0 every gate passed (with --release: main128 releasable), 1 otherwise / stopped, 2 usage.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
C=$(cd -- "$HERE/../.." && pwd)
HELP=$HERE/release-helpers.py
ENV_FILE=$C/config/local.env
CANDIDATE= BASELINE=main-release PILOT_PROFILE=bf128 REF= FB= RUN= MODE=run RELEASE=0 SOAK_ANYWAY=0
CPU_PY=${QWEN36_CPU_PYTHON:-}
while [ $# -gt 0 ]; do
  case "$1" in
    --candidate) CANDIDATE=${2:-}; shift 2;;
    --baseline) BASELINE=${2:-}; shift 2;;
    --pilot-profile) PILOT_PROFILE=${2:-}; shift 2;;
    --reference-report) REF=${2:-}; shift 2;;
    --functional-baseline) FB=${2:-}; shift 2;;
    --run-id) RUN=${2:-}; shift 2;;
    --env) ENV_FILE=${2:-}; shift 2;;
    --cpu-python) CPU_PY=${2:-}; shift 2;;
    --release) RELEASE=1; shift;;
    --dry-run) MODE=dry-run; shift;;
    --soak-anyway) SOAK_ANYWAY=1; shift;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
NAME_RE='^[a-z0-9][a-z0-9.-]{0,60}$'
for v in "$CANDIDATE" "$BASELINE" "$PILOT_PROFILE" "$REF" "$FB"; do
  [[ "$v" =~ $NAME_RE ]] || { echo 'need --candidate, --reference-report, --functional-baseline ([a-z0-9.-] names)' >&2; exit 2; }
done
[ "$CANDIDATE" != "$BASELINE" ] || { echo 'candidate and baseline must differ' >&2; exit 2; }
RUN=${RUN:-$(date -u +%Y%m%dt%H%M%S)}
[[ "$RUN" =~ ^[a-z0-9]{1,16}$ ]] || { echo 'bad --run-id (want [a-z0-9]{1,16})' >&2; exit 2; }
SITE=$(python3 -I "$HELP" env "$ENV_FILE") || { echo "site paths from $ENV_FILE are missing or placeholders" >&2; exit 2; }
eval "$SITE"
S=$QWEN36_ROOT N=$QWEN36_NATIVE
PY=${QWEN36_GPU_PYTHON:-$S/.venv/bin/python}
LOCK=${QWEN36_GPU_LOCK:-$S/run/gpu-exclusive.lock}
export QWEN36_DATA_ROOT=$N QWEN36_HARNESS_SOURCE=$N/config QWEN36_GPU_PYTHON=$PY QWEN36_GPU_LOCK=$LOCK
export QWEN36_BENCH_WARMUP_S=${QWEN36_BENCH_WARMUP_S:-60}

if [ "$RELEASE" = 1 ]; then PAIRS=5 SOAK_S=7200; else PAIRS=3 SOAK_S=1800; fi
T=rg-$RUN
Q=$T-q-$CANDIDATE                     # qualify.py: $Q-teacher, $Q-functional, quality-$Q
PILOT=$T-tool M=$T-m CTX=$T-c128 SOAKP=soak-$T-pilot SOAK=soak-$T
MAN=$N/reports/$M-manifest.json
CMP=$N/reports/$M-$CANDIDATE-vs-$BASELINE-comparison.json
SCREEN=$N/reports/$T-screen.json
STEPS=$N/reports/release-gates-$RUN.steps LOG=$N/reports/release-gates-$RUN.log
SUMMARY=$N/reports/release-gates-$RUN-summary.json
CACHE=$(compgen -G "$S/.venv/lib/python3*/site-packages/freetoken_kernel_cache/jit_cache" | head -1)
CONFIG="candidate=$CANDIDATE baseline=$BASELINE pilot=$PILOT_PROFILE ref=$REF fb=$FB release=$RELEASE"
# Every report this run may create; none may exist before its first step.
OUTPUTS=("$Q-teacher" "$Q-functional" "quality-$Q" "$T-qrun-teacher" "$T-qrun-functional" "$PILOT-tool-pilot"
         "$M-matrix" "$M-manifest" "$T-screen" "$CTX-functional" "$CTX-boundary" "$CTX-backend" "$CTX-cancel"
         "$CTX-run" "$SOAKP" "$SOAK" "release-gates-$RUN-summary")

QUALIFY=("$PY" "$N/qualify.py" --candidate "$CANDIDATE" --name "$Q" --reference-report "$REF" --functional-baseline "$FB")
PILOT_CMD=("$PY" "$N/tool-pilot.py" --profile "$PILOT_PROFILE" --repeats 3 --name "$PILOT")
MATRIX=("$PY" "$N/matrix.py" --profiles "$BASELINE,$CANDIDATE" --pairs "$PAIRS" --name "$M" --output 128 --tool-pilot "$PILOT")
COMPARE=("$PY" "$N/compare-ab.py" --name "$M" --baseline "$BASELINE" --candidate "$CANDIDATE" --output 128)
CONTEXT=(bash "$N/run-context-gates.sh" "$CANDIDATE" 131072 120000 "$CTX")
SOAK_BLOCK='set -e; ! systemctl --user is-active --quiet qwen36-experiment || exit 73
trap "systemctl --user stop qwen36-experiment" EXIT
bash "$1/start-experiment.sh" "$2"; "$3" "$1/check-experiment.py" 300
cd "$1" && "$3" "$1/soak.py" --name "$4" --seconds "$5" --profile "$2" --expected-identity "$6"'
soak_cmd() { SOAK_CMD=(bash -c "$SOAK_BLOCK" soak "$N" "$CANDIDATE" "$PY" "$1" "$2" "$MAN"); }
FAILS=0

note() { local line; line="$(date -u +%FT%TZ) [$RUN] $*"; echo "$line"; [ "$MODE" = run ] && echo "$line" >> "$LOG"; return 0; }
logged() { if [ "$MODE" = run ]; then "$@" >> "$LOG" 2>&1; else "$@"; fi; }
record() { note "VERDICT $1 $2 ${3:-}"; [ "$MODE" = run ] && echo "$1 $2 ${3:-}" >> "$STEPS"; return 0; }
status_of() { [ -f "$STEPS" ] && awk -v s="$1" '$1==s{v=$2} END{print v}' "$STEPS"; }
passed() { [ "$(status_of "$1")" = PASS ]; }
check() {
  local label=$1; shift
  if "$@"; then note "ok: $label"; return 0; fi
  FAILS=$((FAILS+1))
  if [ "$MODE" = run ]; then stop "precondition: $label"; else note "DRY-RUN FAIL: $label"; fi
  return 1
}
retry() {  # retry only while the lock wait timed out (75) or a model unit was active (73)
  local label=$1; shift; local t=0 rc
  while :; do "$@"; rc=$?; { [ "$rc" -ne 75 ] && [ "$rc" -ne 73 ]; } && break; t=$((t+1)); [ "$t" -ge 1200 ] && break; sleep 3; done
  note "$label rc=$rc"; return "$rc"
}
lockrun() { flock -w 3600 -E 75 "$LOCK" "$@"; }
selflock() { flock -w 3600 -E 75 "$LOCK" true && "$@"; }  # tools that take the lock themselves: queue, then let them lock
helper() { "$PY" -I "$HELP" "$@"; }
show() {  # one line, shell-quoted only where needed (the dry-run plan)
  local a out=(); for a in "$@"; do [[ "$a" =~ ^[A-Za-z0-9_./,:=@+-]+$ ]] && out+=("$a") || out+=("$(printf '%q' "$a")"); done
  echo "${out[*]}"
}
summary() {
  [ "$MODE" = run ] || return 0
  local args=(summary --steps "$STEPS" --run "$RUN" --out "$SUMMARY"); [ "$RELEASE" = 1 ] && args+=(--release)
  helper "${args[@]}" 2>&1 | tee -a "$LOG"
}
stop() { note "STOP: $*"; [ "$MODE" = run ] && echo "STOP FAIL $(echo "$*" | tr ' ' '_')" >> "$STEPS"; summary; exit 1; }
units_inactive() { ! systemctl --user is-active --quiet qwen36-inference && ! systemctl --user is-active --quiet qwen36-experiment; }
fresh_names() { local n bad=(); for n in "${OUTPUTS[@]}"; do [ -e "$N/reports/$n.json" ] && bad+=("$n"); done
  [ ${#bad[@]} = 0 ] || { echo "existing reports: ${bad[*]}" >&2; return 1; }; }
overlay_matches() {  # the overlay must run this checkout's bytes (the run receipt hashes the same suffixes)
  local f
  for f in qualify.py quality.py compare-quality.py bench.py matrix.py compare-ab.py tool-pilot.py soak.py soak-analysis.py \
           make-qualification.py run-context-gates.sh; do
    cmp -s "$C/harness/$f" "$N/$f" || { echo "overlay $f differs from $C/harness/$f" >&2; return 1; }
  done
  ( cd "$C/freetoken/python/freetoken" && find . -type f \( -name '*.py' -o -name '*.cu' -o -name '*.h' -o -name '*.cuh' \
      -o -name '*.cpp' \) -not -path '*/__pycache__/*' -print0 | xargs -0 sha256sum ) | ( cd "$N/python/freetoken" && sha256sum --quiet -c - )
}
config_unqualified() { "$PY" -I -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(d.get("qualified") is not False)' "$1"; }
trap 'note "interrupted (signal)"; exit 130' INT TERM

note "release-gates start mode=$MODE $CONFIG pairs=$PAIRS soak_s=$SOAK_S site=$ENV_FILE native=$N"
if [ "$MODE" = run ]; then
  mkdir -p "$N/reports"
  prev=$(awk '$1=="CONFIG"{$1=""; sub(/^ /,""); v=$0} END{print v}' "$STEPS" 2>/dev/null)
  [ -n "$prev" ] && [ "$prev" != "$CONFIG" ] && { note "STOP: run $RUN was started with '$prev'; use a new --run-id"; exit 2; }
  [ -z "$prev" ] && echo "CONFIG $CONFIG" >> "$STEPS"
fi

# ============================================================================================ A (CPU, hard)
if [ "$MODE" = run ] && passed A; then note "A already PASS (skip)"; else
  A0=$SECONDS
  check "units qwen36-inference/qwen36-experiment inactive" units_inactive
  for f in qualify.py quality.py compare-quality.py matrix.py compare-ab.py tool-pilot.py soak.py soak-analysis.py \
           run-context-gates.sh start-experiment.sh check-experiment.py config/$CANDIDATE.json config/$BASELINE.json \
           config/$PILOT_PROFILE.json config/teacher-fixture.json; do
    check "overlay has $f" test -f "$N/$f"
  done
  check "overlay harness + FreeToken sources == this checkout" overlay_matches
  check "candidate config is qualified:false" config_unqualified "$N/config/$CANDIDATE.json"
  check "teacher allowlist has teacher-$CANDIDATE" grep -q "'teacher-$CANDIDATE'" "$N/python/freetoken/engine/qualification.py"
  check "REF $REF / FB $FB are run-bound schema-2 evidence" logged helper evidence --reports "$N/reports" --reference "$REF" --functional-baseline "$FB"
  if [ "$MODE" = dry-run ] || [ ! -s "$STEPS" ] || [ "$(wc -l < "$STEPS")" -le 1 ]; then
    check "report names of run $RUN are unused" fresh_names
  fi
  if [ "$MODE" = run ]; then
    CUDA_VISIBLE_DEVICES= "$PY" -I "$N/python/freetoken/kernel/kernel_provenance.py" check --kernel-dir "$N/python/freetoken/kernel" \
      --cache-dir "$CACHE" --require-extension _pinned_tensor > "$N/reports/$T-kernel-provenance.json" 2>&1
    check "kernel provenance: all prebuilt" logged "$PY" -I "$HERE/final-helpers.py" provenance "$N/reports/$T-kernel-provenance.json"
    check "CPU suite needs --cpu-python (isolated venv, not the serving venv)" test -n "$CPU_PY"
    QWEN36_CPU_PYTHON=$CPU_PY bash "$C/harness/CPU-checks.sh" --output "$N/reports/$T-cpu" >> "$LOG" 2>&1; rc=$?
    check "CPU suite rc=$rc ($N/reports/$T-cpu/summary.json)" test "$rc" = 0
    record A PASS "t=$((SECONDS-A0))s"
  else
    note "dry-run: kernel provenance and the CPU suite run only in real mode"
  fi
fi

if [ "$MODE" = dry-run ]; then
  soak_cmd "$SOAKP" 30; P1=$(show "${SOAK_CMD[@]}"); soak_cmd "$SOAK" "$SOAK_S"; P2=$(show "${SOAK_CMD[@]}")
  if [ "$RELEASE" = 1 ]; then DCMP=$(show "${COMPARE[@]}"); else DCMP="release-helpers.py screen --matrix $M -> $SCREEN (not a gate)"; fi
  cat <<EOF
plan (run $RUN, $([ "$RELEASE" = 1 ] && echo release || echo screen) mode; GPU steps under flock -w 3600 -E 75 $LOCK):
  B  $(show "${QUALIFY[@]}")
     qcheck $N/reports/quality-$Q.json; extract-run $Q-teacher --teacher, $Q-functional
  D  $(show "${PILOT_CMD[@]}")
     $(show "${MATRIX[@]}")
     $DCMP
  C  $(show "${CONTEXT[@]}")
     context-check --name $CTX --teacher-report $Q-teacher --functional-report $Q-functional
  E  $P1
     $P2
     soak-analysis --name $SOAK --profile $CANDIDATE --expected-identity $MAN; soak-check$([ "$RELEASE" = 1 ] && echo ' --release')
  Z  release-helpers.py summary -> $SUMMARY
dry-run precondition failures: $FAILS
EOF
  exit $(( FAILS > 0 ))
fi

# ============================================================================================ B (GPU, hard)
if passed B; then note "B already PASS (skip)"; else
  t0=$SECONDS
  retry "qualify $CANDIDATE" lockrun "${QUALIFY[@]}" >> "$LOG" 2>&1; rc=$?
  helper qcheck "$N/reports/quality-$Q.json" 2>&1 | tee -a "$LOG"; q=${PIPESTATUS[0]}
  if [ "$rc" != 0 ] || [ "$q" != 0 ]; then record B FAIL "qualify_rc=$rc qcheck_rc=$q"; stop "teacher/quality gate failed for $CANDIDATE"; fi
  logged helper extract-run --report "$N/reports/$Q-teacher.json" --profile "teacher-$CANDIDATE" --config "$N/config/$CANDIDATE.json" \
    --teacher --out "$N/reports/$T-qrun-teacher.json" || { record B FAIL "teacher run receipt"; stop "teacher receipt differs from config"; }
  logged helper extract-run --report "$N/reports/$Q-functional.json" --profile "$CANDIDATE" --config "$N/config/$CANDIDATE.json" \
    --out "$N/reports/$T-qrun-functional.json" || { record B FAIL "functional run receipt"; stop "functional receipt differs from config"; }
  record B PASS "t=$((SECONDS-t0))s quality=quality-$Q.json"
fi

# ============================================================================================ D (GPU)
if passed D.pilot; then note "D.pilot already PASS"; else
  retry "tool-pilot $PILOT" selflock "${PILOT_CMD[@]}" >> "$LOG" 2>&1; rc=$?
  [ "$rc" = 0 ] && record D.pilot PASS "name=$PILOT" || { record D.pilot FAIL "rc=$rc"; stop "tool pilot failed: no matrix, no identity for C/E"; }
fi
if passed D.matrix || [ "$(status_of D.matrix)" = INVALID ]; then note "D.matrix already $(status_of D.matrix)"; else
  retry "matrix $M" selflock "${MATRIX[@]}" >> "$N/reports/$M.console.log" 2>&1; rc=$?
  case $rc in 0) record D.matrix PASS "pairs=$PAIRS name=$M";; 3) record D.matrix INVALID "rc=3 arm_below_VRAM_floor";; *) record D.matrix FAIL "rc=$rc";; esac
fi
[ -f "$MAN" ] || stop "matrix manifest $MAN missing: E has no identity reference"
if [ -n "$(status_of D.compare)" ] && [ "$(status_of D.compare)" != ERROR ]; then note "D.compare already $(status_of D.compare)"
elif [ "$(status_of D.matrix)" = FAIL ]; then record D.compare NOT_RUN "matrix_incomplete"
elif [ "$RELEASE" = 1 ]; then
  [ -f "$CMP" ] || retry "compare-ab" selflock "${COMPARE[@]}" >> "$LOG" 2>&1
  v=$("$PY" -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("verdict") if d.get("contract_pass") is True else "ERROR")' "$CMP" 2>/dev/null || echo ERROR)
  record D.compare "$v" "comparison=$(basename "$CMP")"
else
  logged helper screen --reports "$N/reports" --matrix "$M" --baseline "$BASELINE" --candidate "$CANDIDATE" --out "$SCREEN" \
    && record D.compare SCREEN "screen=$(basename "$SCREEN") pairs=$PAIRS" || record D.compare ERROR "screen"
fi

# ============================================================================================ C (GPU)
if passed C; then note "C already PASS"; else
  retry "context gates $CTX" lockrun "${CONTEXT[@]}" >> "$LOG" 2>&1; rc=$?
  if [ "$rc" = 0 ] && logged helper context-check --reports "$N/reports" --name "$CTX" --profile "$CANDIDATE" \
       --teacher-report "$Q-teacher" --functional-report "$Q-functional"; then record C PASS "name=$CTX"
  else record C FAIL "name=$CTX rc=$rc"; fi
fi

# ============================================================================================ E (GPU)
blockers=""
for s in A B D.pilot D.matrix C; do passed "$s" || blockers="$blockers $s=$(status_of "$s")"; done
if [ -n "$blockers" ] && [ "$SOAK_ANYWAY" != 1 ]; then
  record E.pilot NOT_RUN "blocked:${blockers// /,}(use_--soak-anyway)"; record E.soak NOT_RUN "see_E.pilot"
else
  soak_gate() {  # soak_gate STEP NAME SECONDS [--release]
    local step=$1 name=$2 secs=$3; shift 3
    if passed "$step"; then note "$step already PASS"; return 0; fi
    soak_cmd "$name" "$secs"; retry "soak $name" lockrun "${SOAK_CMD[@]}" >> "$LOG" 2>&1; local rc=$?
    "$PY" "$N/soak-analysis.py" --name "$name" --profile "$CANDIDATE" --expected-identity "$MAN" >> "$LOG" 2>&1
    if logged helper soak-check --analysis "$N/reports/$name-analysis.json" "$@"; then record "$step" PASS "name=$name rc=$rc"
    else record "$step" FAIL "name=$name rc=$rc"; fi
  }
  soak_gate E.pilot "$SOAKP" 30
  if passed E.pilot; then
    if [ "$RELEASE" = 1 ]; then soak_gate E.soak "$SOAK" "$SOAK_S" --release; else soak_gate E.soak "$SOAK" "$SOAK_S"; fi
  else record E.soak NOT_RUN "pilot_not_passed"; fi
fi

# ============================================================================================ Z (CPU)
summary
ok=$("$PY" -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["releasable"] if sys.argv[2]=="1" else d["gates_pass"])' "$SUMMARY" "$RELEASE")
note "release-gates done ok=$ok (review $SUMMARY)"
[ "$ok" = True ]
