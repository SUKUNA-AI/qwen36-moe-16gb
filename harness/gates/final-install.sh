#!/usr/bin/env bash
# F: production install + live checks (DRAFT). The parent runs it MANUALLY, only after reviewing final-gates.sh (A-E),
# reports/final-gates-RUN-summary.json and a REVIEWED decision (copy of reports/final-gates-RUN-decision-draft.json with
# every TODO resolved). Touches production: outputs/qwen36-api/{profiles.json,qualification.json}, release.py install
# (Linux service root), release-management.ps1 (Windows controller C:\path\to\LLM\qwen36-api), then manual `qwen36`
# start/restart/stop. Never enables autostart, never edits Caddy/DNS/keys, never prints a key.
# Holds the GPU-exclusive lock for the whole window (ops-live-runbook A1) so no agent GPU job collides with production.
#   F0  preconditions: summary releasable for this run id/gather, reviewed decision (no TODO, only profiles whose gates
#       passed, main128 included, evidence present), INCONCLUSIVE speed only with --accept-inconclusive-speed, units
#       inactive, qwen36-inference not enabled                                                             [hard]
#   F1  outputs/qwen36-api/profiles.json = release-tooling draft + layer_major_attn_batch=2 (main128, long256)
#       [- decode_gather if --gather out], qualified=true exactly for the decided profiles (staged validator) [hard]
#   F2  make-qualification --dry-run, then --output outputs/qwen36-api/qualification.json (never overwritten)  [hard]
#   F3  release.py install --check -> install -> release.py rollback --manifest M --check                    [hard]
#   F4  Windows: release-management.ps1 -CheckOnly -> install -> -RollbackManifest M -CheckOnly              [hard]
#   F5  live (qwen36 via cmd.exe interop): start main128 -> readiness --edge (<=600 s) -> identity -> GLIBC_TUNABLES on
#       qwen36-inference -> network-acceptance via Caddy (60 rows) -> forge 200 / no-key 401 / health 404 -> restart ->
#       readiness -> identity (new instance) -> stop [-> --live-extra: long256/baseline64 start/identity/stop, then
#       main128 re-selected] -> status / units / autostart checks                                       [log + continue]
# Usage: final-install.sh --run-id f1 --gather in|out --decision PATH [--accept-inconclusive-speed] [--check-only]
#                         [--no-windows] [--no-live] [--live-extra]
#   --check-only: F0, F1 (no write), F2 dry-run, F3 install --check, F4 -CheckOnly; no lock, nothing written.
# Exit: 0 all executed steps PASS, 1 a step failed (rollback commands are printed, never run automatically), 2 usage.
set -uo pipefail
W=/path/to/qwen36-workspace
O=$W/outputs/qwen36-api/optimization
OUT=$W/outputs/qwen36-api
N=/home/USER/qwen36-api/optimization
ROOT=/home/USER/qwen36-api
PY=/home/USER/qwen36-api/.venv/bin/python
LOCK=/home/USER/qwen36-api/run/gpu-exclusive.lock
H=$W/work/agents/final-gates
HELP=$H/final-helpers.py
DRAFT=$W/work/agents/release-tooling/staging/profiles-draft.json
TUN=$W/work/agents/release-prep/patches/new-check-glibc-tunables.py
RM_WIN='C:\path\to\qwen36-workspace\outputs\qwen36-api\optimization\release-management.ps1'
QWEN36_CMD='C:\path\to\LLM\qwen36-api\bin\qwen36.cmd'
LOG=$N/reports/final-install.log
NOPROXY=(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy)

RUN= GATHER= DEC= ACCEPT_INC=0 CHECK_ONLY=0 NO_WIN=0 NO_LIVE=0 LIVE_EXTRA=0
ARGS=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) RUN=${2:-}; shift 2;;
    --gather) GATHER=${2:-}; shift 2;;
    --decision) DEC=${2:-}; shift 2;;
    --accept-inconclusive-speed) ACCEPT_INC=1; shift;;
    --check-only) CHECK_ONLY=1; shift;;
    --no-windows) NO_WIN=1; shift;;
    --no-live) NO_LIVE=1; shift;;
    --live-extra) LIVE_EXTRA=1; shift;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
[[ "$RUN" =~ ^[a-z0-9]{1,8}$ ]] || { echo '--run-id (the final-gates run id) is required' >&2; exit 2; }
case "$GATHER" in in|out) ;; *) echo '--gather in|out is required (must equal the final-gates run)' >&2; exit 2;; esac
[ -f "$DEC" ] || { echo '--decision PATH (reviewed decision JSON) is required' >&2; exit 2; }
DEC=$(realpath "$DEC")
if [ "$CHECK_ONLY" = 0 ] && [ "${FINAL_INSTALL_LOCKED:-}" != 1 ]; then
  exec env FINAL_INSTALL_LOCKED=1 flock -w 3600 -E 75 "$LOCK" bash "$0" "${ARGS[@]}"
fi
T=final-$RUN
SUM=$N/reports/final-gates-$RUN-summary.json
STEPS=$N/reports/final-install-$RUN.steps
FAILED=0 STARTED=0

note() { echo "$(date -u +%FT%TZ) [install $RUN] $*" | tee -a "$LOG"; }
record() { note "VERDICT $1 $2 ${3:-}"; [ "$CHECK_ONLY" = 0 ] && echo "$1 $2 ${3:-}" >> "$STEPS"; [ "$2" = FAIL ] && FAILED=1; return 0; }
passed() { [ -f "$STEPS" ] && [ "$(awk -v s="$1" '$1==s{v=$2} END{print v}' "$STEPS")" = PASS ]; }
detail_of() { [ -f "$STEPS" ] && awk -v s="$1" -v k="$2=" '$1==s{line=$0} END{n=split(line,a," "); for(i=3;i<=n;i++) if(index(a[i],k)==1){print substr(a[i],length(k)+1)}}' "$STEPS"; }
jget() { "$PY" -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" "$1" "$2" 2>/dev/null || echo MISSING; }
jstr() { "$PY" -I -c "import json,sys; d=json.loads(sys.argv[1]); print(eval(sys.argv[2]))" "$1" "$2" 2>/dev/null || echo MISSING; }
lastjson() { grep -E '^\{' | tail -1; }
units_inactive() { ! systemctl --user is-active --quiet qwen36-inference && ! systemctl --user is-active --quiet qwen36-experiment; }
rollback_hint() {
  local lm wm; lm=$(detail_of F3 manifest); wm=$(detail_of F4 manifest)
  note "ROLLBACK (not run automatically; owner decision, units stopped first):"
  [ -n "$lm" ] && note "  $PY $O/release.py rollback --manifest $lm --check   # then without --check"
  [ -n "$wm" ] && note "  powershell.exe -NoProfile -File $RM_WIN -RollbackManifest '$wm' -CheckOnly   # then without -CheckOnly"
  return 0
}
stop() { note "STOP: $*"; [ "$CHECK_ONLY" = 0 ] && echo "STOP FAIL $(echo "$*" | tr ' ' '_')" >> "$STEPS"; rollback_hint; exit 1; }
win_ps() {  # win_ps ARGS...  (Windows PowerShell through WSL interop, CRs stripped; output also to the log)
  local out rc; out=$(cd /mnt/c && powershell.exe -NoProfile -ExecutionPolicy Bypass "$@" 2>&1); rc=$?
  out=${out//$'\r'/}; printf '%s\n' "$out" >> "$LOG"; WIN_OUT=$out; return $rc
}
qwen36() {  # the owner's manual command (C:\path\to\LLM\qwen36-api\bin\qwen36.cmd) through cmd.exe interop
  local out rc; out=$(cd /mnt/c && cmd.exe /d /c "$QWEN36_CMD" "$@" 2>&1); rc=$?
  out=${out//$'\r'/}; printf '%s\n' "$out" | tee -a "$LOG" | tail -3; Q_OUT=$out; note "qwen36 $* rc=$rc"; return $rc
}
readiness() {  # readiness PROFILE -> waits <=600 s for readiness.py --edge ready, verifies profile/instance fields
  local p=$1 t0=$SECONDS out rc
  while :; do
    out=$("$ROOT/.venv/bin/python" "$ROOT/readiness.py" --edge 2>/dev/null); rc=$?
    [ "$rc" = 0 ] && break
    [ $((SECONDS-t0)) -ge 600 ] && break
    sleep 10
  done
  printf '%s\n' "$out" | lastjson >> "$LOG"
  READY_INSTANCE=$(jstr "$(printf '%s\n' "$out" | lastjson)" "d.get('instance_id')")
  local ok; ok=$(jstr "$(printf '%s\n' "$out" | lastjson)" \
    "d.get('ready') is True and d.get('backend_ready') is True and d['route']['tunnel'].get('ready') is True and d['route']['https'].get('ready') is True and d.get('model')=='qwen36-35b' and d.get('profile')=='$p' and d.get('profile_verified') is True and bool(d.get('instance_id')) and d.get('instance_id')==d['route']['tunnel'].get('instance_id')")
  note "readiness $p rc=$rc t=$((SECONDS-t0))s ok=$ok instance=${READY_INSTANCE:0:12}"
  [ "$ok" = True ]
}
identity() {  # identity PROFILE [--expect-instance I | --different-from I]
  local out rc; out=$("${NOPROXY[@]}" "$PY" "$O/ops-live-probe.py" identity --live --profile "$@" 2>&1); rc=$?
  printf '%s\n' "$out" | lastjson | cut -c1-2000 >> "$LOG"
  ID_INSTANCE=$(jstr "$(printf '%s\n' "$out" | lastjson)" "d.get('instance_id')")
  note "identity $* rc=$rc instance=${ID_INSTANCE:0:12}"; return $rc
}
cleanup_live() { if [ "$STARTED" = 1 ] && systemctl --user is-active --quiet qwen36-inference; then note "cleanup: qwen36 stop"; qwen36 stop; fi; }

cd "$N" || exit 1
note "final-install start check_only=$CHECK_ONLY gather=$GATHER decision=$DEC no_windows=$NO_WIN no_live=$NO_LIVE live_extra=$LIVE_EXTRA"

# ============================================================================================ F0 preconditions
[ -f "$SUM" ] || stop "missing $SUM (run final-gates.sh first)"
[ "$(jget "$SUM" "d['gather']")" = "$GATHER" ] || stop "summary gather=$(jget "$SUM" "d['gather']") differs from --gather $GATHER"
[ "$(jget "$SUM" "d['releasable']")" = True ] || stop "final-gates summary: main128 not releasable"
SPEED=$(jget "$SUM" "d['steps'].get('D.compare',{}).get('status')")
if [ "$SPEED" = INCONCLUSIVE ] && [ "$ACCEPT_INC" != 1 ]; then
  stop "speed verdict INCONCLUSIVE (non-inferiority not established): pass --accept-inconclusive-speed only with an owner decision and a matching known limitation (no speed claim)"
fi
QUAL=$("$PY" -I "$HELP" decision-check --decision "$DEC" --summary "$SUM" --reports "$N/reports" 2>>"$LOG") || stop "decision check failed (see log)"
note "decided profiles: $QUAL (speed verdict $SPEED)"
units_inactive || stop "qwen36-inference or qwen36-experiment is active"
[ "$(systemctl --user is-enabled qwen36-inference 2>/dev/null)" = enabled ] && stop "qwen36-inference is ENABLED (autostart); not allowed"
record F0 PASS "qualified=$QUAL speed=$SPEED"

# ============================================================================================ F1 profiles.json
PJ=$OUT/profiles.json
if [ "$CHECK_ONLY" = 0 ] && [ -f "$PJ" ]; then
  mkdir -p "$H/backup"; cp -p "$PJ" "$H/backup/profiles.json.$(date -u +%Y%m%dT%H%M%SZ).$(sha256sum < "$PJ" | cut -c1-12)"
fi
out=$("$PY" -I "$HELP" profiles --gather "$GATHER" --draft "$DRAFT" --qualified "$QUAL" --out "$PJ" \
  --validator "$OUT/profile-runtime.py" $([ "$CHECK_ONLY" = 1 ] && echo --check) 2>&1); rc=$?
note "profiles: $out"
[ "$rc" = 0 ] || stop "profiles.json build/validation failed"
record F1 PASS "sha=$(jstr "$(printf '%s\n' "$out" | lastjson)" "d['sha256'][:16]") state=$(jstr "$(printf '%s\n' "$out" | lastjson)" "d['state']")"

# ============================================================================================ F2 qualification receipt
QJ=$OUT/qualification.json
dry=$("$PY" -B "$O/make-qualification.py" --decision "$DEC" --dry-run 2>&1); rc=$?
printf '%s\n' "$dry" >> "$LOG"
[ "$rc" = 0 ] || stop "make-qualification --dry-run refused: $(printf '%s\n' "$dry" | tail -1 | cut -c1-300)"
want=$(jstr "$(printf '%s\n' "$dry" | lastjson)" "d['receipt_sha256']")
note "make-qualification dry-run: profiles=$(jstr "$(printf '%s\n' "$dry" | lastjson)" "d['profiles']") modules=$(jstr "$(printf '%s\n' "$dry" | lastjson)" "d['kernel_required_modules']") receipt=${want:0:16}"
if [ -f "$QJ" ]; then
  [ "$(sha256sum < "$QJ" | cut -c1-64)" = "$want" ] || stop "$QJ exists with a different receipt (never overwritten): owner must move it deliberately"
  note "qualification.json already equals the dry-run receipt"
elif [ "$CHECK_ONLY" = 0 ]; then
  "$PY" -B "$O/make-qualification.py" --decision "$DEC" --output "$QJ" >> "$LOG" 2>&1 || stop "make-qualification --output failed"
  [ "$(sha256sum < "$QJ" | cut -c1-64)" = "$want" ] || stop "written receipt differs from the dry-run receipt"
fi
record F2 PASS "receipt=${want:0:16} $([ "$CHECK_ONLY" = 1 ] && echo dry-run-only)"

# ============================================================================================ F3 Linux release
if passed F3; then note "F3 already PASS (manifest $(detail_of F3 manifest))"; else
  chk=$("$PY" -B "$O/release.py" install --check 2>&1); rc=$?; printf '%s\n' "$chk" >> "$LOG"
  [ "$rc" = 0 ] && [ "$(jstr "$(printf '%s\n' "$chk" | lastjson)" "d.get('check')")" = PASS ] || stop "release.py install --check refused: $(printf '%s\n' "$chk" | tail -1 | cut -c1-300)"
  note "install --check: $(jstr "$(printf '%s\n' "$chk" | lastjson)" "{k: d.get(k) for k in ('default','qualified_profiles','native_libraries','start_profile')}")"
  if [ "$CHECK_ONLY" = 1 ]; then record F3 PASS "install_--check_only (release.py install NOT run)"; else
    ins=$("$PY" -B "$O/release.py" install 2>&1); rc=$?; printf '%s\n' "$ins" >> "$LOG"
    M=$(jstr "$(printf '%s\n' "$ins" | lastjson)" "d.get('rollback_manifest')")
    [ "$rc" = 0 ] && [ -f "$M" ] || stop "release.py install failed (rc=$rc): $(printf '%s\n' "$ins" | tail -1 | cut -c1-300)"
    rb=$("$PY" -B "$O/release.py" rollback --manifest "$M" --check 2>&1); rrc=$?; printf '%s\n' "$rb" >> "$LOG"
    [ "$rrc" = 0 ] && [ "$(jstr "$(printf '%s\n' "$rb" | lastjson)" "d.get('rollback_check')")" = PASS ] || { record F3 FAIL "manifest=$M rollback_check_rc=$rrc"; stop "rollback --check failed for $M"; }
    record F3 PASS "manifest=$M auto_start=$(jstr "$(printf '%s\n' "$ins" | lastjson)" "d.get('auto_start')")"
  fi
fi

# ============================================================================================ F4 Windows controller release
if [ "$NO_WIN" = 1 ]; then record F4 NOT_RUN "--no-windows (run release-management.ps1 per OPERATIONS-release-draft.md section 1.8)"
elif passed F4; then note "F4 already PASS (manifest $(detail_of F4 manifest))"
else
  win_ps -File "$RM_WIN" -CheckOnly; rc=$?
  [ "$rc" = 0 ] && grep -q 'Management release check PASS' <<<"$WIN_OUT" || stop "release-management.ps1 -CheckOnly failed (rc=$rc)"
  if [ "$CHECK_ONLY" = 1 ]; then record F4 PASS "-CheckOnly_only (Windows release NOT installed)"; else
    win_ps -File "$RM_WIN"; rc=$?
    WM=$(sed -n 's/^Management release installed\. Rollback manifest: //p' <<<"$WIN_OUT" | tail -1)
    [ "$rc" = 0 ] && [ -n "$WM" ] || stop "release-management.ps1 install failed (rc=$rc)"
    win_ps -File "$RM_WIN" -RollbackManifest "$WM" -CheckOnly; rc=$?
    [ "$rc" = 0 ] && grep -q 'Management rollback hash check PASS' <<<"$WIN_OUT" || { record F4 FAIL "manifest=$WM rollback_check_rc=$rc"; stop "Windows rollback -CheckOnly failed"; }
    record F4 PASS "manifest=$WM"
  fi
fi
if [ "$CHECK_ONLY" = 1 ]; then note "check-only done (FAILED=$FAILED)"; exit "$FAILED"; fi

# ============================================================================================ F5 live operations
if [ "$NO_LIVE" = 1 ]; then record F5 NOT_RUN "--no-live (ops-live-runbook phases B/D by hand)"; rollback_hint; exit "$FAILED"; fi
command -v cmd.exe >/dev/null && command -v powershell.exe >/dev/null || stop "Windows interop (cmd.exe/powershell.exe) unavailable: use --no-live and the ops-live runbook"
trap cleanup_live EXIT
units_inactive || stop "a model unit became active before the live window"
qwen36 start --profile main128; rc=$?; STARTED=1
[ "$rc" = 0 ] && record F5.start PASS "rc=0" || record F5.start FAIL "rc=$rc"
if readiness main128; then record F5.ready PASS "instance=${READY_INSTANCE:0:16}"; else record F5.ready FAIL "not ready in 600 s"; fi
I1=$READY_INSTANCE
identity main128 --expect-instance "$I1" && record F5.identity PASS "instance=${I1:0:16}" || record F5.identity FAIL "see log"
"$PY" -I "$TUN" --unit qwen36-inference --output "$N/reports/$T-install-tunables-inference-$(date -u +%H%M%S).json" >> "$LOG" 2>&1 \
  && record F5.tunables PASS "qwen36-inference" || record F5.tunables FAIL "GLIBC_TUNABLES not active in every unit process"
NAME=network-release-$T-$(date -u +%Y%m%dT%H%M%S)
"${NOPROXY[@]}" "$PY" "$N/network-acceptance.py" --name "$NAME" >> "$LOG" 2>&1; rc=$?
NR=$N/reports/$NAME.json
nchk=$(jget "$NR" "(d.get('pass') is True, len(d.get('rows',[])), d.get('same_instance'), d.get('instance_id')=='$I1', d.get('tls_verified'), any(r.get('case')=='existing-forge-site' and r.get('pass') for r in d.get('rows',[])), d.get('external_e2e'))")
note "network-acceptance rc=$rc (pass, rows, same_instance, instance==I1, tls_verified, forge_row, external_e2e) = $nchk"
case "$nchk" in "(True, 60, True, True, True, True, 'NOT_RUN')") record F5.network PASS "report=$NAME.json rows=60";;
  "(True, "*) record F5.network FAIL "pass_but_unexpected_shape $nchk";; *) record F5.network FAIL "rc=$rc $nchk";; esac
code() { "${NOPROXY[@]}" curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "$1" 2>/dev/null || echo ERR; }
f=$(code https://forge.example.com/); m=$(code https://llm.example.com/v1/models); hh=$(code https://llm.example.com/health)
note "LAN view (not external E2E): forge=$f llm/v1/models(no key)=$m llm/health=$hh"
[ "$f:$m:$hh" = "200:401:404" ] && record F5.sites PASS "forge=200 models_nokey=401 health=404" || record F5.sites FAIL "forge=$f models_nokey=$m health=$hh"
qwen36 restart; rc=$?
if readiness main128 && identity main128 --different-from "$I1"; then record F5.restart PASS "rc=$rc new_instance=${ID_INSTANCE:0:16}"
else record F5.restart FAIL "rc=$rc"; fi
qwen36 stop; rc=$?
qwen36 status; src=$?
units_inactive && [ "$rc" = 0 ] && [ "$src" != 0 ] && record F5.stop PASS "stop=0 status=$src" || record F5.stop FAIL "stop=$rc status=$src units_inactive=$(units_inactive && echo yes || echo no)"
if [ "$LIVE_EXTRA" = 1 ]; then
  for P in long256 baseline64; do
    case ",$QUAL," in *",$P,"*) ;; *) note "live-extra: $P not qualified, skipped"; continue;; esac
    qwen36 start --profile "$P"; rc=$?
    if [ "$rc" = 0 ] && readiness "$P" && identity "$P"; then record "F5.extra-$P" PASS "instance=${ID_INSTANCE:0:16}"; else record "F5.extra-$P" FAIL "rc=$rc"; fi
    qwen36 stop
  done
  # restore the default selection (config/selected-profile survives a stop): main128 start -> ready -> stop
  qwen36 start --profile main128; readiness main128 && record F5.reselect PASS "main128 selected" || record F5.reselect FAIL "main128 re-selection"
  qwen36 stop
else
  record F5.extra NOT_RUN "--live-extra not given (ops-live-runbook: optional per extra qualified profile)"
fi
STARTED=0
units_inactive && record F5.final-stopped PASS "units inactive" || record F5.final-stopped FAIL "a unit is still active"
[ "$(systemctl --user is-enabled qwen36-inference 2>/dev/null)" != enabled ] && record F5.no-autostart-linux PASS "unit not enabled" || record F5.no-autostart-linux FAIL "unit enabled"
( cd /mnt/c && reg.exe query 'HKCU\Software\Microsoft\Windows\CurrentVersion\Run' /v Qwen36Api ) >/dev/null 2>&1 \
  && record F5.no-autostart-windows FAIL "HKCU Run Qwen36Api present" || record F5.no-autostart-windows PASS "no HKCU Run entry"
note "final-install done FAILED=$FAILED (model left stopped; external E2E from outside the LAN stays NOT_RUN)"
[ "$FAILED" = 0 ] || rollback_hint
exit "$FAILED"
