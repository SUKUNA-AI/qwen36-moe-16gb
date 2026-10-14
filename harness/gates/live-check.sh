#!/usr/bin/env bash
# F5 Linux-side live checks for the manual (Windows-driven) live window; mirrors final-install.sh F5 (WSL interop is
# unavailable, so `qwen36` runs from Windows PowerShell and this script does the Linux-side checks).
# Usage: live-check.sh ready PROFILE | identity PROFILE [--expect-instance I | --different-from I] | tunables | network |
#        sites | units | record STEP STATUS DETAIL
set -uo pipefail
W=/path/to/qwen36-workspace
O=$W/outputs/qwen36-api/optimization
N=/home/USER/qwen36-api/optimization
ROOT=/home/USER/qwen36-api
PY=$ROOT/.venv/bin/python
TUN=$W/work/agents/release-prep/patches/new-check-glibc-tunables.py
LOG=$N/reports/final-install.log
STEPS=$N/reports/final-install-f1.steps
NOPROXY=(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy)
note() { echo "$(date -u +%FT%TZ) [install f1 manual-F5] $*" | tee -a "$LOG"; }
jstr() { "$PY" -I -c "import json,sys; d=json.loads(sys.argv[1]); print(eval(sys.argv[2]))" "$1" "$2" 2>/dev/null || echo MISSING; }
cd "$N" || exit 1
case "${1:-}" in
  ready)
    p=$2; t0=$SECONDS
    while :; do out=$("$PY" "$ROOT/readiness.py" --edge 2>/dev/null); rc=$?; [ "$rc" = 0 ] && break; [ $((SECONDS-t0)) -ge 600 ] && break; sleep 10; done
    last=$(printf '%s\n' "$out" | grep -E '^\{' | tail -1); printf '%s\n' "$last" >> "$LOG"
    ok=$(jstr "$last" "d.get('ready') is True and d.get('backend_ready') is True and d['route']['tunnel'].get('ready') is True and d['route']['https'].get('ready') is True and d.get('model')=='qwen36-35b' and d.get('profile')=='$p' and d.get('profile_verified') is True and bool(d.get('instance_id')) and d.get('instance_id')==d['route']['tunnel'].get('instance_id')")
    inst=$(jstr "$last" "d.get('instance_id')")
    note "readiness $p rc=$rc t=$((SECONDS-t0))s ok=$ok instance=$inst"
    echo "INSTANCE=$inst"; [ "$ok" = True ];;
  identity)
    shift; out=$("${NOPROXY[@]}" "$PY" "$O/ops-live-probe.py" identity --live --profile "$@" 2>&1); rc=$?
    last=$(printf '%s\n' "$out" | grep -E '^\{' | tail -1); printf '%s\n' "$last" | cut -c1-2000 >> "$LOG"
    note "identity $* rc=$rc instance=$(jstr "$last" "d.get('instance_id')")"; [ "$rc" = 0 ];;
  tunables)
    "$PY" -I "$TUN" --unit qwen36-inference --output "$N/reports/final-f1-install-tunables-inference-$(date -u +%H%M%S).json" >> "$LOG" 2>&1; rc=$?
    note "tunables qwen36-inference rc=$rc"; [ "$rc" = 0 ];;
  network)
    I1=$2; NAME=network-release-final-f1-$(date -u +%Y%m%dT%H%M%S)
    "${NOPROXY[@]}" "$PY" "$N/network-acceptance.py" --name "$NAME" >> "$LOG" 2>&1; rc=$?
    NR=$N/reports/$NAME.json
    nchk=$("$PY" -I -c "import json,sys; d=json.load(open(sys.argv[1])); print((d.get('pass') is True, len(d.get('rows',[])), d.get('same_instance'), d.get('instance_id')==sys.argv[2], d.get('tls_verified'), any(r.get('case')=='existing-forge-site' and r.get('pass') for r in d.get('rows',[])), d.get('external_e2e')))" "$NR" "$I1" 2>/dev/null || echo MISSING)
    note "network-acceptance $NAME rc=$rc (pass, rows, same_instance, instance==I1, tls_verified, forge_row, external_e2e) = $nchk"
    echo "$nchk"; [ "$nchk" = "(True, 60, True, True, True, True, 'NOT_RUN')" ];;
  sites)
    code() { "${NOPROXY[@]}" curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "$1" 2>/dev/null || echo ERR; }
    f=$(code https://forge.example.com/); m=$(code https://llm.example.com/v1/models); hh=$(code https://llm.example.com/health)
    note "LAN view (not external E2E): forge=$f llm/v1/models(no key)=$m llm/health=$hh"; [ "$f:$m:$hh" = "200:401:404" ];;
  units)
    a=$(systemctl --user is-active qwen36-inference); b=$(systemctl --user is-active qwen36-experiment); e=$(systemctl --user is-enabled qwen36-inference 2>/dev/null)
    note "units inference=$a experiment=$b inference_enabled=${e:-none}"; [ "$a" != active ] && [ "$b" != active ] && [ "$e" != enabled ];;
  record)
    note "VERDICT $2 $3 ${4:-}"; echo "$2 $3 ${4:-}" >> "$STEPS";;
  *) echo "usage" >&2; exit 2;;
esac
