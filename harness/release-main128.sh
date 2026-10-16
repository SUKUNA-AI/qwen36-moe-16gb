#!/usr/bin/env bash
# release-main128.sh - install the qualified main128 next candidate in place of f1, f1 kept for rollback.
#
# Owner-run in WSL, from a checkout of the release branch: the same tree sync-v5.sh put into the overlay
# ($QWEN36_NATIVE) for the S5 runs. Site paths come from config/local.env (or --env FILE): QWEN36_ROOT,
# QWEN36_NATIVE (= QWEN36_ROOT/optimization), QWEN36_MODEL_DIR (environment also accepted), QWEN36_ARCHIVE.
# Evidence paths are relative to QWEN36_NATIVE (reports/x.json), as release.EVIDENCE_ROOT.
#
#   release-main128.sh [--dry-run|--check|--apply] --label L --run R --teacher Q --fla-component F \
#       --overlap-component C --ab A [--evidence E]... [--limitation TEXT]... [--limitations-file F.json] \
#       [--long256 carry|rerun|drop] [--long256-run R --long256-teacher Q [--long256-bitexact Q]] \
#       [--long256-evidence E] [--long256-limitation TEXT] [--long256-limitations-file F.json] [--baseline64 carry|drop] \
#       [--freetoken-base DIR | --no-export] [--env FILE] [--python P]
#   release-main128.sh --rollback [--dry-run|--check|--apply] --label L [--manifest M] [--env FILE]
#
# Modes (default --dry-run):
#   --dry-run  prints every step, path and command plus the composed decision; read-only checks
#              (site config, units, overlay vs this checkout, evidence files, libraries). Writes nothing.
#   --check    the whole chain in a temporary directory that is removed afterwards: rendered staging tree,
#              decision, make-qualification receipt, evidence export, release.py preflight (no systemctl,
#              no install). Production files are only read; the service may keep running.
#   --apply    OWNER-APPROVED ONLY (the coordinator passes it). Persistent staging in
#              $QWEN36_NATIVE/release-staging/L (never overwritten), receipt, export, f1 snapshot
#              $QWEN36_ROOT/backups/f1-before-L.tar, then release.py install (both units must be stopped;
#              this script never stops or starts them), then the installed profile-runtime.py gate.
#   --rollback restores f1 from the rollback manifest release.py wrote (recorded in staging install.json, or
#              --manifest); --check validates it only, --apply restores (units stopped), then re-gates f1.
#
# Steps of a release: 1 stage (render serving/harness like sync-v5, copy the runtime tree), 2 decision
# (release-main128.py: main128 from the named reports, long256/baseline64 carried from the installed f1
# receipt or dropped), 3 qualification receipt (make-qualification.py), 4 evidence export
# (export-release-evidence.py against upstream FreeToken cac247a from --freetoken-base, default
# $QWEN36_ARCHIVE/work/FreeToken-opt), 5 f1 snapshot + release.py install, 6 installed gate (prepare()).
# A carried long256 keeps f1 run receipts bound to the f1 runtime bytes: make-qualification refuses them on
# this tree, so long256 needs --long256 rerun (fresh qrun teacher on this tree) or --long256 drop.
# rerun binds the new long256 runs; --long256-bitexact names their 0-diff teacher report when the long256
# teacher itself shows the 262144-context long-120000 deviation of f1 (see release-main128.py).
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
checkout=$(cd -- "$here/.." && pwd)
LABEL_RE='^[a-z0-9][a-z0-9._-]{0,40}$'
FT_BASE_COMMIT=cac247a860e316e06580d05aeb05f2e647bde214
# release.py FILES (validator MANAGED_FILES): rendered from serving/ like sync-v5 renders them for native.
FILES=(guard.py ingress.py request_metrics.py serve_guarded.py readiness.py placement.py keepalive.py
  metrics_summary.py profile-runtime.py profiles.json start-inference.sh)
# export-release-evidence.py OVERLAY entries that are not FILES (exact bytes, not installed).
EXPORT_ONLY=(qwen36-inference.service manage.ps1 supervisor.ps1 watch-supervisor.ps1 host-metrics.ps1 get-api-key.ps1)
# make-qualification.py RUN_SERVICE_FILES: the service files every run receipt binds by sha256.
RUN_SERVICE_FILES=(serve_guarded.py guard.py ingress.py request_metrics.py)
HARNESS_TOOLS=(release.py make-qualification.py export-release-evidence.py repo_paths.py release-main128.py)
UNITS=(qwen36-inference.service qwen36-experiment.service)

die() { local code=$1; shift; printf 'release-main128: REFUSED(%s): %s\n' "$code" "$*" >&2; exit "$code"; }
say() { printf 'release-main128: %s\n' "$*"; }
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

mode=dry-run; action=release; label=; env_file=; python=; manifest=; ft_base=; export_on=1
declare -a DECISION_ARGS=() EVIDENCE=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run|--check|--apply) mode=${1#--};;
    --rollback) action=rollback;;
    --label) [ $# -ge 2 ] || die 2 "--label needs a value"; label=$2; shift;;
    --env) [ $# -ge 2 ] || die 2 "--env needs a file"; env_file=$2; shift;;
    --python) [ $# -ge 2 ] || die 2 "--python needs a path"; python=$2; shift;;
    --manifest) [ $# -ge 2 ] || die 2 "--manifest needs a path"; manifest=$2; shift;;
    --freetoken-base) [ $# -ge 2 ] || die 2 "--freetoken-base needs a directory"; ft_base=$2; shift;;
    --no-export) export_on=0;;
    --run|--teacher|--fla-component|--overlap-component|--ab|--evidence|--long256-run|--long256-teacher|--long256-bitexact|--long256-evidence)
      [ $# -ge 2 ] || die 2 "$1 needs a reports/... path"; DECISION_ARGS+=("$1" "$2"); EVIDENCE+=("$2"); shift;;
    --limitation|--limitations-file|--long256|--baseline64|--long256-limitation|--long256-limitations-file)
      [ $# -ge 2 ] || die 2 "$1 needs a value"; DECISION_ARGS+=("$1" "$2"); shift;;
    -h|--help) usage; exit 0;;
    *) die 2 "unknown argument: $1";;
  esac
  shift
done
[[ $label =~ $LABEL_RE ]] || die 2 "--label is required (want $LABEL_RE)"

# ---------------------------------------------------------------- site configuration (as sync-v5.sh)
env_file=${env_file:-$checkout/config/local.env}
site_value() {  # KEY -> value in $env_file (KEY=v | 'v' | "v"; a trailing ' # comment' is dropped)
  local line v
  [ -f "$env_file" ] || return 0
  line=$(grep -E "^[[:space:]]*$1[[:space:]]*=" -- "$env_file" | tail -n 1) || return 0
  v=${line#*=}; v=${v%%[[:space:]]#*}; v=${v#"${v%%[![:space:]]*}"}; v=${v%"${v##*[![:space:]]}"}
  case "$v" in \'*\') v=${v:1:${#v}-2};; \"*\") v=${v:1:${#v}-2};; esac
  printf '%s' "${v%/}"
}
site_path_ok() { [[ $1 =~ ^/[A-Za-z0-9._/+-]+$ ]] && [[ $1 != *'/home/''USER'* ]] && [[ $1 != *'/path/''to/'* ]]; }
root=$(site_value QWEN36_ROOT); native=$(site_value QWEN36_NATIVE); archive=$(site_value QWEN36_ARCHIVE)
model=$(site_value QWEN36_MODEL_DIR); model=${model:-${QWEN36_MODEL_DIR:-}}; model=${model%/}
public_host=$(site_value QWEN36_PUBLIC_HOST); edge_host=$(site_value QWEN36_EDGE_SSH_HOST)
site_path_ok "$root" && site_path_ok "$native" && site_path_ok "$model" \
  || die 2 "set QWEN36_ROOT, QWEN36_NATIVE and QWEN36_MODEL_DIR (absolute site paths) in $env_file or the environment"
[ "$native" = "$root/optimization" ] || die 2 "QWEN36_NATIVE must be QWEN36_ROOT/optimization (release.EVIDENCE_ROOT)"
[ -z "$archive" ] || site_path_ok "$archive" || die 2 "invalid QWEN36_ARCHIVE in $env_file"
python=${python:-$root/.venv/bin/python}
[ -x "$python" ] || die 66 "python not found: $python (pass --python)"
PY=("$python" -E -s -B)  # -B: loading the staged modules writes no __pycache__
staging_root=$native/release-staging
work=$staging_root/$label
[ -n "$ft_base" ] || ft_base=${archive:+$archive/work/FreeToken-opt}
for tool in sha256sum tar cmp sed systemctl git; do command -v "$tool" >/dev/null 2>&1 || die 66 "required tool missing: $tool"; done

# Placeholder -> site value, longest first (identical to sync-v5.sh render()).
# readiness.py carries the public name and the edge SSH alias (serving/README.md: QWEN36_PUBLIC_HOST, QWEN36_EDGE_SSH_HOST).
RENDER_FROM=('/path/''to/qwen36-workspace/outputs/qwen36-api/optimization' '/path/''to/qwen36-workspace'
  '/home/''USER/qwen36-api' '/path/''to/LLM/qwen36-api/model' 'llm.''example.com' 'EDGE_SSH''_HOST')
RENDER_TO=("$native/config" "$archive" "$root" "$model" "$public_host" "$edge_host")
render() {
  local i; local -a args=()
  for i in "${!RENDER_FROM[@]}"; do [ -z "${RENDER_TO[$i]}" ] || args+=(-e "s|${RENDER_FROM[$i]}|${RENDER_TO[$i]}|g"); done
  sed "${args[@]}" -- "$1"
}

units_report() {
  local u state out=
  for u in "${UNITS[@]}"; do
    state=$(systemctl --user show -p ActiveState --value "$u" 2>/dev/null) || state=unknown
    out+="$u=${state:-unknown} "
  done
  printf '%s' "$out"
}
require_units_stopped() {
  local u state
  for u in "${UNITS[@]}"; do
    state=$(systemctl --user show -p ActiveState --value "$u" 2>/dev/null) || die 73 "cannot query $u (fail-closed)"
    case "$state" in inactive|failed) ;; *) die 73 "$u is '${state:-unknown}': stop it first (Windows: qwen36 stop); this script never stops or starts units";; esac
  done
}

installed_gate() {  # the installed profile-runtime.py prepare(): receipt, bytes, kernels, libraries, checkpoint
  "${PY[@]}" - "$root/profile-runtime.py" <<'PY'
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('qwen36_installed_profile_runtime', sys.argv[1])
runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
action, name, profile = runtime.prepare([])
print(json.dumps({'installed_gate': 'PASS', 'start_profile': name, 'context': profile['context'],
                  'qualified': sorted(n for n, p in runtime.load_installed()[0]['profiles'].items() if p['qualified'] is True)}))
PY
}

# ---------------------------------------------------------------- rollback
if [ "$action" = rollback ]; then
  record=$work/install.json
  if [ -z "$manifest" ]; then
    [ -f "$record" ] || die 66 "no $record (pass --manifest with the path release.py printed)"
    manifest=$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rollback_manifest"])' "$record")
  fi
  tool=$work/harness/release.py
  [ -f "$tool" ] || die 66 "staged release.py missing: $tool"
  say "rollback label=$label mode=$mode"
  say "  manifest=$manifest"
  say "  units: $(units_report)"
  say "  check:   QWEN36_CHECKOUT=$work ${PY[*]} $tool rollback --manifest $manifest --check"
  say "  restore: QWEN36_CHECKOUT=$work ${PY[*]} $tool rollback --manifest $manifest"
  [ -f "$work/install.json" ] && say "  f1 snapshot: $("${PY[@]}" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["f1_snapshot"], r["f1_snapshot_sha256"])' "$work/install.json")"
  case "$mode" in
    dry-run) say "dry-run: nothing written"; exit 0;;
    check) QWEN36_CHECKOUT=$work "${PY[@]}" "$tool" rollback --manifest "$manifest" --check; exit 0;;
  esac
  require_units_stopped
  QWEN36_CHECKOUT=$work "${PY[@]}" "$tool" rollback --manifest "$manifest" --check
  QWEN36_CHECKOUT=$work "${PY[@]}" "$tool" rollback --manifest "$manifest"
  installed_gate
  say "rollback done; f1 restored and gated. Start the service yourself (qwen36 start) and check its log."
  exit 0
fi

# ---------------------------------------------------------------- read-only preflight (every mode)
[ "${#EVIDENCE[@]}" -gt 0 ] || die 2 "name the S5 evidence (--run, --teacher, --fla-component, --overlap-component, --ab)"
[ -f "$root/release/qualification.json" ] || die 66 "installed f1 receipt missing: $root/release/qualification.json"
[ -d "$native/python/freetoken" ] || die 66 "overlay runtime tree missing: $native/python/freetoken"
for f in "${FILES[@]}" "${EXPORT_ONLY[@]}"; do [ -f "$checkout/serving/$f" ] || die 66 "serving/$f missing in this checkout"; done
for f in "${HARNESS_TOOLS[@]}" qwen36.cmd; do [ -f "$here/$f" ] || die 66 "harness/$f missing in this checkout"; done
# The S5 runs served through the overlay's rendered service files (run receipts bind RUN_SERVICE_FILES, as
# make-qualification checks): this checkout must render them to the same bytes. Other serving files ship from
# this checkout; an overlay difference there is only reported.
drift=(); notes=()
for f in "${FILES[@]}"; do
  case "$f" in *.py) ;; *) continue;; esac
  if [ -f "$native/$f" ] && ! cmp -s <(render "$checkout/serving/$f") "$native/$f"; then
    case " ${RUN_SERVICE_FILES[*]} " in *" $f "*) drift+=("$f");; *) notes+=("$f");; esac
  fi
done
[ "${#drift[@]}" -eq 0 ] || die 65 "run-bound service files differ from the overlay (rendered): ${drift[*]}; release the tree the S5 runs used"
for f in "${FILES[@]}"; do
  if render "$checkout/serving/$f" | grep -qE '/home/''USER|/path/''to/|EDGE_SSH''_HOST|llm\.''example\.com'; then die 65 "unrendered placeholder left in serving/$f"; fi
done
say "release label=$label mode=$mode"
say "  checkout=$checkout"
say "  root=$root native=$native model=$model"
say "  python=$python"
say "  units: $(units_report)"
[ "${#notes[@]}" -eq 0 ] || say "  note: overlay copies differ from this checkout (shipped from the checkout, not run-bound): ${notes[*]}"
say "  staging=$( [ "$mode" = apply ] && printf '%s' "$work" || printf '%s' '<temporary, removed>')"
say "  export: $( [ "$export_on" = 1 ] && printf 'upstream %s at %s' "${ft_base:-<unset>}" "$FT_BASE_COMMIT" || printf 'skipped (--no-export)')"
[ "$mode" != apply ] || [ ! -e "$work" ] || die 73 "staging $work exists; labels are never reused"
if [ "$mode" = apply ]; then require_units_stopped; fi

if [ "$mode" = dry-run ]; then
  say "composed decision (validator: this checkout's serving/profile-runtime.py):"
  "${PY[@]}" "$here/release-main128.py" decision --validator "$checkout/serving/profile-runtime.py" \
    --profiles-in "$checkout/serving/profiles.json" --f1-receipt "$root/release/qualification.json" \
    --evidence-root "$native" "${DECISION_ARGS[@]}"
  say "plan:"
  say "  1 stage      $work/{serving,harness,freetoken/python/freetoken} (rendered like sync-v5; runtime tree copied)"
  say "  2 decision   release-main128.py decision ... --profiles-out $work/serving/profiles.json --decision-out $work/decision.json"
  say "  3 receipt    QWEN36_CHECKOUT=$work ${PY[*]} $work/harness/make-qualification.py --decision $work/decision.json --output $work/serving/qualification.json"
  [ "$export_on" = 1 ] && say "  4 export     upstream clone at $FT_BASE_COMMIT + this runtime tree -> $work/harness/evidence-exports/$label"
  say "  5 install    units stopped; tar f1 -> $root/backups/f1-before-$label.tar; release.py install --check; release.py install"
  say "  6 gate       installed profile-runtime.py prepare() (no start); rollback: $0 --rollback --apply --label $label"
  say "dry-run: nothing written"
  exit 0
fi

# ---------------------------------------------------------------- staging (check: temporary, apply: persistent)
if [ "$mode" = check ]; then
  work=$(mktemp -d "${TMPDIR:-/tmp}/qwen36-release-check.XXXXXX")
  trap 'rm -rf -- "$work"' EXIT
else
  mkdir -p -m 700 -- "$staging_root"
  mkdir -m 700 -- "$work"
fi
mkdir -p -m 700 -- "$work/serving" "$work/harness" "$work/freetoken/python"
for f in "${FILES[@]}"; do
  [ "$f" = profiles.json ] && continue
  render "$checkout/serving/$f" > "$work/serving/$f"
done
for f in "${EXPORT_ONLY[@]}"; do cp -- "$checkout/serving/$f" "$work/serving/$f"; done
for f in "${HARNESS_TOOLS[@]}"; do render "$here/$f" > "$work/harness/$f"; done
cp -- "$here/qwen36.cmd" "$work/harness/qwen36.cmd"
cp -R -- "$checkout/freetoken/python/freetoken" "$work/freetoken/python/freetoken"
find "$work/freetoken" -name __pycache__ -type d -prune -exec rm -rf -- {} +
say "1 staged: $work"

"${PY[@]}" "$work/harness/release-main128.py" decision --validator "$work/serving/profile-runtime.py" \
  --profiles-in "$checkout/serving/profiles.json" --f1-receipt "$root/release/qualification.json" \
  --evidence-root "$native" "${DECISION_ARGS[@]}" \
  --profiles-out "$work/serving/profiles.json" --decision-out "$work/decision.json"
say "2 decision: $work/decision.json"

QWEN36_CHECKOUT=$work "${PY[@]}" "$work/harness/make-qualification.py" \
  --decision "$work/decision.json" --output "$work/serving/qualification.json"
say "3 receipt: $work/serving/qualification.json sha256 $(sha256sum "$work/serving/qualification.json" | cut -c1-64)"

if [ "$export_on" = 1 ]; then
  [ -n "$ft_base" ] && [ -d "$ft_base" ] || die 66 "upstream FreeToken clone missing (--freetoken-base DIR or --no-export)"
  clone=$work/export/FreeToken-opt
  mkdir -p -m 700 -- "$work/export"
  git clone --quiet --no-checkout -- "$ft_base" "$clone"
  git -C "$clone" -c advice.detachedHead=false checkout --quiet "$FT_BASE_COMMIT"
  "${PY[@]}" "$work/harness/release-main128.py" export-tree --source "$checkout/freetoken" --clone "$clone"
  declare -a RESULTS=()
  for e in "${EVIDENCE[@]}"; do case "$e" in reports/*.json) RESULTS+=(--result "native:$e");; esac; done
  QWEN36_CHECKOUT=$work QWEN36_FREETOKEN_SOURCE=$clone "${PY[@]}" "$work/harness/export-release-evidence.py" \
    --name "$label" --native-root "$native" "${RESULTS[@]}" > "$work/export.json"
  say "4 export: $work/harness/evidence-exports/$label"
fi

if [ "$mode" = check ]; then
  QWEN36_CHECKOUT=$work "${PY[@]}" - "$work/harness" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[1])
import release
profiles, sources, stale, qualified, expected, start, provenance = release.preflight()
print(json.dumps({'preflight': 'PASS', 'files': len(sources), 'qualified_profiles': qualified, 'start_profile': start,
                  'stale_files_removed': len(stale), 'provenance': release.provenance_summary(provenance)}))
PY
  say "check: chain PASS in a temporary staging tree (removed); nothing installed"
  exit 0
fi

# ---------------------------------------------------------------- apply: f1 snapshot, install, gate
require_units_stopped
snapshot=$root/backups/f1-before-$label.tar
[ ! -e "$snapshot" ] || die 73 "$snapshot exists"
mkdir -p -m 700 -- "$root/backups"
declare -a KEEP=()
for f in "${FILES[@]}" release; do [ -e "$root/$f" ] && KEEP+=("$f"); done
tar -C "$root" -cf "$snapshot" -- "${KEEP[@]}"
chmod 600 -- "$snapshot"
snapshot_sha=$(sha256sum "$snapshot" | cut -c1-64)
say "5 f1 snapshot: $snapshot sha256 $snapshot_sha"
QWEN36_CHECKOUT=$work "${PY[@]}" "$work/harness/release.py" install --check
QWEN36_CHECKOUT=$work "${PY[@]}" "$work/harness/release.py" install | tee "$work/install-output.json"
"${PY[@]}" - "$work/install-output.json" "$work/install.json" "$label" "$snapshot" "$snapshot_sha" \
  "$root/backups" "$work/serving/qualification.json" <<'PY'
import hashlib, json, os, pathlib, sys
output, record, label, snapshot, snapshot_sha, backups, receipt = sys.argv[1:]
result = json.loads(pathlib.Path(output).read_text().strip().splitlines()[-1])
manifest = pathlib.Path(result['rollback_manifest']).resolve()
if not manifest.is_relative_to(pathlib.Path(backups).resolve()):
    raise SystemExit('rollback manifest outside the service backups')
data = json.dumps({'label': label, 'rollback_manifest': str(manifest), 'f1_snapshot': snapshot,
                   'f1_snapshot_sha256': snapshot_sha, 'qualification_sha256':
                   hashlib.sha256(pathlib.Path(receipt).read_bytes()).hexdigest(),
                   'qualified_profiles': result['qualified_profiles'], 'installed_files': result['installed_files']},
                  indent=2) + '\n'
fd = os.open(record, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as out:
    out.write(data)
print(data, end='')
PY
installed_gate
say "6 installed and gated. Rollback: $0 --rollback --apply --label $label"
say "Start the service yourself (qwen36 start), then check its log (harness/OPERATIONS.md, main128 next)."
