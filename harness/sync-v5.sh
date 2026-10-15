#!/usr/bin/env bash
# sync-v5.sh - fail-closed, allowlisted sync of this repository checkout into the
# native experiment overlay (QWEN36_NATIVE from config/local.env).
#
# Site paths come from config/local.env (or --env FILE): QWEN36_NATIVE is the overlay,
# QWEN36_ROOT/run/gpu-exclusive.lock the GPU lock. The workspace is the checkout that
# contains this script (harness/sync-v5.sh); QWEN36_WORKSPACE only when it runs from
# elsewhere. Sources are repo-relative: freetoken/python/freetoken/** -> python/freetoken/**,
# serving/*.py -> native root, harness/* -> native root, harness/*.json -> config/.
# Tracked py/sh/config files carry the sanitized placeholders; they are rendered with the
# site values on the way to native (and frozen as rendered bytes):
#   /path/to/qwen36-workspace/outputs/qwen36-api/optimization -> QWEN36_NATIVE/config
#   /path/to/qwen36-workspace -> QWEN36_ARCHIVE      /home/USER/qwen36-api -> QWEN36_ROOT
#   /path/to/LLM/qwen36-api/model -> QWEN36_MODEL_DIR (local.env or environment)
# A placeholder whose value is not configured is a blocker for the file that contains it.
#
# Replaces the blind `sync.sh` for frozen runs (V5 tool pilot, final3-v5 matrix,
# qualifications, long256-final/fallback64, Nsight profile-final-*). It never
# starts/stops units, never touches reports/ (except writing its own new
# manifest), secrets/, private-* (except the LIBS entries), run/ (only flock on the EXISTING GPU lock in
# --apply; it never creates it), logs/, backups/, build/, tools/,
# config/experiment.env, config/teacher-fixture.json or native-only
# config/profile-final-*-ncu<k>.json, and never deletes native files
# (native-only build products such as kernel/_cpu_moe*.so stay).
#
# Modes (exactly one):
#   --print-freeze [--skip-runtime] [--extra-config N]...
#                                  print "sha256  repo-relative-path" for every allowlisted
#                                  source, as rendered for native (reads workspace only)
#   --dry-run [--freeze F] [--label L] [--skip-runtime] [--extra-config N]...
#                                  plan + hash diff; NO writes anywhere
#   --apply --label L --freeze F [--skip-runtime] [--extra-config N]...
#                                  verify against freeze F (before touching the lock),
#                                  stage inside native, keep previous native bytes in
#                                  sync-v5-rollback/L/, replace atomically, write
#                                  reports/sync-v5-L.json and reports/sync-v5-L.sha256
#                                  (never overwritten)
#   --verify --label L             read-only drift check of native AND workspace
#                                  against manifest L
#   --skip-runtime                 leave python/freetoken untouched (runtime drift is still
#                                  reported); use when the runtime tree is not part of the
#                                  frozen decision (e.g. CUDA agents editing freetoken/)
#   --env FILE                     site config instead of <checkout>/config/local.env
#   --extra-config N               also sync harness/N (N = name.json, repeatable): an
#                                  experiment config, or a profile-final-*.json profile config;
#                                  schema-checked like the built-in groups. Pass the SAME set to
#                                  --print-freeze, --dry-run and --apply.
#
# Config groups: CONFIG_JSON (required, explicit), CONFIG_PATTERNS (explicit name patterns
# for generated configs: long256-final-*, bf256-final*, baseline64-final*, main-final*,
# teacher-*; every match must pass the experiment schema), PROFILE_PATTERNS
# (profile-final-*.json, profile.sh name rule + nsys block schema).
#
# GPU lock: --apply takes the same flock as matrix/pilot. Run it directly (preferred); under
# a `flock -n LOCK` wrapper it uses the inherited descriptor (lock held by the caller).
# --dry-run only reads /proc/locks (busy -> 75 unless the holder is the calling wrapper).
#
# Native private libraries (V6): LIBS lists "repo-relative source|private-libraries/<name>.so". The
# bytes are bound ONLY by the --freeze file (no sha256 is hardcoded here: a rebuilt library gets
# a new freeze); --apply refuses (65) if the library differs from the freeze, a missing library
# is a blocker (66). Installed with mode 0600 into a 0700 private-libraries/. No other private-* write.
# Runtime tree: without --skip-runtime every python/freetoken source (TREE_SUFFIX_RE) is synced
# and frozen like any other file; native-only build products (*.so) are preserved, never rebuilt.
#
# Exit codes: 0 ok; 2 usage; 65 freeze/verify mismatch; 66 invalid/missing source;
#             67 destination conflict; 70 failure during/after replacement;
#             73 model unit or overlay process active; 75 GPU-exclusive lock busy/missing.
#
# Test hook: SYNC_V5_SANDBOX=<dir named sync-v5-sandbox*> redirects native,
# workspace and lock to <dir>/native, <dir>/workspace, <dir>/gpu-exclusive.lock.
# It is refused for any path inside QWEN36_ROOT; the site config is optional there.
set -euo pipefail
umask 077
export LC_ALL=C
shopt -s nullglob

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Repo-relative source roots (MANIFEST layout).
RT_REL=freetoken/python/freetoken
SV_REL=serving
HS_REL=harness
# Pre-relocation workspace prefix; split literal so this script never matches its own scan.
OLD_WORKSPACE='/mnt/c/Users/USER/Documents/Codex/''2026-10-13'
LABEL_RE='^[a-z0-9][a-z0-9.-]{0,60}$'
PATH_RE='^[A-Za-z0-9._,=+/-]+$'
TREE_SUFFIX_RE='\.(py|cu|cuh|h|hpp|cpp|c|json|md|service|txt|toml|yaml|yml)$'
SOURCE_SUFFIX_RE='\.(py|cu|cuh|h|cpp)$'   # suffixes hashed by experiment.py / bench.py
CONFIG_KEYS='["context","chunk","memory_ratio","flags","env","kv_dtype","qualified"]'
PROFILE_KEYS='["context","chunk","memory_ratio","flags","env","kv_dtype","qualified","nsys","profile_plan","ncu"]'
CONFIG_NAME_RE='^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.json$'
PROFILE_NAME_RE='^profile-final-[a-z0-9][a-z0-9-]{0,63}\.json$'   # profile.sh: ^profile-final-[a-z0-9][a-z0-9-]{0,63}$
# experiment profile (experiment.py / start-experiment.sh)
EXP_FILTER='(type=="object") and ((keys-$allowed)|length==0)
  and (.context|type=="number") and (.chunk|type=="number")
  and ((.env // {})|type=="object") and ((.flags // [])|type=="array")'
# profile-final-* (mirrors the static part of profile.sh validation; ncu details stay with profile.sh)
PROFILE_FILTER='(type=="object") and ((keys-$allowed)|length==0)
  and (.context|type=="number") and (.chunk|type=="number")
  and ((.env // {})|type=="object") and ((.flags // [])|type=="array")
  and ((.env.QWEN36_NSYS_FORWARDS // "")|tostring|test("^0*[1-9][0-9]{0,4}$"))
  and (.nsys|type=="object")
  and ((.nsys.output_prefix // "")|tostring|test("^(model|ncu)-final-[a-z0-9][a-z0-9-]{0,80}$"))
  and ((.nsys.tool // "nsys") as $t | ($t=="nsys") or ($t=="ncu" and (.ncu|type=="object")))
  and ((.profile_plan // {})|type=="object")'

# ---------------------------------------------------------------- allowlist
# Python from serving/ (service files + their imports, release/ops) -> native root.
ROOT_PY=(guard.py ingress.py request_metrics.py serve_guarded.py placement.py
  readiness.py keepalive.py metrics_summary.py profile-runtime.py
  test-guard.py test-metrics.py)
# Python from harness/ -> native root.
OPT_PY=(
  # service entry (bench.SERVICE_FILES)
  experiment.py
  # paired harness V6 (bench.HARNESS_FILES) + its CPU contract test (harness-v6 runbook)
  bench.py matrix.py compare-ab.py tool-task-bench.py tool_fixture_v5.py tool-pilot.py
  test-paired-contract-cpu.py
  # chunk screening for the bit-exact CUDA MoE profiles + R1 logit-reference comparison (harness-v6 runbook)
  screen-chunk.py compare-logit-reference.py
  # api-guard v2 CPU suites (CPU-checks.sh)
  test-tool-choice-guard-cpu.py test-tool-semantics-guard-cpu.py
  # experiment gates / qualification (native entry points and their imports)
  check-experiment.py quality.py fixtures.py compare-quality.py qualify.py boundary.py
  test-output-budget.py test-backend-v2.py test-resilience.py test-prefill-cancel.py
  prefix-quality.py run-prefix-gates.py diagnose-facts.py functional-matrix.py
  isolate-quality.py lookup-qualification.py check-candidate-metrics.py
  # Nsight Systems/Compute flow (final-nsight-runbook.md: plan/run/ncu-config, analysis in native)
  profile-request.py analyze-model-nsys.py test-analyze-model-nsys-cpu.py
  # soak
  soak.py soak-analysis.py test-soak-cpu.py
  # release / ops
  release.py export-release-evidence.py make-qualification.py inspect-keepalive.py
  network-acceptance.py
  # component checks (component-checks.sh, CPU-checks.sh)
  test-kv.py test-spec-gpu.py test-mtp.py test-complement.py test-log-failure.py)
# Shell from harness/ -> native root (all callers use `bash X`).
SH_FILES=(start-experiment.sh run-experiment.sh profile.sh run-live-gates.sh
  run-context-gates.sh component-checks.sh build-complement.sh link-cublas-headers.sh
  profile-gdn-snapshot.sh CPU-checks.sh sync.sh sync-v5.sh)
# Experiment profile configs from harness/ (schema-checked, all required) -> native config/.
CONFIG_JSON=(
  # S6 long256-final / fallback64 (long256-plan, 2026-10-13 19:00Z)
  baseline64-final.json bf256-final.json long256-final-candidate.json long256-final-retain.json
  long256-final-retain2048.json teacher-baseline64-final.json
  # earlier experiment / qualification configs
  b12x-fixed.json b12x128.json baseline64.json baseline64-candidate.json
  baseline64-snapshot-off-ablation.json bf-prefix128.json bf128.json bf256.json
  calibration64.json chunk1024.json chunk4096.json chunk8192.json ckpt128.json
  complement128.json cuda-both128.json cuda-complement128.json cuda-long256.json
  cuda-matched-no-prefix128.json cuda-matched128.json cuda-prefix128.json cuda128.json
  cuda2048.json cuda4096-safe.json cuda8192-safe.json d2d128.json fixed128.json
  fp128.json fp256.json frequency128.json long256-candidate.json lookup128.json
  main-candidate.json main-retain.json main-retain2048.json mtp128.json
  persistent128.json profile-cuda128.json profile-main128.json profile64.json
  teacher-b12x-fixed.json teacher-bf128.json teacher-chunk8192.json
  teacher-cuda-matched128.json teacher-cuda-prefix128.json teacher-cuda128.json
  teacher-cuda4096-safe.json teacher-fp128.json teacher-main-candidate.json
  teacher-main-retain.json teacher-main-retain2048.json tile128.json triton4096-safe.json
  # harness V6: bit-exact CUDA NVFP4 MoE prefill candidates (final4-r1 arms + chunk screening)
  cuda-bitexact128-2048.json cuda-bitexact128.json cuda-bitexact128-c2048.json cuda-bitexact128-c6144.json cuda-bitexact128-c8192.json cuda-bitexact128-c16384.json teacher-cuda-bitexact128-2048.json teacher-cuda-bitexact128.json)
# Generated experiment configs, matched by explicit name patterns (optional; every match must
# pass the experiment schema, otherwise it is a blocker): ratio variants long256-final-<V>-rNN /
# bf256-final-rNN (long256-fallback64-plan L1), main-final (S5), teacher-<candidate> (qualify.py),
# main-release* / long256-release* (next-release candidates and slot steps).
CONFIG_PATTERNS=('long256-final-*.json' 'bf256-final*.json' 'baseline64-final*.json'
  'main-final*.json' 'teacher-*.json' 'main-release*.json' 'long256-release*.json')
# Nsight profile configs (profile-request.py plan --write) -> native config/, profile schema.
PROFILE_PATTERNS=('profile-final-*.json')
# Native libraries: "repo-relative source|native destination" (group lib; sha256 from --freeze). The pinned
# libraries already live in native private-*/ and the repository ships no .so: a rebuilt library (e.g.
# 'kernels/nvfp4-moe/build/libqwen_nvfp4_bitexact.so|private-libraries/libqwen_nvfp4_bitexact.so') is added
# here explicitly, with a new freeze.
LIBS=()
LIB_DST_RE='^private-libraries/[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.so$'
declare -A LIB_SRC=()
# Destinations that must never be written, whatever the allowlist says. profile-final-*-ncu<k>
# configs are written by `profile-request.py ncu-config` only in native (private kernel symbols).
DENY_DST_RE='^(reports|logs|backups|run|secrets|build|tools|sections|include|cuda-transport|harness-backups|private-[^/]*|sync-v5-rollback|\.sync-v5-stage[^/]*)(/|$)|^config/(experiment\.env|teacher-fixture\.json)$|^config/profile-final-[a-z0-9-]*-ncu[0-9]+\.json$'

die() { local code=$1; shift; printf 'sync-v5: REFUSED(%s): %s\n' "$code" "$*" >&2; exit "$code"; }
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

# ---------------------------------------------------------------- arguments
mode=; label=; freeze=; skip_runtime=0; env_file=
declare -a EXTRA_CONFIG=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run|--apply|--print-freeze|--verify)
      [ -z "$mode" ] || die 2 "exactly one mode is allowed"; mode=${1#--};;
    --label) [ $# -ge 2 ] || die 2 "--label needs a value"; label=$2; shift;;
    --freeze) [ $# -ge 2 ] || die 2 "--freeze needs a file"; freeze=$2; shift;;
    --env) [ $# -ge 2 ] || die 2 "--env needs a file"; env_file=$2; shift;;
    --skip-runtime) skip_runtime=1;;
    --extra-config) [ $# -ge 2 ] || die 2 "--extra-config needs a name.json"
      [[ $2 =~ $CONFIG_NAME_RE ]] || die 2 "invalid --extra-config name (want $CONFIG_NAME_RE, no directories)"
      EXTRA_CONFIG+=("$2"); shift;;
    -h|--help) usage; exit 0;;
    *) die 2 "unknown argument: $1";;
  esac
  shift
done
[ -n "$mode" ] || { usage >&2; exit 2; }
if [ -n "$label" ] && ! [[ $label =~ $LABEL_RE ]]; then die 2 "invalid --label (want $LABEL_RE)"; fi
case "$mode" in
  apply) [ -n "$label" ] || die 2 "--apply requires --label"
         [ -n "$freeze" ] || die 2 "--apply requires --freeze (see --print-freeze)";;
  verify) [ -n "$label" ] || die 2 "--verify requires --label"
          [ "${#EXTRA_CONFIG[@]}" -eq 0 ] && [ -z "$freeze" ] && [ "$skip_runtime" = 0 ] \
            || die 2 "--verify takes only --label (the manifest defines the file set)";;
esac
for tool in sha256sum jq find sort flock stat realpath pgrep systemctl readlink awk; do
  command -v "$tool" >/dev/null 2>&1 || die 66 "required tool missing: $tool"
done

site_value() {  # KEY -> value in $env_file (KEY=v | 'v' | "v"; a trailing ' # comment' is dropped)
  local line v
  [ -f "$env_file" ] || return 0
  line=$(grep -E "^[[:space:]]*$1[[:space:]]*=" -- "$env_file" | tail -n 1) || return 0
  v=${line#*=}; v=${v%%[[:space:]]#*}; v=${v#"${v%%[![:space:]]*}"}; v=${v%"${v##*[![:space:]]}"}
  case "$v" in \'*\') v=${v:1:${#v}-2};; \"*\") v=${v:1:${#v}-2};; esac
  printf '%s' "${v%/}"
}
# Placeholder literals are split so that rendering this script's own synced copy leaves them intact.
site_path_ok() { [[ $1 =~ ^/ ]] && [[ $1 =~ $PATH_RE ]] && [[ $1 != *'/home/''USER'* ]] && [[ $1 != *'/path/''to/'* ]]; }
repo=
if [ -d "$here/../$RT_REL" ] && [ -f "$here/../$HS_REL/sync-v5.sh" ]; then repo=$(cd -- "$here/.." && pwd); fi
env_file=${env_file:-${repo:+$repo/config/local.env}}
site_root=$(site_value QWEN36_ROOT); site_native=$(site_value QWEN36_NATIVE); site_archive=$(site_value QWEN36_ARCHIVE)
site_model=$(site_value QWEN36_MODEL_DIR); site_model=${site_model:-${QWEN36_MODEL_DIR:-}}; site_model=${site_model%/}
workspace=${repo:-$(site_value QWEN36_WORKSPACE)}
native=$site_native; lock=$site_root/run/gpu-exclusive.lock; sandbox=0
if [ -n "${SYNC_V5_SANDBOX:-}" ]; then
  sb=$(realpath -m -- "$SYNC_V5_SANDBOX")
  case "${sb##*/}" in sync-v5-sandbox*) ;; *) die 2 "SYNC_V5_SANDBOX must be a dir named sync-v5-sandbox*";; esac
  if [ -n "$site_root" ]; then case "$sb/" in "$site_root"/*) die 2 "SYNC_V5_SANDBOX must not be inside $site_root";; esac; fi
  [ -d "$sb/native" ] && [ -d "$sb/workspace" ] || die 2 "sandbox needs native/ and workspace/"
  native=$sb/native; workspace=$sb/workspace; lock=$sb/gpu-exclusive.lock; sandbox=1
else
  site_path_ok "$site_root" && site_path_ok "$site_native" \
    || die 2 "set QWEN36_ROOT and QWEN36_NATIVE (absolute site paths, not placeholders) in ${env_file:-config/local.env} or pass --env"
  [ -n "$workspace" ] && [ -d "$workspace/$RT_REL" ] || die 2 "workspace checkout not found (run harness/sync-v5.sh from a checkout or set QWEN36_WORKSPACE)"
fi
for v in "$site_archive" "$site_model"; do
  [ -z "$v" ] || site_path_ok "$v" || die 2 "invalid site path in ${env_file:-environment}: $v"
done
src_tree=$workspace/$RT_REL
out_dir=$workspace/$SV_REL
opt_dir=$workspace/$HS_REL
dst_tree=$native/python/freetoken
reports=$native/reports
snapshot=$workspace/local/notes/serving/HANDOFF-SNAPSHOT.json
# Placeholder -> site value, longest first; applied to py/sh/config/profile entries only.
RENDER_FROM=('/path/''to/qwen36-workspace/outputs/qwen36-api/optimization' '/path/''to/qwen36-workspace'
  '/home/''USER/qwen36-api' '/path/''to/LLM/qwen36-api/model')
RENDER_TO=("$native/config" "$site_archive" "$site_root" "$site_model")
RENDER_KEY=('QWEN36_NATIVE' 'QWEN36_ARCHIVE' 'QWEN36_ROOT' 'QWEN36_MODEL_DIR')
render() {  # stdout: file $1 with every configured placeholder replaced
  local i; local -a args=()
  for i in "${!RENDER_FROM[@]}"; do [ -z "${RENDER_TO[$i]}" ] || args+=(-e "s|${RENDER_FROM[$i]}|${RENDER_TO[$i]}|g"); done
  if [ "${#args[@]}" -eq 0 ]; then cat -- "$1"; else sed "${args[@]}" -- "$1"; fi
}
render_digest() {  # identifies the site values a freeze was rendered with, without printing them
  local i; for i in "${!RENDER_FROM[@]}"; do printf '%s=%s\n' "${RENDER_KEY[$i]}" "${RENDER_TO[$i]}"; done | sha256sum | cut -c1-16
}
rendered_group() { case "$1" in py|sh|config|profile) return 0;; esac; return 1; }

# ---------------------------------------------------------------- safety checks
unit_report=
check_units() {
  local u state lines unit load active rest
  unit_report=
  for u in qwen36-inference.service qwen36-experiment.service; do
    state=$(systemctl --user show -p ActiveState --value "$u" 2>/dev/null) || die 73 "cannot query $u (fail-closed)"
    case "$state" in inactive|failed) ;; *) die 73 "$u is '${state:-unknown}'";; esac
    unit_report+="$u=$state "
  done
  lines=$(systemctl --user list-units --all --no-legend --plain 'qwen36*' 2>/dev/null) || die 73 "cannot list qwen36* units (fail-closed)"
  while read -r unit load active rest; do
    [ -n "$unit" ] || continue
    case "$active" in inactive|failed) ;; *) die 73 "$unit is '$active'";; esac
  done < <(printf '%s\n' "$lines")
}
check_procs() {  # overlay drivers by absolute path OR started relative from cwd=native (e.g. `python matrix.py`)
  local pid cwd args hits= overlay=${site_native:-$native}
  while read -r pid; do
    [ -n "$pid" ] && [ "$pid" != "$$" ] || continue
    cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || true)
    args=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    case "$args" in *"$overlay/"*) hits+="$pid:${args:0:100}; "; continue;; esac
    if [ "$cwd" = "$overlay" ]; then hits+="$pid:${args:0:100}; "; fi
  done < <(pgrep -u "$(id -u)" -f '(experiment\.py|matrix\.py|tool-pilot\.py|tool-task-bench\.py|qualify\.py|quality\.py|compare-ab\.py|soak\.py|lookup-qualification\.py|functional-matrix\.py|isolate-quality\.py|run-prefix-gates\.py|profile-request\.py|run-experiment\.sh|start-experiment\.sh|run-live-gates\.sh|run-context-gates\.sh|profile\.sh)' || true)
  [ -z "$hits" ] || die 73 "native overlay process running: $hits"
}
lock_holders() {  # read-only: PIDs holding (not waiting for) a lock on $lock, from /proc/locks
  local ino maj min key
  read -r ino maj min < <(stat -L -c '%i %Hd %Ld' -- "$lock")
  key=$(printf '%02x:%02x:%s' "$maj" "$min" "$ino")
  awk -v k="$key" '$2 != "->" && $6 == k {print $5}' /proc/locks
}
inherited_lock_fd() {  # fd number of this process that refers to the lock file (flock(1) wrapper), if any
  local real fdp
  real=$(realpath -e -- "$lock" 2>/dev/null) || return 1
  for fdp in /proc/$$/fd/*; do
    [ "$(readlink -- "$fdp" 2>/dev/null)" = "$real" ] && { printf '%s\n' "${fdp##*/}"; return 0; }
  done
  return 1
}
lock_state() {  # read-only: /proc/*/fd + fdinfo "lock:" lines + /proc/locks; never opens or locks the lock file
  # /proc/locks hides a flock whose taker exited (e.g. `exec 9<L; flock -n 9; exec job`), so the
  # descriptors that hold it are found via fdinfo. Best effort for other users' processes;
  # --apply always relies on a real flock -n.
  local real pat f p fd mine=0 other=0 holders
  [ -e "$lock" ] || { echo ABSENT; return 0; }
  real=$(realpath -e -- "$lock") || { echo HELD; return 0; }
  pat=$(printf '%s' "$real" | sed 's/[][*?\\]/\\&/g')
  while IFS= read -r f; do
    p=${f#/proc/}; p=${p%%/*}; fd=${f##*/}
    grep -q '^lock:.*FLOCK' "/proc/$p/fdinfo/$fd" 2>/dev/null || continue
    if [ "$p" = "$$" ] && grep -Eq '^lock:.*FLOCK +ADVISORY +WRITE' "/proc/$p/fdinfo/$fd" 2>/dev/null; then mine=1
    else other=$((other+1)); fi
  done < <(find /proc/[0-9]*/fd -maxdepth 1 -lname "$pat" 2>/dev/null)
  if [ "$mine" = 1 ]; then  # an exclusive flock excludes every other open file description
    echo "HELD-BY-CALLER (exclusive flock held by this process's inherited descriptor)"; return 0
  fi
  holders=$(lock_holders)
  if [ "$other" -gt 0 ] || [ -n "$holders" ]; then echo HELD; return 0; fi
  echo FREE
}
LOCKFD=; LOCK_HOW=
take_lock() {  # --apply only: same flock the V5 matrix/pilot hold for their whole run
  local fd
  [ -d "${lock%/*}" ] || die 75 "lock directory ${lock%/*} missing (fail-closed)"
  [ -f "$lock" ] && [ ! -L "$lock" ] || die 75 "GPU-exclusive lock file $lock missing or not a regular file (never created here)"
  if fd=$(inherited_lock_fd); then
    # flock(1) wrapper: the caller's open file description already holds (or may take) the lock
    flock -n "$fd" || die 75 "GPU-exclusive lock $lock is held by another holder (inherited fd $fd)"
    LOCKFD=$fd; LOCK_HOW="held via inherited fd $fd (caller's flock wrapper)"
    return 0
  fi
  exec {LOCKFD}<"$lock" || die 75 "cannot open $lock"
  flock -n "$LOCKFD" || die 75 "GPU-exclusive lock $lock is held (matrix/pilot/GPU job running)"
  LOCK_HOW="acquired (flock -n fd $LOCKFD)"
}

# ---------------------------------------------------------------- inventory
declare -a E_GROUP=() E_SRC=() E_DST=()
declare -A DST_SEEN=() SRC_SEEN=() SRC_GROUP=() WS_SHA=() NAT_SHA=() NAT_MODE=() FREEZE=()
declare -a BLOCKERS=() RT_ALL=()
add_entry() {
  local g=$1 s=$2 d=$3
  [[ $d =~ $DENY_DST_RE ]] && { BLOCKERS+=("denied destination: $d"); return; }
  [ -z "${DST_SEEN[$d]+x}" ] || { BLOCKERS+=("duplicate destination: $d"); return; }
  DST_SEEN[$d]=1; SRC_SEEN[$s]=1; SRC_GROUP[$s]=$g
  E_GROUP+=("$g"); E_SRC+=("$s"); E_DST+=("$d")
  rendered_group "$g" && [ -f "$workspace/$s" ] || return 0
  local i
  for i in "${!RENDER_FROM[@]}"; do
    [ -z "${RENDER_TO[$i]}" ] && grep -qF -- "${RENDER_FROM[$i]}" "$workspace/$s" \
      && BLOCKERS+=("placeholder ${RENDER_FROM[$i]} in $s needs ${RENDER_KEY[$i]} (config/local.env or environment)")
  done
  return 0
}
build_entries() {
  local f rel n
  [ -d "$src_tree" ] || die 66 "runtime tree missing: $src_tree"
  while IFS= read -r -d '' f; do
    rel=${f#"$src_tree"/}
    case "$rel" in *.pyc) continue;; esac
    if [ -L "$f" ] || [ ! -f "$f" ]; then BLOCKERS+=("runtime tree non-regular file: $rel"); continue; fi
    [[ $rel =~ $PATH_RE ]] || { BLOCKERS+=("runtime tree unsafe name: $rel"); continue; }
    [[ $rel =~ $TREE_SUFFIX_RE ]] || { BLOCKERS+=("runtime tree unexpected suffix: $rel"); continue; }
    RT_ALL+=("$rel")
    [ "$skip_runtime" = 1 ] && continue
    add_entry runtime "$RT_REL/$rel" "python/freetoken/$rel"
  done < <(find "$src_tree" -name __pycache__ -prune -o ! -type d -print0 | sort -z)
  for n in "${ROOT_PY[@]}"; do
    { [ -f "$out_dir/$n" ] && [ ! -L "$out_dir/$n" ]; } || BLOCKERS+=("missing: $SV_REL/$n")
    [ ! -e "$opt_dir/$n" ] || BLOCKERS+=("ambiguous ($SV_REL and $HS_REL): $n")
    add_entry py "$SV_REL/$n" "$n"
  done
  for n in "${OPT_PY[@]}"; do
    { [ -f "$opt_dir/$n" ] && [ ! -L "$opt_dir/$n" ]; } || BLOCKERS+=("missing: $HS_REL/$n")
    [ ! -e "$out_dir/$n" ] || BLOCKERS+=("ambiguous ($SV_REL and $HS_REL): $n")
    add_entry py "$HS_REL/$n" "$n"
  done
  for n in "${SH_FILES[@]}"; do
    { [ -f "$opt_dir/$n" ] && [ ! -L "$opt_dir/$n" ]; } || BLOCKERS+=("missing: $HS_REL/$n")
    if [ -f "$opt_dir/$n" ] && grep -q $'\r' "$opt_dir/$n"; then BLOCKERS+=("CRLF in shell script: $n"); fi
    add_entry sh "$HS_REL/$n" "$n"
  done
  local spec lsrc ldst lextra
  for spec in "${LIBS[@]}"; do
    IFS='|' read -r lsrc ldst lextra <<< "$spec"
    [[ $ldst =~ $LIB_DST_RE ]] && [ -z "$lextra" ] && [[ $lsrc =~ $PATH_RE ]] \
      || { BLOCKERS+=("unsafe library entry: $spec"); continue; }
    { [ -f "$workspace/$lsrc" ] && [ ! -L "$workspace/$lsrc" ]; } || { BLOCKERS+=("missing library: $lsrc"); continue; }
    [ -z "${DST_SEEN[$ldst]+x}" ] || { BLOCKERS+=("duplicate destination: $ldst"); continue; }
    DST_SEEN[$ldst]=1; SRC_SEEN[$lsrc]=1; SRC_GROUP[$lsrc]=lib; LIB_SRC[$lsrc]=$ldst
    E_GROUP+=(lib); E_SRC+=("$lsrc"); E_DST+=("$ldst")
  done
  for n in "${CONFIG_JSON[@]}"; do add_config "$n" required; done
  local pat f
  for pat in "${CONFIG_PATTERNS[@]}" "${PROFILE_PATTERNS[@]}"; do
    for f in "$opt_dir"/$pat; do   # nullglob; LC_ALL=C sorted
      n=${f##*/}
      [ -n "${DST_SEEN[config/$n]+x}" ] && continue
      add_config "$n" "pattern $pat"
    done
  done
  for n in "${EXTRA_CONFIG[@]}"; do
    [ -n "${DST_SEEN[config/$n]+x}" ] && continue
    add_config "$n" "--extra-config"
  done
}
CONFIG_ORIGIN=()
add_config() {  # $1 name.json, $2 origin; profile-final-* -> profile schema, else experiment schema
  local n=$1 origin=$2 f=$opt_dir/$1 kind=config
  [[ $n =~ $CONFIG_NAME_RE ]] || { BLOCKERS+=("unsafe config name ($origin): $n"); return; }
  case "$n" in profile-final-*) kind=profile;; esac
  if [ ! -f "$f" ] || [ -L "$f" ]; then
    BLOCKERS+=("missing ($origin): $HS_REL/$n")
  elif [ "$kind" = profile ]; then
    [[ $n =~ $PROFILE_NAME_RE ]] || BLOCKERS+=("profile config name rejected by profile.sh rule ($origin): $n")
    jq -e --argjson allowed "$PROFILE_KEYS" "$PROFILE_FILTER" "$f" >/dev/null 2>&1 \
      || BLOCKERS+=("not a profile-final config (schema: nsys block, QWEN36_NSYS_FORWARDS) ($origin): $n")
  else
    jq -e --argjson allowed "$CONFIG_KEYS" "$EXP_FILTER" "$f" >/dev/null 2>&1 \
      || BLOCKERS+=("not an experiment config (schema) ($origin): $n")
  fi
  add_entry "$kind" "$HS_REL/$n" "config/$n"
  CONFIG_ORIGIN+=("$kind ${origin// /:} $n")
}
hash_workspace() {  # rendered bytes for py/sh/config/profile, raw bytes for runtime/lib
  local h p s
  local -a raw=()
  for s in "${E_SRC[@]}"; do
    if rendered_group "${SRC_GROUP[$s]}"; then h=$(render "$workspace/$s" | sha256sum) || die 66 "cannot read $s"; WS_SHA[$s]=${h%% *}
    else raw+=("$s"); fi
  done
  if [ "${#raw[@]}" -gt 0 ]; then
    while read -r h p; do WS_SHA[${p#\*}]=$h; done < <(cd "$workspace" && sha256sum -- "${raw[@]}")
  fi
  [ "${#WS_SHA[@]}" -eq "${#E_SRC[@]}" ] || die 66 "could not hash every workspace source"
}
hash_native() {
  local i d h p m
  local -a have=()
  for i in "${!E_DST[@]}"; do d=${E_DST[$i]}; [ -f "$native/$d" ] && have+=("$d"); done
  NAT_SHA=(); NAT_MODE=()
  [ "${#have[@]}" -gt 0 ] || return 0
  while read -r h p; do NAT_SHA[${p#\*}]=$h; done < <(cd "$native" && sha256sum -- "${have[@]}")
  while read -r m p; do NAT_MODE[$p]=$m; done < <(cd "$native" && stat -c '%a %n' -- "${have[@]}")
}
load_freeze() {
  local line
  [ -f "$freeze" ] || die 65 "freeze file not found: $freeze"
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%$'\r'}
    [ -n "$line" ] || continue
    [[ $line == \#* ]] && continue
    [[ $line =~ ^([0-9a-f]{64})\ [\ *](.+)$ ]] || die 65 "malformed freeze line: ${line:0:90}"
    [ -z "${FREEZE[${BASH_REMATCH[2]}]+x}" ] || die 65 "duplicate path in freeze: ${BASH_REMATCH[2]}"
    FREEZE[${BASH_REMATCH[2]}]=${BASH_REMATCH[1]}
  done < "$freeze"
}
declare -a FREEZE_BAD=()
check_freeze() {
  local s p
  FREEZE_BAD=()
  for s in "${E_SRC[@]}"; do
    if [ -z "${FREEZE[$s]+x}" ]; then FREEZE_BAD+=("not in freeze: $s")
    elif [ "${FREEZE[$s]}" != "${WS_SHA[$s]}" ]; then FREEZE_BAD+=("changed since freeze: $s ${FREEZE[$s]:0:12}->${WS_SHA[$s]:0:12}"); fi
  done
  for p in "${!FREEZE[@]}"; do [ -n "${SRC_SEEN[$p]+x}" ] || FREEZE_BAD+=("freeze lists non-allowlisted path: $p"); done
}

# ---------------------------------------------------------------- plan
declare -a P_ACTION=()
declare -A CNT=([NEW]=0 [UPDATE]=0 [SAME]=0)
make_plan() {
  local i d s a
  P_ACTION=()
  for i in "${!E_DST[@]}"; do
    d=${E_DST[$i]}; s=${E_SRC[$i]}
    if [ -z "${NAT_SHA[$d]+x}" ]; then
      if [ -e "$native/$d" ] || [ -L "$native/$d" ]; then BLOCKERS+=("native destination is not a regular file: $d"); fi
      a=NEW
    elif [ "${NAT_SHA[$d]}" = "${WS_SHA[$s]}" ]; then a=SAME
    else a=UPDATE; fi
    P_ACTION+=("$a"); CNT[$a]=$(( ${CNT[$a]} + 1 ))
  done
}
declare -a NATIVE_ONLY_TREE=() OLD_FIXED=() OLD_REMAIN=() UNLISTED=() LIBREFS=()
analyse_extras() {
  local f rel n st lib cfg
  declare -A wsrel=()
  for n in "${RT_ALL[@]}"; do wsrel[$n]=1; done
  if [ -d "$dst_tree" ]; then
    while IFS= read -r -d '' f; do
      rel=${f#"$dst_tree"/}
      case "$rel" in *.pyc) continue;; esac
      [ -n "${wsrel[$rel]+x}" ] && continue
      if [[ $rel =~ $SOURCE_SUFFIX_RE ]] && [ "$skip_runtime" = 0 ]; then
        NATIVE_ONLY_TREE+=("BLOCKER $rel (native-only source file would enter experiment source hash)")
        BLOCKERS+=("native-only runtime source file: python/freetoken/$rel")
      elif [[ $rel =~ $SOURCE_SUFFIX_RE ]]; then NATIVE_ONLY_TREE+=("WARNING $rel (native-only source file; runtime skipped)")
      else NATIVE_ONLY_TREE+=("PRESERVED $rel"); fi
    done < <(find "$dst_tree" -name __pycache__ -prune -o ! -type d -print0 | sort -z)
  fi
  # old-workspace-path references in allowlisted sources / current native targets
  for i in "${!E_DST[@]}"; do
    case "${E_GROUP[$i]}" in runtime|lib) continue;; esac
    if grep -qF -- "$OLD_WORKSPACE" "$workspace/${E_SRC[$i]}"; then
      BLOCKERS+=("workspace source still contains old path: ${E_SRC[$i]}")
    elif [ -f "$native/${E_DST[$i]}" ] && grep -qF -- "$OLD_WORKSPACE" "$native/${E_DST[$i]}"; then
      OLD_FIXED+=("${E_DST[$i]}")
    fi
  done
  for f in "$native"/*.py "$native"/*.sh; do
    n=${f##*/}
    [ -n "${DST_SEEN[$n]+x}" ] && continue
    grep -qF -- "$OLD_WORKSPACE" "$f" && OLD_REMAIN+=("$n")
  done
  # workspace files outside the allowlist (visible on purpose: add explicitly if needed)
  for f in "$out_dir"/*.py "$opt_dir"/*.py "$opt_dir"/*.sh "$opt_dir"/*.json; do
    n=${f##*/}
    case "$f" in *.json) [ -n "${DST_SEEN[config/$n]+x}" ] && continue
                         if [ ! -f "$native/config/$n" ]; then st=config:absent
                         elif cmp -s -- <(render "$f") "$native/config/$n"; then st=config:same; else st=config:DIFFERS; fi
                         # make missed configs visible: a schema-valid JSON here is probably a profile to add
                         if jq -e --argjson allowed "$CONFIG_KEYS" "$EXP_FILTER" "$f" >/dev/null 2>&1; then
                           st+=", EXPERIMENT-CONFIG SCHEMA: add to CONFIG_JSON or --extra-config if a run needs it"
                         elif jq -e --argjson allowed "$PROFILE_KEYS" "$PROFILE_FILTER" "$f" >/dev/null 2>&1; then
                           st+=", PROFILE-CONFIG SCHEMA: rename to profile-final-<v>.json or --extra-config"
                         fi;;
                 *) [ -n "${DST_SEEN[$n]+x}" ] && continue
                    if [ ! -f "$native/$n" ]; then st=native:absent
                    elif cmp -s -- <(render "$f") "$native/$n"; then st=native:same; else st=native:DIFFERS; fi;;
    esac
    UNLISTED+=("${f#"$workspace"/} [$st]")
  done
  # native libraries referenced by allowlisted configs (bench/experiment hash them)
  for i in "${!E_DST[@]}"; do
    case "${E_GROUP[$i]}" in config|profile) ;; *) continue;; esac
    cfg=${E_DST[$i]#config/}
    [ -f "$opt_dir/$cfg" ] || continue
    while IFS= read -r lib; do
      [ -n "$lib" ] || continue
      if [ -f "$lib" ]; then LIBREFS+=("$cfg -> $lib [present]"); else LIBREFS+=("$cfg -> $lib [MISSING]"); fi
    done < <(render "$opt_dir/$cfg" | jq -r '(.env // {}) | to_entries[] | select(.key|endswith("_LIBRARY")) | .value' 2>/dev/null)
  done
}
RT_REPORT=()
runtime_report() {  # read-only: runtime source vs HANDOFF-SNAPSHOT (STOP) and, if skipped, ws vs native
  local h p rel wch=0 nch=0 wsadd=0 nadd=0 skipdiff=0
  declare -A SW=() SN=() RW=() RN=()
  if [ -f "$snapshot" ]; then
    while read -r h p; do SW[$p]=$h; done < <(jq -r '.source_workspace_hashes|to_entries[]|.value+" "+.key' "$snapshot")
    while read -r h p; do SN[$p]=$h; done < <(jq -r '.source_native_hashes|to_entries[]|.value+" "+.key' "$snapshot")
  fi
  if [ "${#RT_ALL[@]}" -gt 0 ]; then
    while read -r h p; do RW[${p#\*}]=$h; done < <(cd "$src_tree" && sha256sum -- "${RT_ALL[@]}")
  fi
  if [ -d "$dst_tree" ]; then
    while IFS= read -r -d '' p; do
      rel=${p#"$dst_tree"/}
      [[ $rel =~ $SOURCE_SUFFIX_RE ]] || continue
      h=$(sha256sum -- "$p"); RN[$rel]=${h%% *}
    done < <(find "$dst_tree" -name __pycache__ -prune -o -type f -print0)
  fi
  for rel in "${!RW[@]}"; do
    [[ $rel =~ $SOURCE_SUFFIX_RE ]] || continue
    if [ -z "${SW[$rel]+x}" ]; then wsadd=$((wsadd+1)); elif [ "${SW[$rel]}" != "${RW[$rel]}" ]; then wch=$((wch+1)); fi
    if [ "$skip_runtime" = 1 ] && [ "${RN[$rel]:-x}" != "${RW[$rel]}" ]; then skipdiff=$((skipdiff+1)); fi
  done
  for rel in "${!RN[@]}"; do
    if [ -z "${SN[$rel]+x}" ]; then nadd=$((nadd+1)); elif [ "${SN[$rel]}" != "${RN[$rel]}" ]; then nch=$((nch+1)); fi
  done
  if [ -f "$snapshot" ]; then
    RT_REPORT+=("runtime source vs HANDOFF-SNAPSHOT (STOP, ${#SW[@]} files): workspace changed=$wch added=$wsadd; native changed=$nch added=$nadd")
  else RT_REPORT+=("HANDOFF-SNAPSHOT.json not found: STOP comparison skipped"); fi
  if [ "$skip_runtime" = 1 ]; then
    RT_REPORT+=("--skip-runtime: python/freetoken NOT synced; workspace source files differing from native: $skipdiff")
  fi
  return 0
}
print_plan() {
  local i a d s
  printf '== plan (%s entries): NEW=%s UPDATE=%s SAME=%s\n' "${#E_DST[@]}" "${CNT[NEW]}" "${CNT[UPDATE]}" "${CNT[SAME]}"
  for i in "${!E_DST[@]}"; do
    a=${P_ACTION[$i]}; d=${E_DST[$i]}; s=${E_SRC[$i]}
    case "$a" in
      SAME) [ "${E_GROUP[$i]}" = runtime ] || printf '  SAME    %-7s %s %s\n' "${E_GROUP[$i]}" "${WS_SHA[$s]:0:12}" "$d";;
      NEW) printf '  NEW     %-7s %s %s <- %s\n' "${E_GROUP[$i]}" "${WS_SHA[$s]:0:12}" "$d" "$s";;
      UPDATE) printf '  UPDATE  %-7s %s->%s %s (mode %s)\n' "${E_GROUP[$i]}" "${NAT_SHA[$d]:0:12}" "${WS_SHA[$s]:0:12}" "$d" "${NAT_MODE[$d]}";;
    esac
  done
  printf '  (runtime SAME entries not listed individually)\n'
  local g; declare -A GC=()
  for g in "${E_GROUP[@]}"; do GC[$g]=$(( ${GC[$g]:-0} + 1 )); done
  printf '== entries by group: runtime=%s py=%s sh=%s config=%s profile=%s lib=%s\n' "${GC[runtime]:-0}" "${GC[py]:-0}" \
    "${GC[sh]:-0}" "${GC[config]:-0}" "${GC[profile]:-0}" "${GC[lib]:-0}"
  printf '== config/profile entries not from the required CONFIG_JSON list: '
  local nonreq=0; for d in "${CONFIG_ORIGIN[@]}"; do case "$d" in *" required "*) ;; *) nonreq=$((nonreq+1));; esac; done
  printf '%s\n' "$nonreq"
  for d in "${CONFIG_ORIGIN[@]}"; do case "$d" in *" required "*) ;; *) printf '  %s\n' "$d";; esac; done
  local nmode=0 want
  for i in "${!E_DST[@]}"; do
    d=${E_DST[$i]}
    case "${E_GROUP[$i]}" in py|config|profile|lib) want=600;; sh) want=700;; *) continue;; esac
    if [ -n "${NAT_MODE[$d]+x}" ] && [ "${NAT_MODE[$d]}" != "$want" ]; then nmode=$((nmode+1)); fi
  done
  printf '== mode normalisation on apply (py/config/profile/lib -> 600, sh -> 700, private-libraries/ -> 700, runtime tree unchanged): %s existing file(s)\n' "$nmode"
  printf '== libraries (group lib, sha256 bound by --freeze only): %s\n' "${#LIB_SRC[@]}"
  for d in "${!LIB_SRC[@]}"; do printf '  LIB     %s %s -> %s%s\n' "${WS_SHA[$d]:0:12}" "$d" "${LIB_SRC[$d]}" \
    "$([ -n "$freeze" ] && [ "${FREEZE[$d]:-}" = "${WS_SHA[$d]}" ] && echo ' [freeze ok]')"; done
  printf '== runtime tree\n'
  for d in "${RT_REPORT[@]}"; do printf '  %s\n' "$d"; done
  printf '== native-only files in python/freetoken (never deleted): %s\n' "${#NATIVE_ONLY_TREE[@]}"
  for d in "${NATIVE_ONLY_TREE[@]}"; do printf '  %s\n' "$d"; done
  printf '== old workspace path (%s) in native targets fixed by this sync: %s\n' "$OLD_WORKSPACE" "${#OLD_FIXED[@]}"
  for d in "${OLD_FIXED[@]}"; do printf '  FIXED   %s\n' "$d"; done
  printf '== old workspace path remaining in native top-level files outside allowlist: %s\n' "${#OLD_REMAIN[@]}"
  for d in "${OLD_REMAIN[@]}"; do printf '  REMAINS %s\n' "$d"; done
  printf '== workspace files NOT synced (outside allowlist): %s\n' "${#UNLISTED[@]}"
  for d in "${UNLISTED[@]}"; do printf '  %s\n' "$d"; done
  printf '== native libraries referenced by allowlisted configs\n'
  for d in "${LIBREFS[@]}"; do printf '  %s\n' "$d"; done
  printf '== preserved, never written: config/experiment.env (not read), config/teacher-fixture.json, config/profile-final-*-ncu<k>.json, active-run.json, reports/, logs/, build/, private-*/, cuda-transport/, include/, tools/, sections/, harness-backups/\n'
}

# ---------------------------------------------------------------- modes
print_freeze() {
  local s
  printf '# sync-v5 freeze candidate; paths relative to the checkout; utc=%s\n' "$(date -u +%FT%TZ)"
  printf '# rendered with site values digest=%s (py/sh/config/profile bytes as written to native)\n' "$(render_digest)"
  printf '# options: skip_runtime=%s extra_config=%s (pass the same options to --dry-run/--apply)\n' \
    "$skip_runtime" "$(if [ "${#EXTRA_CONFIG[@]}" -gt 0 ]; then printf '%s,' "${EXTRA_CONFIG[@]}"; else printf -- '-'; fi)"
  for s in "${E_SRC[@]}"; do printf '%s  %s\n' "${WS_SHA[$s]}" "$s"; done
}

do_verify() {
  local m=$reports/sync-v5-$label.json bad=0 group src dst sha cur
  [ -f "$m" ] || die 65 "manifest not found: $m"
  printf 'sync-v5 VERIFY (read-only) label=%s manifest=%s\n' "$label" "$m"
  while IFS=$'\t' read -r group src dst sha; do
    if [ -f "$native/$dst" ]; then cur=$(sha256sum -- "$native/$dst"); cur=${cur%% *}; else cur=MISSING; fi
    [ "$cur" = "$sha" ] || { printf '  NATIVE DRIFT    %s %s->%s\n' "$dst" "${sha:0:12}" "${cur:0:12}"; bad=$((bad+1)); }
    if [ ! -f "$workspace/$src" ]; then cur=MISSING
    elif rendered_group "$group"; then cur=$(render "$workspace/$src" | sha256sum); cur=${cur%% *}
    else cur=$(sha256sum -- "$workspace/$src"); cur=${cur%% *}; fi
    [ "$cur" = "$sha" ] || { printf '  WORKSPACE DRIFT %s %s->%s\n' "$src" "${sha:0:12}" "${cur:0:12}"; bad=$((bad+1)); }
  done < <(jq -r '.files[] | [.group,.src,.dst,.sha256] | @tsv' "$m")
  printf 'verify: %s drift(s)\n' "$bad"
  [ "$bad" -eq 0 ] || exit 65
}

STAGE=; ROLLBACK=; PHASE=init; MANIFEST_JSON=; MANIFEST_SHA=; FREEZE_SHA=
declare -a DONE_IDX=()
write_manifest() {  # $1 = status; uses noclobber, never overwrites
  local status=$1 tsv i d s a nb na mb ma
  tsv=$STAGE/manifest.tsv
  : > "$tsv"
  for i in "${!E_DST[@]}"; do
    d=${E_DST[$i]}; s=${E_SRC[$i]}; a=${P_ACTION[$i]}
    nb=${NAT_SHA[$d]:--}; mb=${NAT_MODE[$d]:--}
    if [ -f "$native/$d" ]; then na=$(sha256sum -- "$native/$d"); na=${na%% *}; ma=$(stat -c %a -- "$native/$d"); else na=-; ma=-; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${E_GROUP[$i]}" "$s" "$d" "$a" "$nb" "${WS_SHA[$s]}" "$na" "$mb" "$ma" >> "$tsv"
  done
  local units_now; units_now=$unit_report
  ( set -o noclobber
    jq -n --rawfile rows "$tsv" --arg status "$status" --arg label "$label" --arg utc "$(date -u +%FT%TZ)" \
      --arg workspace "$workspace" --arg native "$native" --arg freeze "$(realpath -m -- "$freeze")" --arg render "$(render_digest)" \
      --arg freeze_sha "$FREEZE_SHA" --arg script_sha "$(sha256sum -- "$0" | cut -d' ' -f1)" \
      --arg units "$units_now" --arg rollback "$ROLLBACK" --argjson sandbox "$sandbox" \
      --arg native_only "$(printf '%s\n' "${NATIVE_ONLY_TREE[@]}")" --arg unlisted "$(printf '%s\n' "${UNLISTED[@]}")" \
      --arg old_fixed "$(printf '%s\n' "${OLD_FIXED[@]}")" --argjson skip_runtime "$skip_runtime" \
      --arg runtime_report "$(printf '%s\n' "${RT_REPORT[@]}")" --arg gpu_lock "$LOCK_HOW" \
      --arg extra "$(printf '%s\n' "${EXTRA_CONFIG[@]}")" --arg origins "$(printf '%s\n' "${CONFIG_ORIGIN[@]}")" '
      def lines: split("\n")|map(select(length>0));
      ($rows|lines|map(split("\t")|{group:.[0],src:.[1],dst:.[2],action:.[3],
         sha256_before:(if .[4]=="-" then null else .[4] end),sha256:.[5],
         sha256_native_now:(if .[6]=="-" then null else .[6] end),
         mode_before:(if .[7]=="-" then null else .[7] end),mode:(if .[8]=="-" then null else .[8] end)})) as $files
      | {schema:1,kind:"sync-v5-manifest",status:$status,label:$label,utc:$utc,sandbox:($sandbox==1),
         workspace:$workspace,native:$native,freeze_file:$freeze,freeze_sha256:$freeze_sha,render_digest:$render,
         script_sha256:$script_sha,units_at_start:$units,gpu_lock:("for the whole apply: "+$gpu_lock),
         extra_config:($extra|lines),config_origins:($origins|lines),
         rollback_dir:$rollback,skip_runtime:($skip_runtime==1),runtime_report:($runtime_report|lines),
         counts:{entries:($files|length),new:($files|map(select(.action=="NEW"))|length),
                 update:($files|map(select(.action=="UPDATE"))|length),same:($files|map(select(.action=="SAME"))|length),
                 verified:($files|map(select(.sha256_native_now==.sha256))|length)},
         native_only_runtime_files:($native_only|lines),old_workspace_path_fixed:($old_fixed|lines),
         workspace_files_not_synced:($unlisted|lines),files:$files}' > "$MANIFEST_JSON"
    awk -F'\t' '{print $6"  "$3}' "$tsv" > "$MANIFEST_SHA" )
}
on_exit() {
  local rc=$?
  if [ "$mode" = apply ] && [ "$PHASE" = replacing ]; then
    printf 'sync-v5: FAILED during replacement (rc=%s); replaced so far: %s; previous bytes in %s\n' "$rc" "${#DONE_IDX[@]}" "$ROLLBACK" >&2
    write_manifest FAILED_DURING_REPLACE 2>/dev/null || true
  fi
  if [ -n "$STAGE" ] && [ -d "$STAGE" ]; then  # only our own staging dir
    find "$STAGE" -mindepth 1 -depth -delete 2>/dev/null || true; rmdir -- "$STAGE" 2>/dev/null || true
  fi
}

do_apply() {
  local i d s a mode_new parent got
  MANIFEST_JSON=$reports/sync-v5-$label.json; MANIFEST_SHA=$reports/sync-v5-$label.sha256
  STAGE=$native/.sync-v5-stage.$label; ROLLBACK=$native/sync-v5-rollback/$label
  [ -d "$native" ] && [ -d "$reports" ] || die 67 "native root or reports/ missing"
  for d in "$MANIFEST_JSON" "$MANIFEST_SHA" "$STAGE" "$ROLLBACK"; do
    { [ -e "$d" ] || [ -L "$d" ]; } && die 67 "already exists (labels are single-use): $d"
  done
  FREEZE_SHA=$(sha256sum -- "$freeze" | cut -d' ' -f1)
  trap on_exit EXIT
  mkdir -m 700 -- "$STAGE"
  PHASE=staging
  for i in "${!E_DST[@]}"; do
    a=${P_ACTION[$i]}; [ "$a" = SAME ] && continue
    d=${E_DST[$i]}; s=${E_SRC[$i]}
    mkdir -p -- "$(dirname -- "$STAGE/files/$d")"
    if rendered_group "${E_GROUP[$i]}"; then render "$workspace/$s" > "$STAGE/files/$d"; else cp -- "$workspace/$s" "$STAGE/files/$d"; fi
    got=$(sha256sum -- "$STAGE/files/$d"); got=${got%% *}
    [ "$got" = "${FREEZE[$s]}" ] || die 65 "staged bytes differ from freeze (changed during sync?): $s"
  done
  mkdir -p -m 700 -- "$native/sync-v5-rollback"
  mkdir -m 700 -- "$ROLLBACK"
  for i in "${!E_DST[@]}"; do
    [ "${P_ACTION[$i]}" = UPDATE ] || continue
    d=${E_DST[$i]}
    mkdir -p -- "$ROLLBACK/$(dirname -- "$d")"
    cp -p -- "$native/$d" "$ROLLBACK/$d"
    got=$(sha256sum -- "$ROLLBACK/$d"); got=${got%% *}
    [ "$got" = "${NAT_SHA[$d]}" ] || die 70 "native file changed while backing it up: $d"
  done
  # lock-unaware starters (qualify.py, run-*-gates.sh, manual start, production) may have started meanwhile
  check_units; check_procs
  PHASE=replacing
  for i in "${!E_DST[@]}"; do
    a=${P_ACTION[$i]}; [ "$a" = SAME ] && continue
    d=${E_DST[$i]}
    case "${E_GROUP[$i]}" in
      py|config|profile|lib) mode_new=600;;
      sh) mode_new=700;;
      runtime) if [ "$a" = UPDATE ]; then mode_new=${NAT_MODE[$d]}; else mode_new=600; fi;;
    esac
    parent=$(dirname -- "$native/$d")
    [ -d "$parent" ] || mkdir -p -m 700 -- "$parent"
    chmod "$mode_new" -- "$STAGE/files/$d"
    mv -f -T -- "$STAGE/files/$d" "$native/$d"
    DONE_IDX+=("$i")
  done
  # same policy as the old sync.sh (chmod 600 *.py) extended to sh/config; runtime tree keeps modes
  for i in "${!E_DST[@]}"; do
    case "${E_GROUP[$i]}" in
      py|config|profile) chmod 600 -- "$native/${E_DST[$i]}";;
      lib) chmod 600 -- "$native/${E_DST[$i]}"; chmod 700 -- "$(dirname -- "$native/${E_DST[$i]}")";;
      sh) chmod 700 -- "$native/${E_DST[$i]}";;
    esac
  done
  chmod 700 -- "$native" "$reports"
  PHASE=verifying
  local bad=0
  for i in "${!E_DST[@]}"; do
    d=${E_DST[$i]}; s=${E_SRC[$i]}
    got=$(sha256sum -- "$native/$d"); got=${got%% *}
    [ "$got" = "${FREEZE[$s]}" ] || { printf '  POST-VERIFY MISMATCH %s\n' "$d" >&2; bad=$((bad+1)); }
  done
  if [ "$bad" -ne 0 ]; then write_manifest FAILED_POST_VERIFY; PHASE=done; die 70 "$bad file(s) differ after sync"; fi
  write_manifest PASS
  PHASE=done
  find "$STAGE" -mindepth 1 -depth -delete; rmdir -- "$STAGE"
  [ "$(find "$ROLLBACK" -type f | wc -l)" -gt 0 ] || rmdir -- "$ROLLBACK"
  printf 'sync-v5 APPLY PASS label=%s NEW=%s UPDATE=%s SAME=%s\n  manifest: %s\n  sha256:   %s\n  rollback: %s\n' \
    "$label" "${CNT[NEW]}" "${CNT[UPDATE]}" "${CNT[SAME]}" "$MANIFEST_JSON" "$MANIFEST_SHA" "$ROLLBACK"
}

# ---------------------------------------------------------------- main
if [ "$mode" = verify ]; then do_verify; exit 0; fi
build_entries
if [ "${#BLOCKERS[@]}" -gt 0 ]; then printf '  %s\n' "${BLOCKERS[@]}" >&2; die 66 "${#BLOCKERS[@]} blocker(s) in allowlisted sources"; fi
hash_workspace
if [ "$mode" = print-freeze ]; then print_freeze; exit 0; fi

# freeze is validated from workspace data only, BEFORE units, native or the GPU lock are touched:
# a bad/missing/mismatching freeze makes --apply exit 65 without ever opening the lock file.
[ -z "$freeze" ] || { load_freeze; check_freeze; }
if [ "$mode" = apply ] && [ "${#FREEZE_BAD[@]}" -gt 0 ]; then
  printf '  %s\n' "${FREEZE_BAD[@]}" >&2
  die 65 "${#FREEZE_BAD[@]} freeze mismatch(es); nothing read from native, lock not taken, nothing written"
fi
check_units; check_procs
if [ "$mode" = apply ]; then take_lock; lockinfo=$LOCK_HOW; check_units; check_procs
else lockinfo=$(lock_state); [ "$lockinfo" != HELD ] || die 75 "GPU-exclusive lock $lock is held (matrix/pilot/GPU job running)"; fi
[ -d "$native" ] || die 66 "native root missing: $native"
hash_native; make_plan; analyse_extras; runtime_report

printf 'sync-v5 %s%s label=%s%s\n' "$(printf '%s' "$mode" | tr a-z A-Z)" "$([ "$mode" = dry-run ] && echo ' (no writes)')" \
  "${label:--}" "$([ "$skip_runtime" = 1 ] && echo ' [skip-runtime]')"
printf '  workspace=%s\n  native=%s%s\n  site config: %s (render digest %s)\n  units: %s\n  gpu-lock %s: %s\n' "$workspace" "$native" \
  "$([ "$sandbox" = 1 ] && echo ' [SANDBOX]')" "${env_file:--}" "$(render_digest)" "$unit_report" "$lock" "$lockinfo"
if [ -n "$label" ]; then
  for d in "$reports/sync-v5-$label.json" "$reports/sync-v5-$label.sha256" "$native/sync-v5-rollback/$label"; do
    [ ! -e "$d" ] || printf '  WARNING label already used: %s exists\n' "$d"
  done
fi
print_plan
if [ -n "$freeze" ]; then
  printf '== freeze %s: %s mismatch(es)\n' "$freeze" "${#FREEZE_BAD[@]}"
  for d in "${FREEZE_BAD[@]}"; do printf '  %s\n' "$d"; done
else
  printf '== freeze: not given (apply requires --freeze; create with --print-freeze AFTER agents froze their files)\n'
fi
if [ "${#BLOCKERS[@]}" -gt 0 ]; then printf '== BLOCKERS\n'; printf '  %s\n' "${BLOCKERS[@]}"; fi

if [ "$mode" = dry-run ]; then
  printf 'dry-run: nothing written\n'
  [ "${#BLOCKERS[@]}" -eq 0 ] || exit 66
  exit 0
fi
[ "${#BLOCKERS[@]}" -eq 0 ] || die 66 "${#BLOCKERS[@]} blocker(s); nothing written"
[ "${#FREEZE_BAD[@]}" -eq 0 ] || die 65 "${#FREEZE_BAD[@]} freeze mismatch(es); nothing written"
do_apply
