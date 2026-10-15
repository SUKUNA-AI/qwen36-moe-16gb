#!/usr/bin/env bash
# sync-v5.sh on the repository layout (A3), CPU only, inside SYNC_V5_SANDBOX: repo-relative sources, site paths from
# a synthetic local.env, placeholder rendering, --print-freeze / --dry-run / --apply / --verify, freeze mismatch and
# missing site values. systemctl is a stub that reports both units inactive; nothing outside the sandbox is touched.
# Prints {"tests_run": N, "failed_count": F, "skipped": 0} for cpu_checks.py.
set -uo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/.." && pwd)
tmp=$(mktemp -d "${TMPDIR:-/tmp}/qwen36-sync-v5-test.XXXXXX")
trap 'rm -rf -- "$tmp"' EXIT
sb=$tmp/sync-v5-sandbox-t
mkdir -p "$sb/native/reports" "$sb/native/config" "$sb/workspace" "$tmp/bin"
# The sandbox workspace is a copy of this checkout's synced sources (no .git, caches or private data).
mkdir -p "$sb/workspace/serving" "$sb/workspace/harness" "$sb/workspace/freetoken/python"
find "$repo/serving" "$repo/harness" -maxdepth 1 -type f -print0 | while IFS= read -r -d '' f; do
  cp -- "$f" "$sb/workspace/${f#"$repo"/}"; done
cp -r -- "$repo/freetoken/python/freetoken" "$sb/workspace/freetoken/python/freetoken"
find "$sb/workspace" -name __pycache__ -type d -prune -exec rm -rf {} +
: > "$sb/gpu-exclusive.lock"
cat > "$tmp/bin/systemctl" <<'EOF'
#!/bin/sh
case "$*" in *show*) echo inactive;; *list-units*) ;; esac
exit 0
EOF
chmod 755 "$tmp/bin/systemctl"
export PATH="$tmp/bin:$PATH"
site=$tmp/local.env
cat > "$site" <<EOF
# synthetic site
QWEN36_ROOT=$tmp/root      # runtime root
QWEN36_NATIVE='$tmp/root/optimization'
QWEN36_ARCHIVE=$tmp/archive
QWEN36_MODEL_DIR=$tmp/model
EOF
sync() { SYNC_V5_SANDBOX=$sb bash "$repo/harness/sync-v5.sh" --env "$site" "$@"; }

run=0 failed=0
ok() { run=$((run+1)); echo "ok - $1"; }
bad() { run=$((run+1)); failed=$((failed+1)); echo "FAIL - $1"; }
expect() { local label=$1; shift; if "$@"; then ok "$label"; else bad "$label"; fi; }

freeze=$tmp/freeze.sha256
sync --print-freeze > "$freeze" 2> "$tmp/freeze.err"; rc=$?
expect "print-freeze exits 0 (rc=$rc: $(head -c 300 "$tmp/freeze.err"))" test "$rc" = 0
expect "freeze uses repo-relative sources" grep -q '  freetoken/python/freetoken/quality_evidence\.py$' "$freeze"
expect "freeze lists serving/ and harness/ sources" bash -c 'grep -q "  serving/guard\.py$" "$1" && grep -q "  harness/bench\.py$" "$1" && grep -q "  harness/main-release\.json$" "$1"' _ "$freeze"
expect "next-release configs come in by pattern" grep -q '  harness/main-release-shortroute-fla-wu-overlap\.json$' "$freeze"
expect "no old workspace prefixes" bash -c '! grep -Eq "  (outputs/qwen36-api|work/FreeToken-opt)/" "$1"' _ "$freeze"
expect "freeze header carries no site path" bash -c '! grep -q "$2" "$1"' _ "$freeze" "$tmp"
raw=$(sha256sum < "$repo/harness/experiment.py" | cut -c1-64)
rendered=$(sed -e "s|/home/USER/qwen36-api|$tmp/root|g" -e "s|/path/to/LLM/qwen36-api/model|$tmp/model|g" "$repo/harness/experiment.py" | sha256sum | cut -c1-64)
expect "experiment.py is frozen as rendered bytes" grep -q "^$rendered  harness/experiment\.py$" "$freeze"
expect "rendered differs from raw" test "$raw" != "$rendered"
rt=$(sha256sum < "$repo/freetoken/python/freetoken/quality_evidence.py" | cut -c1-64)
expect "runtime tree is frozen raw" grep -q "^$rt  freetoken/python/freetoken/quality_evidence\.py$" "$freeze"

before=$(find "$sb/native" | sort | sha256sum)
sync --dry-run --label t1 --freeze "$freeze" > "$tmp/dry.txt" 2>&1; rc=$?
expect "dry-run exits 0 (rc=$rc)" test "$rc" = 0
expect "dry-run plans NEW entries into an empty overlay" grep -q '^== plan ([0-9]* entries): NEW=[1-9]' "$tmp/dry.txt"
expect "dry-run freeze has 0 mismatches" grep -q ': 0 mismatch(es)$' "$tmp/dry.txt"
expect "dry-run writes nothing" test "$before" = "$(find "$sb/native" | sort | sha256sum)"

sync --apply --label t1 --freeze "$freeze" > "$tmp/apply.txt" 2>&1; rc=$?
expect "apply exits 0 (rc=$rc: $(tail -c 300 "$tmp/apply.txt"))" test "$rc" = 0
expect "native experiment.py has the site root and model" bash -c 'grep -qF "$2/root/optimization" "$1/experiment.py" && grep -qF "$2/model" "$1/experiment.py" && ! grep -q "/home/USER\|/path/to/" "$1/experiment.py"' _ "$sb/native" "$tmp"
expect "native start-experiment.sh takes configs from native config/" grep -qF "workspace='$sb/native/config'" "$sb/native/start-experiment.sh"
expect "native candidate config has rendered library paths" bash -c 'grep -qF "$2/root/optimization/private-libraries/" "$1/config/main-release-shortroute.json"' _ "$sb/native" "$tmp"
expect "native runtime tree is byte-identical" cmp -s "$repo/freetoken/python/freetoken/quality_evidence.py" "$sb/native/python/freetoken/quality_evidence.py"
expect "native sync-v5.sh keeps its placeholder table" grep -qF "'/home/''USER/qwen36-api'" "$sb/native/sync-v5.sh"
expect "manifest written" test -f "$sb/native/reports/sync-v5-t1.json"
sync --verify --label t1 > "$tmp/verify.txt" 2>&1; rc=$?
expect "verify after apply: 0 drifts (rc=$rc)" bash -c '[ "$1" = 0 ] && grep -q "verify: 0 drift" "$2"' _ "$rc" "$tmp/verify.txt"
sync --dry-run --label t2 > "$tmp/dry2.txt" 2>&1; rc=$?
expect "second dry-run: everything SAME" bash -c '[ "$1" = 0 ] && grep -q "NEW=0 UPDATE=0" "$2"' _ "$rc" "$tmp/dry2.txt"

echo '# tampered' >> "$sb/workspace/harness/quality.py"
sync --apply --label t3 --freeze "$freeze" > "$tmp/mismatch.txt" 2>&1; rc=$?
expect "apply refuses a freeze mismatch with 65" test "$rc" = 65
expect "refused apply writes no manifest" test ! -e "$sb/native/reports/sync-v5-t3.json"
sync --verify --label t1 > /dev/null 2>&1; rc=$?
expect "verify reports workspace drift with 65" test "$rc" = 65

grep -v QWEN36_MODEL_DIR "$site" > "$tmp/nomodel.env"
SYNC_V5_SANDBOX=$sb env -u QWEN36_MODEL_DIR bash "$repo/harness/sync-v5.sh" --env "$tmp/nomodel.env" --print-freeze > /dev/null 2> "$tmp/nomodel.err"; rc=$?
expect "missing QWEN36_MODEL_DIR blocks files that need it (66)" bash -c '[ "$1" = 66 ] && grep -q "needs QWEN36_MODEL_DIR" "$2"' _ "$rc" "$tmp/nomodel.err"
printf 'QWEN36_ROOT=/home/USER/qwen36-api\nQWEN36_NATIVE=/home/USER/qwen36-api/optimization\n' > "$tmp/placeholder.env"
bash "$repo/harness/sync-v5.sh" --env "$tmp/placeholder.env" --dry-run > /dev/null 2> "$tmp/ph.err"; rc=$?
expect "placeholder site config is refused (2) before any native access" bash -c '[ "$1" = 2 ] && grep -q "QWEN36_ROOT and QWEN36_NATIVE" "$2"' _ "$rc" "$tmp/ph.err"
SYNC_V5_SANDBOX=$tmp/root/optimization/sync-v5-sandbox-x bash "$repo/harness/sync-v5.sh" --env "$site" --print-freeze > /dev/null 2>&1; rc=$?
expect "sandbox inside QWEN36_ROOT is refused (2)" test "$rc" = 2

tracked=$repo/harness/sync-next-freeze.sha256
expect "tracked next freeze exists" test -f "$tracked"
expect "tracked next freeze lists only existing repo-relative sources" bash -c '
  n=0; while read -r h p; do case "$h" in "#"*|"") continue;; esac; [ -f "$1/$p" ] || exit 1; n=$((n+1)); done < "$2"; [ "$n" -gt 100 ]' _ "$repo" "$tracked"

printf '{"tests_run": %s, "failed_count": %s, "skipped": 0}\n' "$run" "$failed"
[ "$failed" = 0 ]
