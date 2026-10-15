#!/usr/bin/env bash
# Usage: run-context-gates.sh PROFILE CONTEXT LENGTH [NAME]
# NAME (new runs): reports NAME-{run,functional,boundary,backend,cancel}.json, refused if any exists.
# Without NAME the historical f1 names (functional-PROFILE, boundary-PROFILE, ...) are kept.
set -euo pipefail
root=${QWEN36_DATA_ROOT:-/home/USER/qwen36-api/optimization}
python=${QWEN36_GPU_PYTHON:-/home/USER/qwen36-api/.venv/bin/python}
profile="${1:?profile}"; context="${2:?context}"; length="${3:?length}"; name="${4:-}"
if [ -n "$name" ]; then
  [[ "$name" =~ ^[a-z0-9][a-z0-9.-]{0,80}$ ]] || { echo "bad NAME: $name" >&2; exit 2; }
  run_copy=$name-run functional=$name-functional boundary=$name-boundary backend=$name-backend cancel=$name-cancel
  for report in "$run_copy" "$functional" "$boundary" "$backend" "$cancel"; do
    if [ -e "$root/reports/$report.json" ]; then echo "report exists, use a new NAME: $report.json" >&2; exit 2; fi
  done
else
  run_copy=context-$profile-run functional=functional-$profile boundary=boundary-$profile
  backend=backend-$profile cancel=prefill-cancel-$profile
fi
for unit in qwen36-inference qwen36-experiment; do
  if systemctl --user is-active --quiet "$unit"; then
    echo "Refusing to take over active $unit" >&2; exit 73
  fi
done
bash "$root/start-experiment.sh" "$profile"
trap 'systemctl --user stop qwen36-experiment' EXIT
"$python" "$root/check-experiment.py" 300
cp "$root/active-run.json" "$root/reports/$run_copy.json"
"$python" "$root/quality.py" --name "$functional" --long "$length" > "$root/reports/$functional.console.log"
"$python" "$root/boundary.py" --context "$context" --name "$boundary" > "$root/reports/$boundary.console.log"
"$python" "$root/test-backend-v2.py" --context "$context" --name "$backend"
"$python" "$root/test-prefill-cancel.py" --length "$length" --name "$cancel"
echo "Completed context gates: $profile${name:+ ($name)}"
