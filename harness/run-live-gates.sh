#!/usr/bin/env bash
set -euo pipefail
root=/home/USER/qwen36-api/optimization
python=/home/USER/qwen36-api/.venv/bin/python
profile="${1:?profile required}"
context="${2:?context required}"
for unit in qwen36-inference qwen36-experiment; do
  if systemctl --user is-active --quiet "$unit"; then
    echo "Refusing to take over active $unit" >&2
    exit 73
  fi
done
bash "$root/start-experiment.sh" "$profile"
trap 'systemctl --user stop qwen36-experiment' EXIT
"$python" "$root/check-experiment.py" 300
cp "$root/active-run.json" "$root/reports/live-gates-$profile-run.json"
"$python" "$root/test-output-budget.py"
cp "$root/reports/output-budget.json" "$root/reports/output-budget-$profile.json"
"$python" "$root/test-backend-v2.py" --context "$context" --name "backend-$profile"
"$python" "$root/test-resilience.py"
cp "$root/reports/resilience.json" "$root/reports/resilience-$profile.json"
"$python" "$root/boundary.py" --context "$context" --name "boundary-$profile"
