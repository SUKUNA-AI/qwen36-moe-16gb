#!/usr/bin/env bash
set -euo pipefail
root=/home/USER/qwen36-api/optimization
python=/home/USER/qwen36-api/.venv/bin/python
profile="${1:?profile}"; context="${2:?context}"; length="${3:?length}"
for unit in qwen36-inference qwen36-experiment; do
  if systemctl --user is-active --quiet "$unit"; then
    echo "Refusing to take over active $unit" >&2; exit 73
  fi
done
bash "$root/start-experiment.sh" "$profile"
trap 'systemctl --user stop qwen36-experiment' EXIT
"$python" "$root/check-experiment.py" 300
cp "$root/active-run.json" "$root/reports/context-$profile-run.json"
"$python" "$root/quality.py" --name "functional-$profile" --long "$length" > "$root/reports/functional-$profile.console.log"
"$python" "$root/boundary.py" --context "$context" --name "boundary-$profile" > "$root/reports/boundary-$profile.console.log"
"$python" "$root/test-backend-v2.py" --context "$context" --name "backend-$profile"
"$python" "$root/test-prefill-cancel.py" --length "$length" --name "prefill-cancel-$profile"
echo "Completed context gates: $profile"
