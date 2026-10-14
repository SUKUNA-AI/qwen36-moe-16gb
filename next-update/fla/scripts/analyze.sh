#!/bin/bash
# CPU only: analyse the two captures named in results/nsys-base-{on,off}.txt
set -uo pipefail
D=/path/to/qwen36-workspace/work/next-update/fla
N=/home/USER/qwen36-api/optimization
PY=/home/USER/qwen36-api/.venv/bin/python
cd /tmp
for arm in on off; do
  base=$(cat $D/results/nsys-base-$arm.txt)
  [ -f "$N/reports/$base.sqlite" ] || { echo "missing sqlite for $arm ($base)"; continue; }
  nice -n 19 $PY -I $D/scripts/fla_nsys.py "$N/reports/$base.sqlite" --label fla-$arm --out $D/results/nsys-fla-$arm.json
  cp "$N/reports/$base.run-receipt.json" "$D/results/$base.run-receipt.json" 2>/dev/null || true
  cp "$N/reports/$base.profile-receipt.json" "$D/results/$base.profile-receipt.json" 2>/dev/null || true
done
nice -n 19 $PY -I $D/scripts/compare_arms.py $D/results/nsys-fla-on.json $D/results/nsys-fla-off.json $D/results/nsys-compare.json > /dev/null && echo COMPARE_OK
