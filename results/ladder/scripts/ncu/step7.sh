# step7.sh VARIANT...: Nsight Compute in eager decode (--cuda-graph-max-bs 0): six kernels per variant, <=15 min per capture,
# stops at the first failed capture (diagnosis only, no further attempts)
. "$(dirname "$0")/env.sh"; set -uo pipefail
D=$(dirname "$0"); mkdir -p "$L/ncu2"; install -d -m 700 "$L/private"
NC=$N/tools/ncu/ncu
for v in "$@"; do
  T=$L/private/$v.targets.json; EV=$v-eager
  python3 - "$N/config/$v.json" "$N/config/$EV.json" <<'PY'
import json, os, sys
c = json.load(open(sys.argv[1])); c['flags'] = list(c.get('flags', [])) + ['--cuda-graph-max-bs', '0']
fd = os.open(sys.argv[2], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, 'w') as f: json.dump(c, f)
PY
  python3 "$D/ncu_pick2.py" "$T" > "$L/private/$v.ncu-picks2.txt"
  while read -r label scope rank cat; do
    [ -n "${rank:-}" ] || continue
    bash "$D/prestep.sh" "ncu-eager $v $label" || exit 73
    out=$(CUDA_VISIBLE_DEVICES= "$P" -I "$N/profile-request.py" ncu-config --variant "$EV" --targets "$T" --rank "$rank" --scope "$scope" --nvtx-phase --count 1 --write 2>&1 | tail -1)
    cfg=$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["config"])' "$out" 2>/dev/null) || { echo "ncu-config failed $v $label: ${out:0:160}"; exit 1; }
    python3 - "$N/config/$cfg.json" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c['ncu']['sections'] = ['SpeedOfLight', 'MemoryWorkloadAnalysis', 'ComputeWorkloadAnalysis', 'Occupancy', 'LaunchStats']
c['ncu']['metrics'] = c['ncu']['metrics'] + ['sm__pipe_tensor_cycles_active.avg', 'sm__cycles_elapsed.avg']
json.dump(c, open(p, 'w'))
PY
    echo "=== ncu-eager $v $label ($cat, $scope rank $rank) -> $cfg $(date -u +%FT%TZ)"
    export cfg
    flock -w 7200 -E 75 "$LOCK" timeout 900 bash -uo pipefail <<'SH'; rc=$?
! systemctl --user is-active --quiet qwen36-inference || exit 73
! systemctl --user is-active --quiet qwen36-experiment || exit 73
trap 'systemctl --user stop qwen36-experiment.service 2>/dev/null' EXIT
bash "$N/start-experiment.sh" "$cfg" || exit 70
"$P" "$N/profile-request.py" run --profile "$cfg" --wait-ready 600 --wait-exit 600
SH
    systemctl --user stop qwen36-experiment.service 2>/dev/null
    base=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["output_base"])' "$N/reports/profile-final-current.json")
    echo "capture rc=$rc base=$base"
    d=$N/reports/$base.details.csv r=$N/reports/$base.raw.csv
    if [ -f "$d" ]; then
      row=$(python3 "$D/ncu_parse.py" "$d" "$r" "$C/harness/analyze-model-nsys.py" 2>&1 | tail -1)
      echo "{\"variant\":\"$v\",\"label\":\"$label\",\"scope\":\"$scope\",\"rank\":$rank,\"target_category\":\"$cat\",\"parsed\":$row}" >> "$L/ncu2/$v.jsonl"
    else
      echo "NO REPORT: stop for diagnosis"; journalctl --user -u qwen36-experiment --since "-16min" --no-pager -o cat 2>/dev/null | tail -n 12 | cut -c1-220; ps -eo pid,etime,args | grep -E "ncu|experiment" | grep -v grep | head -3 | cut -c1-160; exit 2
    fi
  done < "$L/private/$v.ncu-picks2.txt"
done
echo "step7 done $(date -u +%FT%TZ)"
