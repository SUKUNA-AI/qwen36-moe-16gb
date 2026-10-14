#!/bin/bash
# One GPU session (<= 15 min) for the PCIe H2D ceiling. Run via: flock -w 3600 -E 75 <lock> bash gpu_ceiling.sh
set -uo pipefail
D=/path/to/qwen36-workspace/work/next-update/gather-prefetch
export LD_LIBRARY_PATH=/home/USER/qwen36-api/cuda-13.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
if systemctl --user is-active --quiet qwen36-inference.service || systemctl --user is-active --quiet qwen36-experiment.service; then
  echo "model unit active; refusing" >&2; exit 73
fi
t0=$(date +%s)
snap() { { echo "== $1 $(date -u +%T)"; nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max --format=csv,noheader; uptime; } >> $D/logs/ceiling-activity.txt 2>&1; }
snap start
TMA=""
for C in 8 35 70 140; do TMA="$TMA,tma:$C:1:4:16384,tma:$C:1:16:4096,tma:$C:4:4:4096"; done
TMA="tma:8:1:3:16384${TMA},tma:4:1:4:16384,tma:16:1:4:16384,tma:16:2:4:8192"
LDG="ldg:70:1:plain,ldg:140:1:plain,ldg:280:1:plain,ldg:70:4:plain,ldg:140:4:plain,ldg:280:4:plain,ldg:16:4:plain,ldg:140:4:nc"
timeout 780 $D/build/ceiling_bench --kinds thp,hostalloc,4k,thp --footprint-mb 4096 --flat-mb 64 --ns 1,4,8 --reps 12 \
  --sm "$TMA,$LDG" --out $D/results/ceiling-s1.json 2> $D/logs/ceiling-s1.log
rc=$?
snap end
echo "ceiling rc=$rc wall=$(( $(date +%s) - t0 ))s" | tee -a $D/logs/ceiling-s1.log >&2
exit $rc
