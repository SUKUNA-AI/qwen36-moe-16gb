#!/usr/bin/env bash
set -euo pipefail
umask 077
root=/home/USER/qwen36-api
source=/path/to/qwen36-workspace/work/FreeToken-opt
reports="$root/optimization/reports/gdn-snapshot"
mkdir -p "$reports"
chmod 700 "$reports"
export PYTHONPATH="$source/python"
export CUDA_HOME="$root/cuda-13.4"
export PATH="$CUDA_HOME/bin:$root/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:/usr/lib/wsl/lib:${LIBRARY_PATH:-}"
export TRITON_PTXAS_PATH="$root/cuda-13.4/bin/ptxas"
export QWEN36_ENGINE_METRICS=0
export QWEN36_PROFILE_HELPERS=1
mode="${1:?systems or compute}"
arm="${2:?bf16 or fp32}"
case "$arm" in bf16|fp32) ;; *) exit 2;; esac
if [ "$mode" = systems ]; then
  "$root/optimization/tools/target-linux-x64/nsys" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \
    --wait=primary --force-overwrite=true --output="$reports/nsys-$arm" \
    "$root/.venv/bin/python" "$source/tests/kernels/test_gdn_fp32_snapshot.py" --profile "$arm"
  "$root/optimization/tools/target-linux-x64/nsys" stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
    --format csv --output "$reports/nsys-$arm-summary" "$reports/nsys-$arm.nsys-rep"
elif [ "$mode" = compute ]; then
  "$root/optimization/tools/ncu/ncu" --target-processes all --profile-from-start off --set full \
    --section-folder "$root/optimization/sections" --launch-count 1 \
    --kernel-name 'regex:chunk_gated_delta_rule_fwd_kernel_h_blockdim64' --force-overwrite \
    -o "$reports/ncu-$arm" \
    "$root/.venv/bin/python" "$source/tests/kernels/test_gdn_fp32_snapshot.py" --profile "$arm"
  report="$reports/ncu-$arm.ncu-rep"
  if [ ! -f "$report" ]; then report="$reports/ncu-$arm.ncu-repz"; fi
  "$root/optimization/tools/ncu/ncu" --import "$report" --csv --page details > "$reports/ncu-$arm.csv"
else
  exit 2
fi
find "$reports" -maxdepth 1 -type f -name "*-$arm*" -exec chmod 600 {} +
