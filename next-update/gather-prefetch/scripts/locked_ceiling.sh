#!/bin/bash
flock -w 3600 -E 75 /home/USER/qwen36-api/run/gpu-exclusive.lock bash /path/to/qwen36-workspace/work/next-update/gather-prefetch/scripts/gpu_ceiling.sh
rc=$?
echo "locked rc=$rc" >&2
exit $rc
