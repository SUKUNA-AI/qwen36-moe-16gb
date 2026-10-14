#!/bin/bash
set -u
N=/home/USER/qwen36-api/optimization
echo "host: $(hostname) $(date -u +%FT%TZ)"
ls -la /home/USER/qwen36-api/run/ 2>&1 | head -20
systemctl --user is-active qwen36-inference qwen36-experiment 2>&1
ls -la $N/reports/ | grep -i "model-final-main-release" | head
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv 2>&1
free -g
nproc
ls /home/USER/qwen36-api/cuda-13.4/bin | head -30
