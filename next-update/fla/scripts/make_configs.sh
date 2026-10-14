#!/bin/bash
set -euo pipefail
umask 077
cd /tmp
nice -n 19 /home/USER/qwen36-api/.venv/bin/python -I /path/to/qwen36-workspace/work/next-update/fla/scripts/make_configs.py
grep -c "" /home/USER/qwen36-api/optimization/config/profile-final-fla-on.json
