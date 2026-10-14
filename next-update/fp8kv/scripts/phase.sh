#!/usr/bin/env bash
ls -la --time-style=+%FT%T /home/USER/qwen36-api/run/ | grep -E "phase|lock"; date -u +%FT%TZ
systemctl --user is-active qwen36-experiment qwen36-inference 2>&1 | tr '\n' ' '; echo
