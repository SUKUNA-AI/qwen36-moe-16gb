#!/bin/bash
# run_py.sh <script.py> args... : venv python, isolated, cwd /tmp
cd /tmp
exec /home/USER/qwen36-api/.venv/bin/python -I "$@"
