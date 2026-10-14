#!/bin/bash
for p in $(fuser /home/USER/qwen36-api/run/gpu-exclusive.lock 2>/dev/null); do ps -o pid,etime,args -p $p | tail -1 | cut -c1-200; done
systemctl --user is-active qwen36-experiment; systemctl --user show -p ExecStart qwen36-experiment | cut -c1-200
