#!/bin/bash
ls /home/USER/qwen36-api/optimization/config | grep -c shortroute
systemctl --user show -p ExecStart qwen36-experiment | grep -c shortroute
sha256sum /path/to/qwen36-workspace/work/next-update/short-route/shortroute.patch /path/to/qwen36-workspace/work/next-update/short-route/configs/main-release-shortroute.json | cut -c1-16
