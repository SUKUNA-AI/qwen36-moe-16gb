# Shared rules for next-update agents (2026-10-14 evening)

Context: release f1 of the home Qwen3.6-35B-A3B API (nvidia/Qwen3.6-35B-A3B-NVFP4, all 256 experts, top-k 8) is INSTALLED.
Production (qwen36-inference) is STOPPED and must stay stopped: never start/stop/restart it, never run `qwen36`.
Experiments use the systemd user unit qwen36-experiment via the native harness (start-experiment.sh <profile>, check-experiment.py 300).

Paths: workspace W=/path/to/qwen36-workspace (Windows C:\path\to\qwen36-workspace); native harness N=/home/USER/qwen36-api/optimization
(configs in $N/config, reports in $N/reports); venv python /home/USER/qwen36-api/.venv/bin/python; FreeToken workspace tree
$W/work/FreeToken-opt (synced to $N/python). STATE log: $W/outputs/qwen36-api/optimization/STATE.md (read it for history; do NOT write it).
Release run configs: $N/config/main-release.json (main128) and long256-release.json; reference teacher-bf128 (bf128.json).

Shell: Windows host. Run Linux work as `MSYS_NO_PATHCONV=1 wsl -d archlinux -u USER -- bash /mnt/<drive>/.../script.sh` and WRITE multi-line
commands into script files first (inline `bash -c '...$var...'` gets expanded by the WSL login shell). Windows `python` is a Store stub: never call it.

DO NOT EDIT: work/FreeToken-opt, outputs/, the native tree $N (hash-frozen, synced by the parent). Work in your own directory
work/next-update/<your-task>/ (copies, patches, scripts, results). Deliver patches + evidence; the parent applies and syncs.

GPU (single RTX 5070 Ti, shared):
- Every GPU session: `flock -w 3600 -E 75 /home/USER/qwen36-api/run/gpu-exclusive.lock <cmd>` (blocking wait), always stop
  qwen36-experiment in a trap when you started it, max 20 min per session (Nsight agent: max 40), then release the lock.
- Phase gating: only start GPU sessions when your phase marker exists: phase 1 = always; phase 2 = /home/USER/qwen36-api/run/phase1.done;
  phase 3 = /home/USER/qwen36-api/run/phase2.done. Poll every 60 s; do CPU work meanwhile.
- nsys/ncu: never longer than 15 min per capture, never measure speed under a profiler, never sum overlapped CUDA durations.
- Speed A/B: paired/rotating arms, QWEN36_BENCH_WARMUP_S=60 sustained warm-up, same session, report medians + every round.
- Quality: bit-exact teacher forcing (qualify.py) is the gate for any change claimed bit-exact; needs allowlist name teacher-<candidate>
  in engine/qualification.py (owner pre-approved adding names: list the names you need, the parent adds them).

Secrets: never read or print /home/USER/qwen36-api/secrets/* or $N/config/experiment.env; scripts may load keys in-process (bench.headers()).
No GitHub issues/PRs/posts: drafts only, local files. Report back with a compact final message: verdict, numbers, files, NOT_RUN.
