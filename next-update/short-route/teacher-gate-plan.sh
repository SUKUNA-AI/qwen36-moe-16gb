#!/bin/bash
# PARENT-ONLY run plan for the short-route candidate (NOT executed by the short-route agent).
# Candidate = main-release + --moe-prefill-hit-d2d + QWEN36_LAYER_MAJOR_MIN_TOKENS=2049 (patched tree, no shim).
# Allowlist name needed in engine/qualification.py: 'teacher-main-release-shortroute'.
set -euo pipefail
W=/path/to/qwen36-workspace
H=$W/work/next-update/short-route
N=/home/USER/qwen36-api/optimization

# 1. Apply the patch to the workspace tree (3 files; base sha256 in $H/orig must match), then the usual native sync.
#    cd $W/work/FreeToken-opt && patch -p1 < $H/shortroute.patch      # patch sha256 in the agent report
#    + add 'teacher-main-release-shortroute' to the allowlist tuple in python/freetoken/engine/qualification.py
#    + sync to $N/python (sync-v5.sh, parent procedure)
# 2. Install the candidate config (built from main-release.json by scripts/make_candidate.py 2049):
#    install -m 600 $H/configs/main-release-shortroute.json $N/config/main-release-shortroute.json
# 3. Teacher (bit-exact vs bf128 reference 125/125, max_abs 0) + functional + compare, under the GPU lock (~15-20 min):
#    flock -w 3600 -E 75 /home/USER/qwen36-api/run/gpu-exclusive.lock \
#      $N/../.venv/bin/python $N/qualify.py --candidate main-release-shortroute
# 4. Gate is NON-VACUOUS only if the hit/miss split actually ran during the teacher: in $N/reports/engine.jsonl rows of
#    the teacher run (run_id from active-run.json) require expert_counter_deltas.prefill_d2d_bytes > 0 on >= 1 short case
#    AND layer_major_prefill.prefills > 0 (long cases still layer-major). Journal must NOT contain
#    "hit-D2D requested but unavailable" nor "layer-major prefill not used: hit-D2D".
# 5. Release tooling: profile-runtime.py SWITCH_EXCLUDED_FLAGS forbids layer_major_prefill + --moe-prefill-hit-d2d;
#    allow it only together with QWEN36_LAYER_MAJOR_MIN_TOKENS (new tunable, e.g. 'layer_major_min_tokens', int >= 1).
# 6. Speed confirmation (optional): paired matrix main-release vs main-release-shortroute, 5 pairs, QWEN36_BENCH_WARMUP_S=60.
#    Expected (this agent, 2 sessions x 5 rounds): 1-chunk prompts / prefix hits / edits -0.10 s TTFT, >= 2 chunks unchanged.
echo "plan only; see comments"
