<!-- READY, NOT POSTED. Target: FlashML-org/FreeToken, new issue (feature proposal). Paste from the "# " title line down. Checked against upstream main 4d2e2e2 (2026-10-19). -->

# [Feature] Testing/CI: a teacher-forced, bit-exact logit-reference gate for kernel changes (plus optional per-request engine telemetry)

## Summary

All of our kernel and scheduler changes were accepted only if a teacher-forced run reproduced a pinned reference
**byte for byte**. The engine replaces the sampled token with a fixture's target token and records per-position NLL.
At fixed positions it also dumps the full logit vector, which is compared bitwise with a pinned file. Upstream has
nothing equivalent today, and we propose it as a testing/CI feature.

## Problem in upstream today

- No teacher forcing or logit-reference path exists on main 4d2e2e2 (no match for teacher/forced-token/logit
  reference under `python/` or `tests/`). The model-level gate is `tests/e2e/test_aime.py`, an accuracy check that
  cannot see a 1-ulp change. `logprobs` is not supported (`server/openai_api.py:658-659`).
- Kernel tests compare against tolerances. That cannot show that a "drop-in" kernel leaves model outputs unchanged.

## What we changed

- `freetoken/python/freetoken/engine/qualification.py`, called from `engine.py` right before sampling:
  - with `QWEN36_TEACHER_FIXTURE=<json>`, a batch-1 request whose `seed` matches a fixture case is forced to emit the
    case's target IDs;
  - per-position NLL is computed as `logsumexp(row) - row[target]`;
  - with `QWEN36_LOGIT_REFERENCE=<dir>`, the fp32 logit row of every 8th forced position is saved as `.npy`.
- The reference profile writes the logit files only with an explicit opt-in (`QWEN36_TEACHER_CREATE_REFERENCE=1`),
  never overwrites one, and checks its own run against the pinned bytes.
- A candidate profile is compared per checkpoint: `max_abs`, relative L2, finiteness and `byte_equal`. The fixture
  sha256, target sha256 and reference file digests are bound into the result.
- It runs only under an explicit allowlist of experiment profile names, so a serving profile can never be
  teacher-forced. It syncs with the CPU on purpose, so speed runs exclude it.
- Results hold only losses and numeric differences, never text.
- `engine/telemetry.py` (`QWEN36_ENGINE_METRICS`, default on): CUDA events around each forward, and per-request rows
  (prefill/decode GPU ms, forwarded tokens, expert-cache statistics and prefill H2D/D2D byte deltas, PyTorch allocator
  peaks, KV bytes, cancellation). The rows are
  written by a bounded background writer that drops rows rather than block. Related: #76, #552.

## Evidence

- The gate (`docs/METHODOLOGY.md` §1): 20 cases, 943 forced tokens, 125 logit checkpoints (contexts up to 120K).
  A change claimed bit-exact passes only with 125/125 at `max_abs` 0 and no new functional failure.
- It separated real from apparent changes. Every one of 13 ladder configurations passed 125/125 at `max_abs` 0
  (`results/ladder/teacher.json`).
- On our 256K profile one case showed `max_abs` 1.232. We traced it to prefix-cache history, not kernels: it
  reproduced exactly with the same request history and vanished on a clean cache (`docs/RESULTS.md` §2).
- It also measured a deliberate numeric change: keeping NVFP4 global scales in FP32 shifted teacher PPL by −1.13%
  (95% CI [−2.27%, −0.08%]).

## Limitations / not measured

- Batch 1 only. Fixtures are keyed by request `seed` (which needs our seed patch, cf. #693), and the profile
  allowlist is hard-coded.
- The telemetry writer currently imports a helper from our serving layer, so it would need a FreeToken-native sink.
- Telemetry overhead was not measured separately.
- References are only valid for one exact stack (GPU, driver, torch/Triton versions). A toolchain bump needs a new
  reference.

## Proposal

We could contribute:

1. a generic `--teacher-fixture` / `--logit-reference` engine option (env or CLI), with a small public synthetic
   fixture;
2. a pytest `needs_weights` gate that runs it against a pinned reference for a given checkpoint.

Questions: is a GPU- and stack-specific bitwise reference acceptable for CI, or should the gate default to a
tolerance with bitwise as an opt-in? Should telemetry be a separate issue?
