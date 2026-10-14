# Tasks for a remote coding agent

This repository serves Qwen3.6-35B-A3B (NVFP4, 256 experts, top-8) on a single 16 GB RTX 5070 Ti (SM120) through a FreeToken fork with expert offload. Release f1 is in production. Its quality guarantee is **bit-exact teacher forcing**: every optimization reproduces the reference logits exactly (125/125 comparisons, max_abs 0).

You can read, write and compile code. You almost certainly **cannot** run the model or our kernels on the target GPU: no SM120 device is in the cloud. GPU validation happens later on the owner's machine with the harness in `harness/`. Work so that this later step is quick and mechanical.

## Hard rules

1. **Bit-exactness is not negotiable.** A change that can alter any computed value counts as a numerics change. It must be (a) disabled by default behind a new env flag, and (b) explicitly marked "needs teacher forcing" in your notes. Transport-only changes (which bytes move where, overlap, scheduling) must provably leave every kernel's inputs byte-identical. Explain why in the commit message.
2. **Never change the released profiles** (`config/profiles.json`: main128, long256, baseline64) or their default behaviour. New behaviour goes behind new env flags whose default reproduces today exactly.
3. **sm_120 specifics:**
   - Never use `cp.async.bulk.shared::cluster` destinations. On sm_120 ptxas lowers them to a driver syscall that permanently grows the context stack, costing ~1.37 GiB of VRAM. Use `.shared::cta`.
   - Compile for `sm_120a` with CUDA 13.x. Check `cuobjdump -sass` for `CALL.ABS` / `__cuda_syscall` and report any you find.
4. **No secrets, hosts or personal paths.** Deployment-specific values live in a gitignored `local/` directory that you do not have. Use the placeholders already in `serving/` and `config/`.
5. **One task per branch.** Name it `agent/<task-id>`, give it a short description, and make the commits easy to review. Do not touch unrelated files. Do not reformat existing code.
6. **Tests for everything you can run:** CPU unit tests (pytest style, as in `harness/` and `freetoken/` tests), CPU reference implementations for kernels, and compile-only checks for CUDA. Add a `GPU_VALIDATION.md` to each branch with the exact commands the owner must run on the RTX 5070 Ti: the byte-exact component test, the teacher gate, and the A/B.

## Tasks, in priority order

### T1. Short-prompt route (hit-d2d)
`docs/ROADMAP.md` and `results/` (short-route) describe the change and the measured gains: 64-token TTFT 0.565 → 0.446 s, edits −0.12 s, 2K+ unchanged. The patch exists (`patches/` or `docs/ROADMAP.md` references). Integrate it cleanly:
- env `QWEN36_LAYER_MAJOR_MIN_TOKENS` (unset = current behaviour) together with `--moe-prefill-hit-d2d`;
- relax the `profile-runtime` rule that refuses layer-major with hit-d2d, only when the threshold is set;
- add the profile switch key with its validation, plus CPU tests.

### T2. Enable fused FLA as an optional profile switch
The fused CUDA FLA path (`QWEN36_CUDA_GDN_FLA=wu`) is bit-exact and already in the tree. Measured effect: TTFT −3% at 16K, −0.9% at 120K. Add it as a profile switch (`cuda_gdn_fla`) end to end: profile-runtime table, release tooling evidence list, tests. Default off.

### T3. Overlap the decode expert gather with the shared expert
The decode gather already runs at the PCIe ceiling (~42 GB/s; this WSL2 host tops out at 41–45 GB/s), so the only remaining gain is to hide it. Issue the routed experts' gather on a side stream and run the shared-expert MLP of the same layer meanwhile. Join with an event before the routed-expert compute, and capture the whole thing in the decode CUDA graph.
- Byte-identical by construction: the same kernels see the same inputs.
- Expected: +3–4% decode tokens/s.
- Deliver: the change behind an env flag, a CPU-side graph/ordering test where possible, and the GPU validation steps.

### T4. Harness robustness
- `compare-quality.py` selects the last `teacher.jsonl` row per profile and case without binding it to the run id. Bind it to the report's run id(s), including the clean-cache restart run.
- `teacher.jsonl` rotates at 2 MiB. Make the reference lookup read rotated files, or keep the reference rows in a separate file.
- `profile-request.py run` treats `/v1/models` 200 as ready. Wait for `/health` like `check-experiment.py` does.
- Add a regression unit test that asserts the stream waits in `scheduler.py` (the commit/decode ordering verified in `results/`).

### T5. Review
Read the whole repository and report bugs, races, unsafe assumptions and missing tests in `REVIEW.md` on its own branch. Rank them by severity, with `file:line` references. Pay special attention to:
- offload cache eviction under memory pressure;
- the layer-major window logic;
- CUDA-graph capture/replay invariants;
- the guard's request validation;
- the release/qualification receipt checks.

### T6. Stack migration study (analysis only, no upgrade)
Today the stack is torch 2.11.0, Triton 3.6.0, FlashInfer 0.6.18.post1 and FreeToken 0.1.3 (base cac247a8), on driver 617.14. Upstream FreeToken now ships a newer stack, around torch 2.14 and Triton 3.8. Produce `MIGRATION.md` on its own branch:
- **Exact target versions** and whether their wheels run on driver 617.14 / CUDA 13.x. Installing new drivers is not allowed.
- **Upstream changes that touch code we replicate bit-exactly:**
  - the Triton NVFP4 MoE prefill kernel (our CUDA replica is in `kernels/nvfp4-moe`);
  - the FLA GDN chunk kernels (`kernels/gdn`);
  - the FlashInfer prefill/decode plan logic that our attention batching depends on;
  - the FP8 projections (`kernels/fp8-gemv`);
  - the decode gather call sites.

  For each, say whether its arithmetic order or plan semantics changed (diff the sources).
- **The FreeToken rebase:** which of our fork changes (`CHANGES` vs upstream) conflict with the new upstream, with a proposed rebase order.
- **A validation plan for the owner's GPU:**
  - a new teacher-bf128 reference on the new stack;
  - our component bit-exact suites against the new kernels;
  - the release teacher gate;
  - a paired A/B of original FreeToken (old stack vs new stack) to see what the new stack alone gains.

## Not in scope
FP8 KV cache (rejected for now), MTP/speculative decoding, vision, the RMSNorm (1+w) fp32 change, and refill/sync-lean (measured gains below noise). Do not pursue them.

## What "done" looks like
Each branch has the code, the tests that pass in your environment, `GPU_VALIDATION.md`, and a short summary of expected gains and risks. The owner runs the GPU validation; only changes that pass teacher forcing (where applicable) and a paired A/B get into a release.
