# Methodology

This document describes how f1 was qualified. The rules were fixed **before** the runs they judge.
Where a rule changed during the project, the change and its reason are recorded here.

## 1. Quality: bit-exact teacher forcing

**Reference.** `teacher-bf128` is upstream FreeToken mechanics at 131072 context: Triton NVFP4 MoE,
chunk 2048, BF16 KV, no f1 switches, with FP32 NVFP4 global scales. The reference was regenerated once,
when the FP32-global fix went in. The old FP16-global reference was kept, to measure that basis shift
(§1.3).

**Fixture.** 20 fixed cases:
- short Russian/English chat;
- code;
- tool calling;
- long documents at 4K, 16K and 120K tokens.

They contain 943 forced target tokens and 125 logit checkpoints, which are complete, finite logit
vectors at fixed positions.

**Procedure** (`qualify.py` / `quality.py --teacher`):
1. Start the candidate profile on a fresh experiment server. Its name must be on an explicit
   teacher-forcing allowlist in `engine/qualification.py`, so a production profile can never be
   teacher-forced by accident.
2. Feed every case with the reference's target tokens forced. Record per-position NLL and the logits at
   the checkpoints.
3. Compare with the reference:
   - max |Δlogit| and max relative L2 per checkpoint;
   - the per-case NLL lists;
   - relative perplexity growth with a bootstrap CI.
4. Run the 5 functional cases with field-level checks: russian, executable-code, tool-cycle,
   long-facts-16384 and long-facts-120000.

**Pass rule for a change claimed bit-exact:**
- 125/125 checkpoints with max_abs = 0;
- forced targets complete;
- PPL gate pass;
- no **new** functional failure vs the reference.

Every f1 component passed this rule individually, and the final combined build passed it again.

### 1.1 Clean-cache 120K case (harness v7)

**Why.** The prefix cache dedups the shared chat header across requests ([ARCHITECTURE.md](ARCHITECTURE.md)
§5.1). Whether long-120000 sees a cold header therefore depends on what ran before it and on whether
admitting 120K tokens evicts the header nodes, which in turn depends on the KV pool size. So on the
256K profile that case differed from the 128K reference only because of cache history.

**The fix.**
- `quality.py --teacher` restarts the experiment unit with the same profile right before long-120000.
- It fails loudly unless:
  - every selected row comes from this run;
  - the restarted server served only long-120000.
- The engine flush endpoint exists, but the API guard blocks it, so a restart is used instead. It costs
  ~2-2.5 min per run.

**Reference validity.** The reference's own long-120000 row is bit-identical to 3 certainly-cold runs.
With v7, both main-release and long256-release give 125/125 at 0.0.

### 1.2 Interim 256K rule (PASS_EQUIV_R1B256)

Before v7 existed, the release gate accepted long256 under a pre-stated no-weakening rule. All of these
had to hold:
- quality, PPL and forced-target gates pass, with no new functional failures;
- the **only** non-zero case is long-120000;
- its max_abs, max_rel_l2, per-case NLL deltas and PPL growth **equal the previously qualified 256K
  profile's to the last digit**.

The equivalence held. GPU confirmations C (clean cache gives 0.0) and D (128K reproduces the deviation
under the same history) then established the cause.

### 1.3 Basis decision: FP32 vs FP16 global scales

This was decided by a pre-registered rule (frozen sha) on two inputs:
- teacher NLL vs the legacy FP16 reference;
- a private 60-case eval (12 each of code with sandboxed tests, Russian, tool use, agent history, and
  long documents at 16K/64K/120K).

The FP32 basis was to be chosen only if all three held:
1. PPL growth <= +0.1% with CI95 upper <= +1%;
2. FP32 passes >= FP16 passes - 1, with no category losing more than one;
3. no tool case that passes on FP16 fails on FP32.

Outcome: FP32 was chosen. PPL changed -1.13% (CI [-2.27%, -0.08%]), the eval scored 59/60 vs 57/60, and
there were no tool losses. The eval cases are private and not published.

### 1.4 Known limitations are compared, not waived

Two functional cases fail on the FP32 basis, both in the reference and in every candidate:
- executable-code;
- long-facts-16384 (the total field).

An independent high-precision CPU recomputation reproduces the greedy outputs byte for byte. Both are
first-token near-ties of the quantised checkpoint (margins 0.48 and 0.09 logit). The gate criterion
stays "no new functional failure vs the reference". The decoding contract and the prompts were not
changed.

## 2. Speed: paired matrix and the compare-ab rule

**Design.**
- 2 arms, the baseline (`bf128`) and the candidate, **5 pairs** with rotating arm order.
- One session, with an immutable manifest that binds source, config, libraries, launcher sha256 and
  run identity.
- Exactly 128 generated tokens per request; tool cycles run to natural EOS.
- Cases:
  - fresh 64 / 4K / 16K / 64K / 120K;
  - agent history continuation;
  - last-message edit at 16K;
  - tool-result edit at 16K;
  - tool-cycle task.
- No profiler is attached during speed runs.

**Rule (harness v6, compare-ab):**
- The metric is **full-request elapsed time** per case. TTFT and decode are reported, not gated (owner
  decision).
- For each case, take the log ratio candidate/baseline over the 5 pairs, then a t-interval with df = 4
  at 97.5%.
- **Non-inferiority:** lower bound >= 1/1.03 on every gating case. If the upper bound is below 1/1.03,
  the result is FAIL.
- **Gain:** at least one family member with a gain of >= 5%, Bonferroni-corrected with
  F = 9 x (profiles - 1).
- **Invalid arm:** min free VRAM below 0.5 GiB.
- Exit codes are 0 PASS / 1 FAIL / 2 INCONCLUSIVE.
- With 5 pairs the test is conservative on noisy short cases. Earlier candidates were recorded as
  INCONCLUSIVE and were not relabelled after the fact.

Each component also went through a smaller **paired A/B**: 3 alternating rounds, 60 s warm-up, medians
plus every round reported, outputs checked identical. This was used to decide what goes into the build.
The 5-pair matrix judges the release.

## 3. Soak

- **Run.** 7200 s on a fresh service start with the release identity, sequential requests, 256 output
  tokens.
- **Schedule.** Fresh 64 / 4K / 16K / 64K / 120K, prefix follow-up, arithmetic check, tool cycle and
  client cancel. Minimum coverage per 60-iteration block: 12 fresh per length, 12 prefix, 9 arithmetic,
  4 tool, 4 cancel.
- **Bindings.** Every request is joined to engine telemetry by run id and request UID. The service
  identity is re-checked periodically. Log rotation, gaps and drops are tracked; a single drop fails
  the run.
- **Memory.** Early/late host memory is compared per prompt length and at idle. More than 256 MiB of
  growth (the 268 MB flag) fails. Too few samples gives INCONCLUSIVE rather than PASS.
- **Environment.** `GLIBC_TUNABLES` is verified in the service process environment before and after
  the run.
- **Verdict.** PASS / FAIL / INCONCLUSIVE with reasons. The soak does not gate VRAM, but VRAM is
  sampled from the engine state and at 1 Hz per request.
- The original soak tool could produce a false PASS: an empty row set counted as a pass, and the cancel
  check closed on the first empty chunk. Both were fixed and covered by mutant tests before the f1 soak.

## 4. Context gates (C0 / L4 / F1)

One gate per profile, single runs, on the frozen release identity:
- **C0**: main128 at 131,072.
- **L4**: long256 at 262,144.
- **F1**: baseline64 at 65,536.

Each evaluation (`results/context-gates/*-evaluation.json`) checks:
- profile mechanics, source identity and native libraries equal to the matrix reference arm;
- the functional case set, with short-case output hashes identical to the reference and failures
  equal to the reference;
- the long fresh case passes and is truly uncached: 120K for C0, 250K for L4, 60K for F1;
- the boundary request: exactly `context` tokens (prompt + output) accepted and completed, +1 rejected
  with 400;
- backend contract checks, and cancel during prefill with a correct follow-up;
- placement: KV dtype BF16, expert slots valid, host experts fully pinned, GDN pool unchanged, KV
  capacity covering the context, fused gate on all 10 attention layers, snapshot flags matching env;
- the TTFT watchdog margin;
- VRAM class: **FLOOR 0.5 GiB** hard (below is BELOW_FLOOR, fail) and **TARGET 1.0 GiB**. FLOOR-band
  results are allowed only as PASS_WITH_KNOWN_LIMITATION.

The VRAM thresholds come from an owner decision. Free VRAM is NVML free, which includes Windows desktop
usage. Foreign usage is not refused at start, because WDDM shrinks it under pressure.

## 5. Freeze, sync and provenance

- **Workspace and native tree.** Code is edited in a workspace, then synced to the native (Linux)
  harness tree. A sha256 freeze file covers every synced file. Any mismatch stops a gate unless it is a
  **documented freeze exception**, which records the file and its new hash. f1 had exactly 6.
- **Kernel provenance.**
  - Every kernel module is bound to its source sha256 in a manifest (83 modules in f1).
  - Production sets `FREETOKEN_DISABLE_JIT=1`. A missing or unbound prebuilt is refused rather than
    compiled.
  - ctypes libraries are bound by sha256 and an ABI probe, and rehashed at release and at start.
- **Receipts.** Every run writes a receipt with:
  - source hash (`04dbae8b...` for f1);
  - service file hashes;
  - package versions (torch 2.11.0, triton 3.6.0, flashinfer-python 0.6.18.post1, sglang-kernel 0.4.5,
    transformers 5.16.1);
  - checkpoint revision;
  - config.

  Gates compare identities across steps. The release receipt (`5968b76f...`) is built only from passing
  evidence.
- **Install.** The release installer stages the qualified native tree byte-for-byte and writes a
  rollback manifest. A rollback check must pass before going live.

## 6. Noise handling

- **Where noise comes from.** The noise sits in the GPU pipeline, not in HTTP: engine GPU-event times
  vary as much as client wall time. Sources include:
  - CPU launch/sync gaps under contention;
  - placement of the WSL VM on P- or E-cores;
  - boost clocks;
  - the desktop and browser video sharing the GPU (WDDM time-slicing);
  - host RAM bandwidth for PCIe.
- **Measured noise.** CV 0.4-1.7% at 64K in a quiet session; 3-6% under CPU load; up to 30% between
  sessions.
- **Rules:**
  - Arms are compared only **paired, in the same session, in rotating/alternating order**. Absolute
    numbers from different sessions are labelled cross-session.
  - Sustained warm-up `QWEN36_BENCH_WARMUP_S=60`: repeated 16K prefills plus forced tokens before
    measuring.
  - Report medians **and** every round.
  - Background agents are niced (`nice -n 19`, `ionice -c3`) and stopped for the final matrix.
  - No clock locking. The machine stays in normal desktop use, by owner decision.
  - Never measure speed under a profiler. Never sum overlapped CUDA durations; use critical-path
    attribution.
  - A single-run "regression" is re-tested with a paired A/B before anything is decided. Two such cases
    turned out to be noise; one turned out to be real VRAM paging.
- **GPU arbitration.** All GPU work runs under one exclusive `flock`, with blocking waiters only and
  hard per-step timeouts. A watchdog flags an idle GPU, long lock holders or stuck profilers.
