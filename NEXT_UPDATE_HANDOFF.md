# Next-update handoff — T3/T5/T6, 2026-10-15

Branch: `agent/cloud-next-update`. Fetched and fast-forwarded the owner's five new
commits to **31c10060375d4bd068e14f22bdf3f918cbc18f2b** before implementation.
Prior cloud work through 7c30166 and all owner changes are preserved. No main
update, force push, merge commit, upstream PR, service/deployment operation,
production upgrade or model download.

## New deliverables

| Task | Result |
|---|---|
| T3 | Real default-off same-layer gather/shared overlap; producer fence, join before routed GEMM, exception cleanup, pre-created capture streams/events; CPU control-flow/mutant tests and dedicated owner CUDA replay/stress test. |
| T5 | Repository-wide source inventory/syntax checks plus subsystem review, ranked confirmed findings and separate GPU hypotheses in REVIEW.md; repeatable CPU reproductions; 184 additional cache/daemon tests integrated. This is not a proof that every possible bug/race was found. |
| T6 | MIGRATION.md: exact public target versions/wheel hashes, driver-family assessment, source and compiler diffs, 11 textual conflict files, semantic cache conflicts, rebase order and isolated owner validation plan. Analysis only, no stack upgrade. |

Flag **QWEN36_DECODE_SHARED_OVERLAP=1**, profile field `decode_shared_overlap`.
Unset/0 uses the serial path. Eligible: Qwen GPU offload, one decode token, TP1,
pinned banks/fused copy layout. Unsupported modes remain serial. Configured refill
disables overlap before capture, regardless of whether a refill job is active.
The wrapper clears inherited env. Unknown raw values fail. Qualified=true is
explicitly refused for T3; no old receipt can qualify it accidentally.

Unqualified configs:

- `harness/main-release-overlap.json`: T3 alone on main-release.
- `harness/main-release-shortroute-fla-wu-overlap.json`: T1+T2-wu+T3.

Released profiles/production dependency locks/f1 historical results are unchanged.
The +3-4% decode estimate is historical, not a result from this implementation.
For 256K, read `next-update/long256-analysis/` and MIGRATION.md's final section:
T3 targets decode; it does not remove the dominant long-prefill attention work.

## Commits and cherry-pick/revert

Apply on the fetched owner head 31c1006, in this order (plus the final handoff
commit shown by `git log -1`):

1. `f7e3ac2` — T3 runtime, config, CPU/GPU tests and runbook.
2. `eca9d92` — T5 review, reproduction/inventory helpers, expanded CPU suites.
3. `abeb3e5` — T6 migration study and pinned source/metadata evidence.
4. `7aab36a` — T3 exporter/qualification mapping integration and regressions.

T3 requires **both f7e3ac2 and 7aab36a**. To remove T3, first disable the flag/use
main-release; if reverting code, revert 7aab36a then f7e3ac2. T5/T6 are independent
of runtime enablement; T5's requirements are test-only. Revert dependent handoff
text if removing its referenced commits. Do not revert the prior evidence gates
to accept an invalid report. Do not cherry-pick the earlier delivered commits
again: they already exist in the owner's branch.

## Validation and remaining blockers

- Complete mandatory CPU: **32 suites, 892 passed/executed, 0 failed, 3 skipped**;
  also passed from `/` with a physical checkout/log path containing spaces.
- T3 optimized Python: **10 tests passed**, including exception paths and missing
  producer/consumer wait mutants. Explicit env/profile checks remain active.
- Syntax/import scope, red runs and exact commands: CPU_VALIDATION.md;
  machine-readable final summaries: `validation/cpu-t3-*.json`.
- **NOT_RUN:** CUDA compile/SASS (no nvcc/cuobjdump), GPU/model/capture replay,
  private teacher fixtures and Windows/WSL. No GPU bit-exactness, release readiness
  or speed claim. T3 implementation is complete as an experimental candidate;
  target execution is still an owner gate.
- T5's remaining release blocker: **H1 functional reports are not yet run-bound**.
  Further confirmed findings: FTW index validation, legacy -O budget asserts,
  site paths/export basis/frozen final driver, 250K profiling/teacher coverage,
  and a T1 post-borrow test gap. REVIEW.md includes conditions/reproductions;
  no device race is reported as confirmed without evidence.
- T6 target: Torch 2.14.1 / Triton 3.8.0 / sglang-kernel 0.4.9 / flashlib 0.4.0,
  retaining FlashInfer 0.6.18.post1. Driver 617.14 meets the published CUDA13
  driver-family floor, but loading/execution/ABI and compiler numerics need GPU
  validation. Rebase and new-stack qualification are not performed by this study.

## Owner's first commands

Cloud push to the existing `agent/cloud-next-update` branch was rejected with
**HTTP 403**, so the delivery was a ZIP with the complete branch bundle,
incremental patches from 31c1006 and exact committed sources. The owner reviewed
it and published the branch on 2026-10-15. To restore the bundle offline instead:

```bash
git clone -b agent/cloud-next-update /absolute/path/qwen36-t3-t5-t6.bundle qwen36-review
cd qwen36-review
git remote set-url origin https://github.com/<owner>/qwen36-moe-16gb.git
git log -6 --oneline
```

With the owner's working write credentials, publish using
`git push -u origin agent/cloud-next-update` (never force). If the remote advances,
fetch and inspect it before integrating; do not overwrite those changes.
After delivery has been published, an existing clone can instead use:

```bash
git switch agent/cloud-next-update
git pull --ff-only origin agent/cloud-next-update
```

In either restored checkout, run:

```bash
QWEN36_CPU_PYTHON=/absolute/path/to/cpu-python \
  bash harness/CPU-checks.sh --output /absolute/path/to/new-cpu-logs
```

Then follow GPU_VALIDATION.md's read-only preflight and site configuration.
Under the existing exclusive lock and with only the experiment unit controlled,
run the new component first:

```bash
flock -w 3600 -E 75 "$LOCK" timeout 1800 "$P" -m pytest -v \
  "$C/freetoken/tests/moe/test_decode_overlap_gpu.py"
```

All four cases must execute without skips. Validate default-off -> T1 -> T2 ->
T3 independently -> combined, then run-bound teacher with the clean-cache case
mapping, prefix/cancel/long256 request coverage and unprofiled five-pair rotating
A/B. Close H1 before any release acceptance decision. MIGRATION.md's new-stack
reference is a separate owner opt-in; it never overwrites the old reference.
