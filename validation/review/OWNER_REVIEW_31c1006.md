# Historical owner review, preserved from fetched 31c1006

## Owner-side review, 2026-10-15

Review of the delivered bundle (base `7c78a48`, cloud head `7c30166`) before any
GPU work. The cloud CPU result was reproduced in an isolated Linux venv from
`harness/requirements-cpu.txt`: 30 suites, 695 passed, 0 failed, 2 archive-only
skips. After the fixes below: 30 suites, 698 passed, 0 failed, the same 2 skips.
No GPU, model, private fixture or Windows test was run.

### Verdict per item

| Item | Verdict |
|---|---|
| T1 short route | Keep. With `QWEN36_LAYER_MAJOR_MIN_TOKENS` unset the released profiles take exactly main's path (traced: same calls, order and GPU work). Routing matches the archived measured patch (`input_len - cached_len`). Hit-D2D cannot read borrowed slots: epochs force it off and borrowed slots stay unmapped afterwards. |
| T2 fused FLA switch | Keep. Evidence binding is sound. Raw engine values outside unset/0/1/wu/ho, or FLA without CUDA GDN, now fail instead of being ignored. Released profiles never set them. |
| T4 run-bound evidence | Keep the model (identity, rotation reading, snapshots, complete functional sets, byte equality, one gate for make-qualification and qcheck, `/health` wait, stream-ordering test). It needed the fixes below to be usable. |
| P0 CPU runner, P2 strict provenance, code worker | Keep. All 83 manifest modules pass the strict recipe check, so `FREETOKEN_STRICT_PROVENANCE=1` is redundant with the wrapper's inventory check, not a new start risk. Any change to `profile-runtime.py` or the runtime tree needs a new release receipt anyway. |
| T3 | Deferred: design note only, nothing to run. |
| T5 | Partial: this file covers the changed paths, not a whole-repository audit. |
| T6 | Not delivered: there is no `MIGRATION.md`. |

### Fixed on this branch

- `15a8166` T1: hit-D2D mode opened the epoch at a second call site (after the FLA
  cache, Nsight and timing event), with cleanup that differed from `close()`. It
  is back to the single, measured call site.
- `cee127b` profiles: `layer_major_min_tokens` is capped at chunk + 1 (a larger
  threshold routes multi-chunk prompts chunked, measured slower).
  `layer_major_attn_batch` now accepts only 1-4, the values `attn_batch()` honours.
  Before, 5-64 validated and silently ran k=1; the cloud long256 analysis found this.
- `4e8d704` T4:
  - A reference can be created again (explicit `QWEN36_TEACHER_CREATE_REFERENCE=1`, exclusive create, never replaced). Without it, T6 and new fixtures were impossible.
  - Engine state binds before it is assigned.
  - A conflicting duplicate in `teacher.jsonl` fails only the reports that select it; damaged lines are listed instead of failing every comparison.
  - `qualify.py` requires schema-2 evidence arguments and checks evidence and report names before starting a unit.
  - `quality.py` refuses an existing report name before the GPU pass, and its log poll retries partial lines.

### Still open (not blocking the T1/T2 GPU gates)

1. **PPL-only comparison.** `bind_teacher` requires logit comparisons, and
   `compare-quality.py` exits 1 unless the result is bit-exact. A deliberately
   non-bit-exact arm therefore gets no usable report: the T6 stack A/B and any profile
   without `QWEN36_LOGIT_REFERENCE`. A PPL-only mode is needed before T6.
2. **`gates/final-gates.sh` is not schema-2 ready.** It calls `qualify.py` without
   evidence arguments, reuses f1 report names, and `extract-run --teacher` does not
   expect `QWEN36_TEACHER_REFERENCE_REPORT`. It stays the frozen f1 driver; the next
   release needs a new driver (see also MANIFEST TODO 7).
3. **Functional reports are not bound to a run.** `functional_rows` checks
   completeness but not the report's run profile or sources against the candidate.
4. **T1 GPU test gap.** `test_short_route_gpu.py` repopulates every slot per phase,
   so it cannot catch stale borrowed bytes after an epoch. A phase that borrows,
   fills, ends the epoch and then runs hit-D2D without repopulating is still missing.
5. The archived layer-major CPU suite (24 tests) and the short-route toy suite are
   archive-only and were not run against this tree.
6. A NaN in an engine row makes the telemetry writer drop it (`allow_nan=False`), so
   `quality.py` reports "rows missing" without a cause. It fails closed.
7. Process: the delivery bundles T1, T2 and T4 in one branch, against rule 5
   (one task per branch). The commits are separable; `NEXT_UPDATE_HANDOFF.md` gives
   the order.
