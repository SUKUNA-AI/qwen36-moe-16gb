# T5 review — current findings and coverage

Reviewed fresh owner branch `31c10060375d4bd068e14f22bdf3f918cbc18f2b` plus T3
`f7e3ac2`. This review covers repository subsystems beyond the prior patch. The
full tracked text inventory is `validation/review/inventory.json`: 1,168 files,
including 762 Python and 94 shell scripts, with hashes, line counts and indexed
stream/native-load/process/receipt sites. All Python parsed; all shell text passed
`bash -n` after CRLF normalization. Inventory scans are not manual line-by-line
verification, a security audit certification, or evidence that GPU races are absent.

The prior owner review is preserved verbatim in
`validation/review/OWNER_REVIEW_31c1006.md`; fixed items there are historical, not
new open findings. User explicitly authorized this integration branch, so its
multi-task layout is not an outstanding process violation.

## Confirmed remaining findings

### H1 — Functional evidence can come from a different run and source

**Status 2026-10-15 (S1): fixed.** `functional_rows(report, profile, teacher_run)` validates the functional
receipt, binds every row's `run_id`/`serving_run_id` to it and requires the serving identity of the teacher run
except the three `RUN_DIAGNOSTIC_ENV` knobs; `quality.py` re-reads the receipt after every request.


**High; reproduced.** `harness/compare-quality.py:29` calls
`freetoken/python/freetoken/quality_evidence.py:209` (`functional_rows`), which
checks matrix/case completeness and value types but never validates `run`.
`harness/quality.py:136` captures the functional receipt only after all requests.
A synthetic otherwise-valid teacher comparison plus functional reports whose
run/profile/source are unrelated still returns bit_exact_pass=true. Changing the
server during the functional pass is also not detected at each request boundary.
This can falsely attribute passing functional behavior to the candidate. Teacher
case/run binding itself remains enforced. Before release, bind functional receipts
and per-case runs to the correct non-teacher arm and verify matching sources,
checkpoint, packages, library bytes and config excluding only declared teacher knobs.

### M1 — Operational launchers still require owner-local path configuration

**Medium; confirmed.** `harness/run-experiment.sh:3`, `harness/experiment.py:4`,
`harness/profile.sh:8` contain `/home/USER` or `/path/to/LLM` site paths.
`rg -n '/home/USER|/path/to/LLM'` on those files reproduces this. Direct use in an
arbitrary checkout can select a nonexistent or wrong runtime. CPU entry and the
read-only owner preflight work; production install/sync was not run or migrated.

### M2 — Exporter assumes standalone upstream Git history

**Medium; reproduced.** `harness/export-release-evidence.py:274` requires
`cac247a860e316e06580d05aeb05f2e647bde214` and HEAD==BASE inside freetoken/.
`git -C freetoken cat-file -e cac247a860e316e06580d05aeb05f2e647bde214` fails in
this monorepo. Fetching the object alone does not repair path/HEAD assumptions.
Use branch commits/diff for delivery; do not manufacture an upstream basis.
The migration study's separate upstream clone does not change this repository.

### M3 — FTW index version and tensor uniqueness are unchecked

**Status 2026-10-15 (S1): fixed.** One `_load_index()` validates format, version, alignment, shard names and
contiguity, duplicate tensors, nbytes and bounds before any shard IO (`tests/checkpoint/test_ftw_index.py`).


**Medium; reproduced on CPU.**
`freetoken/python/freetoken/checkpoint/ftw.py:217` checks format with assert but
ignores version; line 223 builds a dict that overwrites duplicate tensor names.
A temporary index with version=999 and two same-named tensors is accepted.
Future/corrupt checkpoint layouts can be interpreted under the wrong format or
silently load the later tensor. The released pinned model is not demonstrated
corrupt. Validate version, uniqueness, shard ranges and local paths before opening
shards; keep malformed-index tests separate from real checkpoint/GPU tests.

### M4 — Memory-budget rejection disappears under python -O

**Medium; reproduced; conditional on optimized Python.**
`freetoken/python/freetoken/engine/cache_budget.py:65` and :86 use asserts for
positive sizes and budget fit. `plan_cache_budget(300,100,10,4,4,False,10,4)` under
`-O` returns `(4,10,False)`: a 500-byte geometry from a 300-byte budget. The normal
interpreter rejects it. This can defer rejection to allocation/OOM under memory
pressure. Production launcher does not currently request -O; use explicit runtime
exceptions before adding that mode. Other internal asserts need a similar policy
review, not a blanket removal of invariants.

### M5 — Frozen final release driver is incompatible with schema 2

**Status 2026-10-15 (S1): replaced** by `harness/gates/release-gates.sh`; `final-gates.sh` stays the f1 archive.


**Medium; confirmed.** `harness/gates/final-gates.sh` still calls qualify without
mandatory reference/functional-baseline arguments and reuses f1 report names;
its teacher environment extraction predates the reference-report field. Existing
final checker is strict, so this fails closed. Do not execute this archival driver
for a new release. Use the reviewed GPU_VALIDATION commands with fresh names.

### M6 — 256K profiling and long-context bit-exact coverage are incomplete

**Medium; reproduced.** `harness/profile-request.py:34` caps final_requests at
120,000 document tokens even for context=262144. The teacher case set in
`quality_evidence.py` also ends at 120K. 250K functional/boundary tests and the
historical throughput result do not establish 250K bit-exactness or an attention
breakdown. Extend profiling via a separately named workload and explicitly review
any new private fixture; do not rename the old fixture or alter f1 evidence.

### M7 — Short-route component does not test stale borrowed bytes after close

**Status 2026-10-15 (S1): test added** (post-close phase without `resident()`); GPU result in the S1 report.


**Medium; confirmed test gap.** `freetoken/tests/moe/test_short_route_gpu.py:29`
repopulates all resident slots before each phase. It tests D2D exclusion inside an
epoch, but cannot catch incorrect restoration after borrowed bytes were overwritten.
Add a post-close short phase without resident() before accepting T1. This does not
establish a current race; model prefix/teacher/cancel gates remain required.

Read-only reproduction of H1/M3/M4/M6:

```bash
/path/to/cpu-python tools/reproduce-review-findings.py
```

The resulting booleans in `validation/review/reproductions.json` mean **defect
reproduced**, not PASS qualification. No production receipt/reference was created.

## Hypotheses requiring target-device evidence

- **HYP-H2, hybrid only:** `layers/moe.py:_decode_hybrid` clamps inactive route ids
  to slot 0 with zero weights. `0 * NaN` can contaminate a sum if unused slot bytes
  decode to NaN. New upstream explicitly zeros inactive rows for mixed pools.
  This branch uses a single layout; no real-kernel reproduction was obtained.
  Released GPU-only profiles and T3 eligibility exclude hybrid. Test a cold cache,
  max_fetch=0 and poisoned unused slots before enabling hybrid, without changing
  routing arithmetic in this task.
- **HYP-H3, T3 only:** CPU traces establish enqueue order, not CUDA capture/replay
  behavior. Driver event reuse, byte readiness and full model arithmetic must pass
  `test_decode_overlap_gpu.py` and teacher gates on SM120. No missing wait was
  demonstrated in the existing scheduler; its three waits were retained.
- **HYP-M8, stack migration:** unchanged Triton source may emit different reductions
  with Triton 3.8/LLVM/PTXAS. MIGRATION.md names actual source/compiler differences;
  no CUDA replica is certified against the target stack here.

## Coverage and disposition by subsystem

| Area | Review performed / available evidence | Limit |
|---|---|---|
| MoE cache / memory pressure | ensure/copy/LRU bridge, pinned-bank lifetime, rebuild drain, borrowed-tail mapping, hybrid inactive slots; T3 ordering tests | device LRU/copy and allocation failure need GPU |
| Layer-major | plan_store/layout/windows, task order, epoch/close/cancel, grouped attention plan guards; existing T1 CPU + 256K sizing | no CUDA execution; M7 |
| Scheduler / prefix pools | pre-schedule/restore/commit waits, abort acknowledgements, idle rebuild, free-list/radix invariants | 184 additional cache/daemon tests pass; one deliberately unsupported geometry skips |
| CUDA graphs | GraphRunner capture/replay/reset/destroy, fixed addresses, default-off fallback; T3 fork/join exception paths | replay cannot be qualified on CPU |
| HTTP guard / protocol | auth, duplicate keys, depth/size/time bounds, concurrent admission and cleanup; OpenAI handler rejects response_format/logit_bias/function_call | existing guard/tool/worker tests; no live server |
| Checkpoint / model loading | FTW index/short reads/thread cleanup, host pinning, Qwen weight layout, quant registry interfaces | M3; other model families inventoried, not numerically qualified |
| Native / Triton / CUDA | loader recipe/binary gates, launch bounds, bulk-copy shared::cta, GDN/FP8/NVFP4 reference paths | no nvcc/cuobjdump; no SASS or race-sanitizer PASS |
| Harness / acceptance | teacher rotations/run mapping, functional matrix, final consumers, component/library evidence, bootstrap direction | H1, M2, M5, M6; private archive NOT_RUN |
| Daemon / lifecycle | child identity, stop/reap/locking, import safety; existing CPU tests | no owner service manipulation |
| Serving / release / edge | receipt inventory/env clearing, library ABI/source hashes, site launchers, Windows artifacts static inventory | M1; WSL/PowerShell/external deployment NOT_RUN |
| Tests / archives / docs | full source syntax inventory, reviewed mandatory CPU matrix, historical f1 preserved | no claim that every repository test ran |

CPU-enablement fixes from this review: radix tests now import the existing pure
CHUNK_SIZE constant, and the cold-handle error test accepts Torch's RuntimeError
as well as ValueError. No prefix cache arithmetic changed. The expanded mandatory
CPU runner includes actual radix/free-list/daemon tests, with isolated test-only
FastAPI dependencies pinned. Missing dependencies and collection errors are recorded
in CPU_VALIDATION.md rather than counted as successful tests.
