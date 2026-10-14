# Cloud next-update handoff

Branch `agent/cloud-next-update`, based on clean public
`7c78a4824faf03c0eee6f8b3e6fa9cd4cf7df401`. Code is committed locally; no push,
upstream PR/issue, release, deployment, installed runtime or production operation.
`serving/profiles.json`, historical f1 results and production dependency locks are
unchanged. This is an unqualified CPU-verified candidate.

## Delivered

- P0: source-relative CPU runner with explicit Python/paths, mandatory nonzero
  coverage, per-suite logs and separate NOT_RUN categories. CPU state helpers use
  real source/arithmetic and CPU radix C++, including a checkout with spaces.
- T4: per-case teacher run binding to report receipts and clean-cache restarts;
  strict rotated-log reader plus run snapshots; fixture/target/reference/source
  identity, typed complete finite rows, functional matrix coverage/status, separate
  evidence/quality/byte-exact verdicts and the final qualification gate. Missing
  baseline arrays fail; existing arrays and fixture targets are never overwritten.
  Reports use exclusive creation. `/health` readiness is bounded and authenticated.
  Actual scheduler control flow is regression-tested with fake streams; existing
  waits were retained. Elapsed-time speedup documentation corrected.
- T1: archived short-route patch integrated into engine/scheduler/cache. Remaining
  uncached tokens select the route; D2D stays disabled throughout borrowed-slot
  epochs, with begin failure cleanup and short-path rearming. Default unset/0
  routing is retained. New GPU transport test covers enabled/disabled hits and
  BF16/native NVFP4/Marlin banks; NOT_RUN here.
- T2: existing `cuda_gdn_fla` value field completed, no duplicate boolean switch.
  Fused evidence must bind variant, actual library, component rows and candidate
  teacher run receipts. Generic cuda_gdn receipts cannot qualify FLA. Unknown raw
  engine values/dependencies now fail; `1` remains the engine's legacy `wu` alias.
- P2: `FREETOKEN_STRICT_PROVENANCE=1` requires manifest/recipe/binary binding before
  prebuilt load and forbids rebuilding. Production pins it with DISABLE_JIT=1;
  DISABLE_JIT alone remains a build-availability policy in development. The
  stronger wrapper verification remains. Fixture execution runs in a bounded
  Linux child (input/AST/CPU/address-space/output limits, wall timeout and cleanup),
  preserving the syntax policy. It is not a full security sandbox; unsupported
  platforms fail instead of using in-process exec.
- T3: **DEFERRED**, no active overlap path/flag. Concrete producer/consumer ownership,
  graph capture/replay and negative-control plan is in
  `next-update/gather-prefetch/OVERLAP-DESIGN.md`.

## Candidates and compatibility

All three configs are `qualified:false`, under `harness/`:

| Config | Explicit enablement |
|---|---|
| `main-release-shortroute.json` | `--moe-prefill-hit-d2d`, `QWEN36_LAYER_MAJOR_MIN_TOKENS=2049` |
| `main-release-fla-wu.json` | `QWEN36_CUDA_GDN_FLA=wu`, existing CUDA GDN enabled |
| `main-release-shortroute-fla-wu.json` | both, only after isolated gates |

Profile field `layer_major_min_tokens`: exact integer 0..2147483647, needs
layer_major_prefill. Combined layer-major/hit-D2D needs a positive threshold.
The engine preserves the archived parser contract: malformed/negative raw env
falls back to 0; production profile validation rejects those values and bools.
FLA profiles accept exactly `wu`/`ho` with cuda_gdn; absent disables it and inherited
FLA env is cleared. Engine unset/0/empty disables, legacy `1` selects wu. Neither
candidate config nor historical patch measurements is a new qualification result.

## Reviewable commits and rollback

Apply in order; the CPU scaffolding and evidence changes are dependencies of the
feature tests/gates. A following documentation commit contains this handoff and
validation summaries; its SHA is `git rev-parse HEAD` in the delivered branch.

| Commit | Purpose |
|---|---|
| `9ce1f6a` | public CPU paths/runner/import contour |
| `5ac6b98` | run-bound quality/reference/functional gates, health/ordering regression |
| `e60e7d8` | T1 optional short-route and epoch lifetime |
| `7c74b38` | T2 fused evidence and candidate configs |
| `f11b7e7` | strict provenance loader policy |
| `0c236bd` | bounded fixture-code worker |
| `47534d7` | invalid raw FLA env rejection |
| `883e400` | GPU bank-layout/producers component coverage (unrun) |
| `6d3bfad` | declared teacher fixture version gate |

For a clean owner checkout of the same base, cherry-pick these commits in this
order and then the documentation commit. Revert dependents in reverse order;
T1/T2 can also be disabled by using the unchanged main-release config. Do not
revert receipt/evidence hardening merely to accept invalid reports. The combined
config needs both features. A local bundle/combined patch is supplied with the
final response for offline review; no remote branch was published.

## Verified and next actions

Final scope/results: `CPU_VALIDATION.md` and `validation/`; exact commands and
counts are there. GPU numerical correctness, graphs, speed and release suitability
remain **NOT_RUN**. CUDA compile/SASS, model/private fixtures and Windows/WSL are
unavailable; archived toy/soak before copies are absent. No acceptance receipt
was created. Source export from monorepo history and remaining site launcher paths
are confirmed limitations in `REVIEW.md`.

First run the documented CPU entry point. On the owner machine, stage/review only
the experiment overlay and configure its local paths; use the read-only
`next-update-preflight.py` command in `GPU_VALIDATION.md`. If a schema-2 reference
is missing, deliberately verify the existing immutable reference on GPU under a
new report name; do not generate missing arrays or rewrite f1. Then default-off,
T1, T2 separately: byte components -> clean-cache run-bound teacher/functional ->
request boundaries/prefix/edit/cancel -> unprofiled rotating five-pair A/B. Only
then test the combined candidate. All GPU work is an explicit owner action under
the existing exclusive lock, managing the experiment unit only.
