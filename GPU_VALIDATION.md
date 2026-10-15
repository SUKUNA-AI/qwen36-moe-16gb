# Owner GPU validation — NOT_RUN in cloud

CPU results do not establish GPU correctness, bit-exactness, release qualification
or speed. No GPU/model/private-fixture/WSL test and no CUDA compile was run here.
T3 overlap is DEFERRED; there is no overlap switch to enable.

## 1. Local prerequisites and first commands

Use the owner's pinned SM120 stack, model and **reviewed experiment overlay**.
Do not change production `release/python`, start production, run a blanket sync,
install dependencies into the serving venv, or generate an acceptance receipt.
The public deployment launchers still contain site placeholders; use reviewed,
site-configured experiment launchers. `sync-v5.sh` is not migrated by this change.

Set these absolute paths; `N` is the existing experiment overlay (normally
`S/optimization`), `C` is this checkout, and `P` is the existing GPU Python:

```bash
C=/absolute/path/to/qwen36-moe-16gb
S=/absolute/path/to/owner-runtime
N="$S/optimization"
P="$S/.venv/bin/python"
MODEL=/absolute/path/to/pinned-NVFP4-checkpoint
LOCK="$S/run/gpu-exclusive.lock"
GDN=/absolute/path/to/pinned/libqwen36_gdn.so
export QWEN36_DATA_ROOT="$N" QWEN36_HARNESS_SOURCE="$N/config"
export QWEN36_GPU_PYTHON="$P" QWEN36_GPU_LOCK="$LOCK" QWEN36_MODEL_DIR="$MODEL"
export FREETOKEN_PYTHON="$N/python" PYTHONPATH="$N/python:$N:$C/harness"
```

Required separately:

- Model `nvidia/Qwen3.6-35B-A3B-NVFP4`, revision
  `1355db6a052410cfd62085d94b58866fd0f2c3c5`, locally present; no downloads.
- Existing checkpoint/private fixture `N/config/teacher-fixture.json`; fixed 20
  cases, target IDs and tokenizer. No synthetic production fixture/receipt.
- Immutable pinned `N/reports/logit-reference/{seed}-{offset}.npy` files, plus a
  schema-2 baseline report or a deliberate owner verification of the existing
  reference described below. A missing reference is a blocker.
- The four pinned libraries and their ABI/source/binary manifests, compiled
  FreeToken extensions, manifest-bound kernel cache and unchanged owner GPU venv.
- Reviewed copies of this branch's harness helpers and FreeToken sources in the
  experiment overlay; its source hashes must match what the process imports.
  Site launcher paths must refer to this overlay and existing owner Python/model.
- Candidate configs `main-release-shortroute`, `main-release-fla-wu`, and
  `main-release-shortroute-fla-wu` under `N/config`, with owner-local library paths
  resolved and `qualified:false`. Keep `main-release` and released profiles intact.

First repeat the CPU command in `CPU_VALIDATION.md`. Then, with a valid baseline:

```bash
REF=owner-run-bound-reference
"$P" "$C/harness/next-update-preflight.py" --native "$N" --python "$P" \
  --model "$MODEL" --reference-report "$N/reports/$REF.json" \
  --reference-dir "$N/reports/logit-reference"
```

This command only reads files and queries unit state. It fails for missing or
legacy reference bindings, active/unknown units, and unconfigured site launchers.
Use `--wsl-interop` before any later Windows release step: it requires WSLInterop
and `cmd.exe`. Cloud does not execute Windows release tooling. Inspect CUDA device,
stack versions and kernel provenance/ABI with the existing owner tools before GPU
work; a source match alone does not verify a `.so`.

### Existing reference verification without rewriting f1

Old f1 reports remain historical and cannot silently qualify schema 2. If no new
run-bound baseline report exists, the owner may **explicitly** rerun teacher-bf128
against a copy of the existing pinned `.npy` reference, with a new report name.
The engine now requires every baseline file to exist and the newly computed
float32 bytes to match it exactly. It never replaces baseline arrays.
Do not invent missing hashes/fixtures or use this to generate a missing reference.
A deliberate new basis (new fixture, the T6 stack) is a separate owner decision:
`QWEN36_TEACHER_CREATE_REFERENCE=1` in a teacher-bf128 config whose
`QWEN36_LOGIT_REFERENCE` is a new directory creates the missing files exclusively.

Preserve all existing reports. Configure `teacher-bf128` to use the existing
fixture and immutable reference copy. With both units verified inactive:

```bash
REF=cloud-reference-$(date -u +%Y%m%dT%H%M%SZ)
FB=cloud-functional-base-$(date -u +%Y%m%dT%H%M%SZ)
export REF FB N P
flock -w 3600 -E 75 "$LOCK" timeout 3600 bash -euo pipefail <<'SH'
trap 'systemctl --user stop qwen36-experiment.service' EXIT
bash "$N/start-experiment.sh" teacher-bf128
"$P" "$N/check-experiment.py" 300
"$P" "$N/quality.py" --name "$REF" --teacher
systemctl --user stop qwen36-experiment.service
bash "$N/start-experiment.sh" bf128
"$P" "$N/check-experiment.py" 300
"$P" "$N/quality.py" --name "$FB" --long 120000
SH
```

This is GPU work explicitly invoked by the owner. Failure to match the pinned
reference bytes stops validation. Save first-run and clean-cache receipts,
case→run mappings, JSONL rotations and run-scoped snapshots together. The
long-120000 row must belong to the fresh restart; that restart may serve no other
teacher case. Expected 20 cases and all forced tokens/checkpoints must be complete
and finite; expect 943 tokens/125 checkpoints only when the owner fixture confirms
those totals. Never replace that fixture to obtain these numbers.

## 2. Components: default-off, T1, then T2

Run only under the existing exclusive lock, with timeouts and raw results separate
from f1. The C++/CUDA sources are unchanged; any optional rebuild has a **new binary
hash**, not automatic qualification. No pin may be weakened to load it.

Compile-only, only if CUDA 13.x and the specified host compiler are present:

```bash
CUDA_HOME=/absolute/path/to/cuda-13.x CUDA_CCBIN=g++-15 \
  BUILD_OUT=/absolute/path/to/isolated-gdn-build bash "$C/kernels/gdn/build.sh"
```

The script targets `sm_120a`. Inspect produced SASS with cuobjdump for
`CALL.ABS`/`__cuda_syscall`; inspect bulk-copy destinations for `.shared::cluster`.
Absence of nvcc/cuobjdump or an unsuitable target is NOT_RUN/FAIL, not PASS.
Compilation alone is not an execution test. Do not substitute a rebuilt library
for the pinned GDN binary during tests below without a separate owner validation.

T1 transport component (both hit-D2D off and on, 50 alternating short/borrowed-slot
epochs per mode, BF16 and both NVFP4 bank layouts with mixed small/large banks) and existing serial gather capture/replay:

```bash
flock -w 3600 -E 75 "$LOCK" timeout 1800 "$P" -m pytest -v \
  "$C/freetoken/tests/moe/test_short_route_gpu.py" \
  "$C/freetoken/tests/moe/test_decode_gather.py"
```

Require byte identity of all banks, positive D2D traffic in enabled short epochs,
zero D2D traffic while tail slots are borrowed, and no skipped mandatory GPU case.
This test does not evaluate model logits or qualify overlap. It also repopulates
every slot before each phase, so it cannot catch hit-D2D reading stale borrowed
bytes after an epoch. Before the model gates, run the archived layer-major CPU suite
(24 tests incl. epoch reopening and the abort sweep, not in this repository) against
this tree, and treat T1 telemetry plus the teacher gate as the end-to-end check.

T2: unchanged GDN off path, `wu`, `ho`, then full fused byte/state stress:

```bash
export QWEN36_CUDA_GDN_LIBRARY="$GDN"
flock -w 3600 -E 75 "$LOCK" timeout 1800 "$P" "$C/kernels/gdn/test_gdn_bitexact.py" \
  --lib "$GDN" --out "$N/reports/cloud-gdn-off.json"
flock -w 3600 -E 75 "$LOCK" timeout 1800 "$P" "$C/kernels/gdn/test_gdn_bitexact.py" \
  --lib "$GDN" --fla --fla-variant wu --out "$N/reports/cloud-gdn-wu.json"
flock -w 3600 -E 75 "$LOCK" timeout 1800 "$P" "$C/kernels/gdn/test_gdn_bitexact.py" \
  --lib "$GDN" --fla --fla-variant ho --out "$N/reports/cloud-gdn-ho.json"
flock -w 3600 -E 75 "$LOCK" timeout 1800 "$P" "$C/kernels/gdn/fla/test_fla_fused.py" \
  --lib "$GDN" --out "$N/reports/cloud-fla-byte-state.json"
```

Use unique output names on reruns; preserve raw failures. FLA component evidence
must identify the exact library SHA, variant and nonzero fused-call count, with
complete passing rows. Generic `cuda_gdn` evidence is insufficient for FLA.
`cuda_gdn_fla_evidence` needs run receipts, schema-2 bitexact reports and component
reports; the receipt records the variant/library hash. Clean-cache candidate run
IDs in the teacher comparison must equal the listed receipt run IDs.

## 3. Model gates and request coverage

Check default-off `main-release`, T1, T2-wu separately, then the combined candidate.
`ho` needs its own config/evidence if intended for qualification; no ho release
candidate is asserted by this branch. There is no T3 model arm.

For each `candidate` below, use a fresh `NAME` and both units verified inactive:

```bash
candidate=main-release-shortroute  # repeat: main-release, main-release-fla-wu, main-release-shortroute-fla-wu
NAME=cloud-$candidate-$(date -u +%Y%m%dT%H%M%SZ)
flock -w 3600 -E 75 "$LOCK" timeout 3600 "$P" "$N/qualify.py" \
  --candidate "$candidate" --name "$NAME" --reference-report "$REF" --functional-baseline "$FB"
```

`qualify.py` requires both evidence arguments and checks them, and every output
report name, before it starts a unit. The final `final-helpers.py qcheck REPORT`
delegates to the same strict checker; new final-driver runs no longer use the
historical R1b equivalence override. `gates/final-gates.sh` itself is the frozen f1
driver and is not schema-2 ready (it calls `qualify.py` without the evidence
arguments, and `extract-run --teacher` does not expect
`QWEN36_TEACHER_REFERENCE_REPORT`): the next release needs a new driver.
Require `evidence_valid`, `quality_gate_pass`, `bit_exact_pass`, expected==observed,
all checkpoint max_abs/relative_l2==0 and all_byte_equal, complete forced targets, and no new functional
failure. Functional reports specify their context matrix and successful HTTP
statuses. Functional baseline and candidate must each be complete; equal missing
sets cannot pass. `quality.py` refuses to overwrite reports.

For T1 also inspect the run-bound `engine.jsonl` telemetry: positive
`prefill_d2d_bytes` on at least one short case, positive layer-major prefills on long
cases, no hit-D2D-unavailable fallback. A zero-traffic run does not exercise T1.
Repeated requests, prefix hits and edits must show the **remaining uncached**
token count, not just document length. The exact 2048/2049 route boundary includes
chat-template tokens; `bench --lengths 2048,2049` alone is a document-length screen,
not an exact uncached boundary proof. Use the owner's tokenizer to pad/count the
full rendered request and verify engine telemetry at threshold−1/threshold/+1.

Owner locked request session, stopped experiment trap (one profile at a time):

```bash
export candidate NAME
flock -w 3600 -E 75 "$LOCK" timeout 3600 bash -euo pipefail <<'SH'
trap 'systemctl --user stop qwen36-experiment.service' EXIT
bash "$N/start-experiment.sh" "$candidate"
"$P" "$N/check-experiment.py" 300
"$P" "$N/bench.py" --name "$NAME-screen" --lengths 64,2048,2049,4096,16384,65536,120000 --repeats 3 --output 128
"$P" "$N/test-prefill-cancel.py" --length 16384 --name "$NAME-cancel16k"
"$P" "$N/test-prefill-cancel.py" --length 120000 --name "$NAME-cancel120k"
SH
```

`bench` includes history continuation and 16K last-message/tool-result edits. For
prefix snapshots use `run-prefix-gates.py --profile "$candidate" --name NEWNAME
--lengths 16384,120000` under flock; it handles experiment stop/start in a finally
block. Include a long original prompt with a suffix below 2049 after a cache hit.
Repeat T1 with several chunk sizes (1024/2048/4096) in separate unqualified configs.
For long256, create a distinct unqualified derivative of `long256-release` with
only the feature under test; test 120K, 250K, cancellation, repeated requests and
`boundary.py --context 262144 --name NEWNAME`. A teacher-enabled long256 variant
needs an explicit reviewed allowlist entry; this branch adds only the three 128K
candidate names. Do not enable teacher forcing on a released serving profile.

## 4. Paired rotating A/B, no profiler

Use five pairs, sustained `QWEN36_BENCH_WARMUP_S=60`, output 128 and the project
methodology. Report every pair plus medians, elapsed, TTFT and decode separately.
No Nsight/NCU attached. Each matrix freezes source/config/library/launcher identity
and refuses drift; use a new name after source/config changes.

`tool-pilot.py` and `matrix.py` acquire the **same** lock internally. Do not hold an
outer flock while calling them (that would cause their exit 75). If busy, block
with `flock -w 3600 -E 75 "$LOCK" true`, then retry; their own acquisition remains
atomic and a competing owner still yields exit 75, never a false PASS.

```bash
export QWEN36_BENCH_WARMUP_S=60
PILOT=cloud-tool-pilot-$(date -u +%Y%m%dT%H%M%SZ)
"$P" "$N/tool-pilot.py" --profile bf128 --repeats 3 --name "$PILOT"
AB=cloud-ab-$(date -u +%Y%m%dT%H%M%SZ)
"$P" "$N/matrix.py" --profiles main-release,main-release-shortroute --pairs 5 \
  --name "$AB" --output 128 --tool-pilot "$PILOT"
"$P" "$N/compare-ab.py" --name "$AB" --baseline main-release --candidate main-release-shortroute
```

Repeat T2 versus main-release, then combined versus main-release. The comparison
uses baseline/candidate elapsed-time speedup. Keep matrix raw JSON and receipts;
exit 0/1/2 means PASS/FAIL/INCONCLUSIVE, not a release decision. The screen's 2048/
2049 cases are additional diagnostic coverage, not a changed historical f1 matrix.
Set owner session timeouts around pilot/matrix appropriate to their full schedules;
never take over another experiment unit.

## 5. T3 future prerequisite

See `next-update/gather-prefetch/OVERLAP-DESIGN.md`. Before a future T3 arm, add a
component that captures the producer fork/gather/shared/join in one graph and
stress-replays alternating miss metadata, 0..8 misses, slot reuse, repeated requests
and cancel/drain. All bytes, untouched slots and outputs must match the serial
reference; negative controls deleting producer waits or the join must fail.
The existing gather replay test proves only serial gather transport. There is no
completed overlap test/CLI to run or an overlap config to qualify in this branch.
