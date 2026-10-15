# CPU validation — T3/T5/T6 follow-up, 2026-10-15 UTC

Latest code tested: `7aab36a` on `agent/cloud-next-update`. Follow-up began clean
at `7c30166`; on the user's request, `git fetch origin` and a fast-forward brought
in five owner commits to `31c10060375d4bd068e14f22bdf3f918cbc18f2b`. No merge commit,
main update or overwrite of owner work. That fetched revision is this follow-up's
baseline. T3/T5/T6 supersede the previous DEFERRED/not-delivered statuses below.

Same isolated CPU environment as the prior run: Python 3.12.14, Torch 2.8.0+cpu,
Triton 3.6.0, pytest 8.4.2. Added only test dependencies fastapi 0.115.12,
starlette 0.46.2, anyio 4.12.1 via:

```bash
uv pip install --python /workspace/.venv-qwen36-cpu/bin/python \
  fastapi==0.115.12 starlette==0.46.2 anyio==4.12.1
```

Exit 0; these are now in requirements-cpu.txt. T6 target wheels were inspected
through metadata; Torch 2.14/Triton 3.8 were **not installed**. Production locks,
released profile values and model weights are unchanged.

## Before / integration / final results

- Baseline command: `QWEN36_CPU_PYTHON=/workspace/.venv-qwen36-cpu/bin/python bash
  harness/CPU-checks.sh --output /workspace/scratch/t3-baseline-cpu` — exit0,
  30 suites, 698 passed/executed, 0 failed, 2 archive skips.
- First expanded pytest attempt also selected `engine/test_cache_budget.py`:
  collection exit2, no tests executed; importing engine through the CPU namespace
  requires model-layer initialization. `validation/review/collection-before.xml`
  records this unsupported import contour; it is not a passed budget suite.
- Cache/radix/daemon run before CPU adjustments: exit1, 47 passed, 57 failed,
  81 setup errors, zero skips. Causes: GPU-module import for CHUNK_SIZE, Ninja
  missing from PATH, missing FastAPI, and Torch's RuntimeError vs the test's
  expected ValueError for an empty cold handle. Compact evidence is in
  `validation/review/cpu-before.json`. No prefix arithmetic was changed.
- Corrected expanded suite: exit0, 184 passed, 0 failed/errors, one unsupported
  hybrid geometry skip, CUDA uninitialized. It now runs as the mandatory
  `test-core-cache-cpu.py` with exact JUnit counts.
- First full T3 integration: exit1, 29/32 suites passed, 3 failed suites with
  17 test failures/errors (892 executed; 3 skipped). It caught missing overlap
  export/receipt mappings and the switch-list expectation. See
  `validation/cpu-t3-integration-red.json`; fixed in `7aab36a`, without relaxing
  profile/evidence gates.
- Final complete CPU command below: **exit0, 32 suites, 892 passed/executed,
  0 failed, 3 skipped**. See `validation/cpu-t3-final.json`.
- `PYTHONPATH=freetoken/python:harness /workspace/.venv-qwen36-cpu/bin/python -O
  harness/test-decode-overlap-cpu.py`: exit0, **10 tests passed**. Explicit
  checks/mutants execute under optimization; this does not make all legacy engine
  assertions safe under -O (REVIEW M4).

```bash
QWEN36_CPU_PYTHON=/workspace/.venv-qwen36-cpu/bin/python \
  bash harness/CPU-checks.sh --output /workspace/scratch/t3-t5-t6-regression-cpu
```

The three skips are two pre-existing archive-only cases plus hybrid radix
page_size=128, which the implementation explicitly does not support. Missing
required dependencies, collection errors and zero execution fail the runner.
Each legacy guard/metrics/logging script is one integrated scenario, not its
individual assertion count. Fake stream tests execute real helper/model methods
and kill missing-wait mutants; they do not prove absence of CUDA races.

The follow-up also reproduces functional attribution, FTW version/duplicates,
-O budget rejection and long256 profiling findings with temporary/synthetic data:
`/workspace/.venv-qwen36-cpu/bin/python tools/reproduce-review-findings.py`, exit0.
Here exit0 means the documented defects were reproduced, not that they were fixed.
See REVIEW.md. No synthetic fixture is used as production evidence.

## Path, syntax and publication checks

The same 32-suite set also passed from `/` using a physical source copy at
`/workspace/scratch/qwen t3 with spaces` and a log directory with spaces:

```bash
cd /
QWEN36_CPU_PYTHON=/workspace/.venv-qwen36-cpu/bin/python \
  bash '/workspace/scratch/qwen t3 with spaces/harness/CPU-checks.sh' \
  --output '/workspace/scratch/t3 cpu logs with spaces'
```

Exit0: **892 passed/executed, 0 failed, 3 skipped**;
`validation/cpu-t3-spaces.json`. No symlink/CWD assumption was used.
`git diff --check` passed. All tracked Python and normalized shell source passed
AST/bash syntax review; the new GPU component also passed py_compile.
The publication scan initially flagged cuDNN's public four-part package version
as an IPv4 address. Two explicit file/rule allowlist entries document that false
positive; no credential or site address was substituted or exposed.

## Separate unavailable checks

GPU/model/byte-exact/capture-replay/speed, CUDA compile-only/SASS, private teacher
fixtures and Windows/WSL release steps: **NOT_RUN**. nvcc, cuobjdump and pwsh are
absent; there is no owner GPU/model/private runtime. The new GPU test is syntax
checked, not counted as an executed CPU test. T3 cannot be qualified by profile
validation. MIGRATION.md is a source/metadata study, not an environment upgrade.

---

## Historical first cloud run (2026-10-14)

Initial checkout: `7c78a4824faf03c0eee6f8b3e6fa9cd4cf7df401`, branch `main`, clean
status. The selected cloud workspace was initially empty; the public repository
was cloned to `/workspace/qwen36-moe-16gb`. No newer HEAD superseded the task.
Work branch: `agent/cloud-next-update`. No push, merge, release, service operation,
model download, installed-owner-runtime or production modification occurred.

## Environment and reproducible setup

Linux 6.18.44 x86_64, Python 3.12.14, GCC 14.2.0. Isolated CPU venv:
`/workspace/.venv-qwen36-cpu`. Main CPU dependencies: pytest 8.4.2, numpy 2.3.3,
requests 2.32.5, torch **2.8.0+cpu**, transformers 4.57.1, triton 3.6.0,
apache-tvm-ffi 0.1.13.post3 and Ninja 1.13.0. Full exact installed versions are in
`harness/requirements-cpu.txt`. These are CPU-test dependencies, not a proposed
production stack or a qualification environment. Existing production locks and
released profile values are unchanged.

Example isolated setup, with explicit Python and trusted package indexes:

```bash
uv venv --python 3.12 /absolute/path/to/cpu-venv
PY=/absolute/path/to/cpu-venv/bin/python
uv pip install --python "$PY" --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match -r harness/requirements-cpu.txt
QWEN36_CPU_PYTHON="$PY" bash /absolute/path/to/checkout/harness/CPU-checks.sh \
  --output /absolute/path/to/cpu-logs
```

The explicit version lock includes `torch==2.8.0+cpu`. No serving/CUDA extras were
installed. CPU tests compile the existing radix C++ helper and synthetic ABI ELF
fixtures. They do not compile CUDA. TVM-FFI's optional torch C-DLPack addon is
explicitly disabled for CPU runs; its ordinary DLPack fallback is used. In paths
with spaces, the test-only bootstrap stages byte-identical radix source/headers
in a temporary directory because TVM-FFI's Ninja writer does not escape source
paths. Production loader/build policy is not changed by that bootstrap.

`torch.cuda.is_available()` and `torch.cuda.is_initialized()` both returned false.
`nvcc`, `cuobjdump`, `nvidia-smi` and `pwsh` were absent from PATH; no CUDA toolkit
was found in `/usr/local/cuda*` or `/opt/cuda*`.

## Before changes

- `bash harness/CPU-checks.sh`: **exit 127**, zero tests. First Python path was
  `/home/USER/qwen36-api/.venv/bin/python`; reproducing the original script from
  the pinned Git revision still returns 127.
- `python3 -c 'import pytest, numpy; ...'`: **exit 1**, pytest absent in the base
  interpreter. Dependencies were then installed only in the isolated venv.
- Public scripts were enumerated and invoked directly with
  `PYTHONPATH=freetoken/python:serving:harness`, `CUDA_VISIBLE_DEVICES=`, explicit
  venv Python, per-script 90s timeouts and captured stdout/stderr. This exploratory
  baseline included the unavailable private eval script and guard scripts without
  their newly supplied arguments: **22 script invocations, 9 exit 0, 13 errors**;
  178 executed cases/scenarios: 165 passed, 13 test errors, plus 2 skips. Most failures were old
  layout, imports, unavailable fixtures or invocation problems, not numerical
  regressions. Exact per-suite data: `validation/cpu-before.json`.
- Original quality false acceptance reproduced with
  `"$PY" harness/reproduce-quality-before-cpu.py`: **exit 0**, original comparator
  reported quality PASS for an empty candidate functional set, unbound historical
  teacher rows, `max_abs=0.25` and `relative_l2=0.01`. Synthetic public data only;
  `validation/quality-before.txt` records the result. New regressions reject those
  inputs. Other regressions exercise actual helpers/method bodies, including
  mutants that delete/reorder the scheduler commit wait. Final `qcheck` is also
  exercised through its real isolated/optimized CLI; incomplete/legacy/byte-mismatched
  reports fail and the historical R1b override cannot bypass it in new runs.

Intermediate red runs exposed and repaired: missing public source paths,
production-package metadata in a CPU fixture, a missing archived candidate JSON,
bytecode writes affecting immutable-tree tests, missing CPU JIT dependencies,
Ninja paths with spaces, and a FP8 provenance test that accidentally examined its
patched empty fixture tree and skipped. The final FP8 source-pin test executes;
it does not rely on that skip.

## Final commands and coverage

Executed after integration, using the same public CPU suites plus the added
regressions and the original speculative CPU primitive tests:

```bash
QWEN36_CPU_PYTHON=/workspace/.venv-qwen36-cpu/bin/python \
  bash harness/CPU-checks.sh --output /workspace/scratch/qwen36-complete-cpu
```

The full suite was also copied into a real checkout directory with spaces and
launched from `/`, with logs in a directory with spaces:

```bash
cd /
QWEN36_CPU_PYTHON=/workspace/.venv-qwen36-cpu/bin/python \
  bash '/workspace/scratch/qwen with spaces/harness/CPU-checks.sh' \
  --output '/workspace/scratch/cpu complete logs with spaces'
```

Final exact results are recorded in `validation/cpu-final.json` and
`validation/cpu-spaces.json`; the table below is generated from the final summary.
The three legacy guard/metrics/logging scripts count as one integrated scenario
apiece, not as a count of their assertions. Subtests and 25 killed paired-contract
mutants are not double-counted as separate parent test methods.

**Both commands: exit 0; 30 suites, 695 executed and passed, 0 failed, 2 optional archive-only skips.**

| Suite | Passed/executed | Skipped | Exit |
|---|---:|---:|---:|
| `test-analyze-model-nsys-cpu.py` | 29 | 0 | 0 |
| `test-code-runner-cpu.py` | 6 | 0 | 0 |
| `test-cpu-entry-cpu.py` | 3 | 0 | 0 |
| `test-export-release-evidence-cpu.py` | 13 | 0 | 0 |
| `test-gdn-snapshot-cpu.py` | 6 | 0 | 0 |
| `test-keepalive-cpu.py` | 19 | 0 | 0 |
| `test-lookup-qualification-cpu.py` | 38 | 0 | 0 |
| `test-network-acceptance-cpu.py` | 19 | 0 | 0 |
| `test-next-profile-cpu.py` | 5 | 0 | 0 |
| `test-ops-live-probe-cpu.py` | 37 | 0 | 0 |
| `test-paired-contract-cpu.py` | 57 | 0 | 0 |
| `test-prefix-quality-cpu.py` | 9 | 0 | 0 |
| `test-profile-health-cpu.py` | 4 | 0 | 0 |
| `test-profile-runtime-cpu.py` | 88 | 0 | 0 |
| `test-provenance-loader-cpu.py` | 2 | 0 | 0 |
| `test-quality-evidence-cpu.py` | 12 | 0 | 0 |
| `test-readiness-transport-cpu.py` | 8 | 0 | 0 |
| `test-release-cpu.py` | 13 | 0 | 0 |
| `test-release-qualification-cpu.py` | 56 | 0 | 0 |
| `test-retain-snapshot-cpu.py` | 40 | 0 | 0 |
| `test-short-route-cpu.py` | 5 | 0 | 0 |
| `test-soak-cpu.py` | 77 | 2 | 0 |
| `test-speculative-cpu.py` | 9 | 0 | 0 |
| `test-stream-ordering-cpu.py` | 3 | 0 | 0 |
| `test-tool-choice-guard-cpu.py` | 41 | 0 | 0 |
| `test-tool-semantics-guard-cpu.py` | 74 | 0 | 0 |
| `test-worker-fault-v2-cpu.py` | 19 | 0 | 0 |
| `test-guard.py` | 1 | 0 | 0 |
| `test-metrics.py` | 1 | 0 | 0 |
| `test-log-failure.py` | 1 | 0 | 0 |

The only skipped tests are two comparisons against unavailable **archive-only
pre-change soak copies**. No mandatory current CPU suite was skipped. Missing
required dependencies, failed collection or zero executed cases produce a visible
runner failure. Private/GPU/Windows omissions are NOT_RUN categories, not PASS.

Additional checks:

- `python -O harness/test-quality-evidence-cpu.py`, `test-next-profile-cpu.py`,
  `test-provenance-loader-cpu.py` with the same PYTHONPATH/empty CUDA visibility:
  exact counts/exit codes in `validation/optimized-checks.txt`; all gate checks use
  explicit exceptions, not removable asserts.
- Built-in `compile()` over every changed/new Python file: count and filenames in
  `validation/static-checks.json`. CPU entry imports are exercised by the runner;
  GPU source files receive syntax validation only.
- `bash -n harness/CPU-checks.sh harness/start-experiment.sh harness/gates/final-gates.sh`: exit 0.
- `git diff 7c78a4824faf03c0eee6f8b3e6fa9cd4cf7df401 --check`: exit 0. Existing CRLF
  files retain their format; local `core.whitespace=trailing-space,space-before-tab,cr-at-eol`
  treats CRLF correctly. For a different checkout, use the equivalent `git -c`
  option instead of reformatting unrelated files.
- `python tools/scan-publish.py --all-files --json <scratch-result>`: exit 0,
  zero findings; no private source/teacher data was added.
- Released profiles: `git diff <initial-SHA> --exit-code -- serving/profiles.json`:
  exit 0. Historical `results/` files and production dependency locks are unchanged.

## NOT_RUN and limits

| Category | Status and concrete reason |
|---|---|
| CUDA 13.x compile-only / SM120a SASS | NOT_RUN — no nvcc or cuobjdump; no unsuitable target substituted |
| GPU/model/component and teacher gates | NOT_RUN — CPU torch, no NVIDIA GPU, model, pinned native libraries or owner kernel cache |
| Private eval-v1 suite | NOT_RUN — `harness/eval-v1/cases_v1.py` / `sandbox_v1.py` absent; no fixture fabricated |
| Owner teacher fixture/reference | NOT_RUN — private target IDs/arrays/receipts absent; baseline files are never generated or overwritten to fix this |
| PowerShell/Windows/WSL | NOT_RUN — pwsh, Windows interop and owner services absent |
| Archived short-route toy suites | NOT_RUN — archived `work/layer-major/tests/layer_major/toy.py` is absent; new CPU tests cover real routing/epoch control flow, not model arithmetic |
| T3 gather/shared overlap | DEFERRED — no active path; ownership/capture plan documented separately |

CPU synthetic state/control-flow results do not prove CUDA ordering, GPU graphs,
bit-exact model output, release suitability or speed. The owner runbook is
`GPU_VALIDATION.md`. `REVIEW.md` records two remaining confirmed layout/export
limitations; neither was hidden behind a passing CPU verdict.
