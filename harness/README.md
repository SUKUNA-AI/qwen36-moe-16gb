# harness/ — experiments, quality gates, qualification, soak and release tooling

Latest versions only (harness V6 + quality.py v7). The scripts were written for a two-tree setup:
this source tree (formerly `outputs/qwen36-api/optimization` of the old workspace) is synced by
`sync-v5.sh` into the native experiment overlay `${QWEN36_NATIVE}` (= `${QWEN36_ROOT}/optimization`,
flat layout, configs under `config/`), and every GPU run happens there under one exclusive GPU lock.

Tracked copies use placeholders (`/home/USER/...`, `/path/to/qwen36-workspace`, see
`serving/README.md`); the owner's working copies with real values are under `local/harness/`.

| Group | Files |
|---|---|
| Experiment server | `experiment.py`, `start-experiment.sh`, `run-experiment.sh`, `check-experiment.py`, `*.json` run configs (schema: context, chunk, memory_ratio, flags, env, kv_dtype, qualified) |
| Quality gate (teacher forcing, bit-exact) | `quality.py` (v7: clean-cache restart before `long-120000`), `fixtures.py` (synthetic documents), `compare-quality.py`, `compare-logit-reference.py`, `qualify.py`, `teacher-*.json`, `functional-matrix.py`, `diagnose-facts.py`, `isolate-quality.py`, `boundary.py`, `prefix-quality.py`, `run-prefix-gates.py` |
| Paired speed matrix | `bench.py`, `matrix.py`, `compare-ab.py`, `screen-chunk.py`, `tool-task-bench.py`, `tool_fixture_v5.py`, `tool-pilot.py`, `check-candidate-metrics.py`, `harness-v6-runbook.md` |
| Context / live gates | `run-context-gates.sh`, `run-live-gates.sh`, `test-backend-v2.py`, `test-resilience.py`, `test-prefill-cancel.py`, `test-output-budget.py` |
| Soak | `soak.py`, `soak-analysis.py`, `test-soak-cpu.py` |
| Nsight | `profile.sh`, `profile-request.py`, `analyze-model-nsys.py`, `profile-final-*.json`, `final-nsight-runbook.md` |
| Release | `release.py`, `make-qualification.py`, `export-release-evidence.py`, `release-management.ps1`, `qwen36.cmd`, `sync-v5.sh`, `sync-final-freeze.sha256`, `gates/` (final gate driver used for release f1: A CPU suites -> B qualification -> D pilot/matrix -> C context gates -> E soak -> F install/live) |
| Operations checks | `network-acceptance.py`, `ops-live-probe.py`, `ops-live-*.ps1`, `ops-control-tests.ps1`, `ops-readiness-tests.py`, `inspect-keepalive.py`, `ops-live-runbook.md`, `OPERATIONS.md` |
| CPU test suites | `test-*-cpu.py`, `test-*-cpu.ps1`, `CPU-checks.sh`, `component-checks.sh` |
| Evaluation runner | `eval-v1.py`, `eval-v1-compare.py`, `eval-v1-runbook.md`; the case set `eval-v1/` is private and gitignored |

Private inputs that are never published: the teacher fixture targets
(`${QWEN36_NATIVE}/config/teacher-fixture.json`), `teacher.jsonl` logit references, the eval-v1
case set, and raw reports with prompts/answers (`reports/`).

Known path couplings that still point at the old workspace layout (`outputs/qwen36-api`,
`work/FreeToken-opt`) are listed as TODO in `MANIFEST.md`.

## Public checkout CPU entry point

Use Linux and an isolated Python 3.12 environment. The CPU lock is
[`requirements-cpu.txt`](requirements-cpu.txt), deliberately separate from the
production stack. Install `torch==2.8.0+cpu` from the PyTorch CPU index, then the
other pinned packages from that file (`uv pip install --python "$PY" ...`). Do not
install the CUDA serving extras. C++ compilation of the existing CPU radix helper
needs a C++20 compiler and Ninja; no nvcc is used by this entry point.

```bash
QWEN36_CPU_PYTHON=/absolute/path/to/cpu-venv/bin/python \
  bash /absolute/path/to/checkout/harness/CPU-checks.sh --output /absolute/path/to/cpu-logs
```

Paths resolve from `repo_paths.py`, independently of CWD and with spaces supported.
Overrides: `QWEN36_CHECKOUT`, `QWEN36_FREETOKEN_SOURCE`, `QWEN36_SERVING_SOURCE`,
`QWEN36_HARNESS_SOURCE`, `QWEN36_KERNELS_SOURCE`, `QWEN36_GUARD_UNDER_TEST`.
All overrides should be absolute paths. The output directory receives per-suite
logs and `summary.json`; script reports also go there.

The runner executes every public `harness/test-*-cpu.py`, the three legacy
serving/logging script scenarios, and the original speculative primitive tests
through `test-speculative-cpu.py`. It runs each script in a fresh process, reports
executed/skipped counts, and fails for errors, collection failures, missing CPU
dependencies or zero executed tests. Three legacy integrated scripts count as one
scenario apiece, not as a count of their assertions. `pytest` default discovery
would miss these scripts. `test-eval-v1-cpu.py` is separately NOT_RUN because
`eval-v1/cases_v1.py` and `sandbox_v1.py` are private and absent. PowerShell tests,
CUDA compilation, GPU/model tests and private fixtures are separate categories;
`component-checks.sh` is **not** a CPU runner.

The test-only namespace bootstrap `cpu_imports.py` bypasses eager package model
initializers. State/cache arithmetic is the real source, including the existing
CPU radix C++ module and real FLA integer metadata helpers. No model is loaded,
no CUDA context is initialized and no fixtures are downloaded. This is CPU
control-flow/state validation, not GPU numerical evidence.

New teacher comparisons use schema 2: exact public case specifications,
per-case run receipts (including v7 clean-cache restarts), fixture/target hashes,
reference file hashes and a run-scoped `teacher_evidence` snapshot. Historical
reports remain unchanged and cannot silently become schema-2 acceptance evidence.
Missing reference evidence fails; comparison never creates a new reference.
`quality_gate_pass` means non-inferiority, `bit_exact_pass` separately requires zero
logit errors. `make-qualification.py` requires both and complete source bindings.
