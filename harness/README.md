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
