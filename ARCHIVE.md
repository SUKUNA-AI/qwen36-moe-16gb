# ARCHIVE — what stayed in the previous workspace

The previous workspace (`${QWEN36_ARCHIVE}`, real path in `config/local.env`; placeholder
`/path/to/qwen36-workspace`) is kept **unchanged** as the read-only archive. This repository took
only the latest version of each component. Nothing was moved or deleted there. The installed
runtime (`${QWEN36_ROOT}`: release, venv, reports, traces, secrets) and the Windows install
(`${QWEN36_WIN_ROOT}`) were not touched either; they are not part of the archive.

## `outputs/qwen36-api/` (serving, first deployment)

| Left behind | Why |
|---|---|
| `README.md`, `STATE.md`, `CLAUDE-HANDOFF.md`, `START-CLAUDE.md`, `README-release-draft.md`, `BENCHMARKS.md`, `FP8-KV-NOTES.md`, `HANDOFF-SNAPSHOT.json`, `qualification.json`, `versions-draft.json` | Working notes with real hosts/paths; copied unchanged to `local/notes/serving/` (gitignored). Public summaries are in `docs/`. |
| `benchmark*.json`, `smoke*.json`, `https-from-caddy*.json`, `backend-from-caddy*.json`, `network-final.json`, `lan-exposure*.json`, `acceptance.json`, `artifact-manifest.json`, `crash-recovery.json`, `manual-restart.json`, `stop-final.json`, `gpu-cache-*.json`, `placement-*.json`, `metrics-*.json`, `runtime-final.json`, `download-receipt.json`, `deployment-owned-files.json`, `guard-test-result.json`, `secret-hygiene.json`, `caddy-secret-hygiene.json`, `cloudflare-setup.json` | First-deployment receipts (raw FreeToken baseline, 2026-10-13); superseded by the release-f1 results in `results/`. |
| `benchmark.py`, `summarize-benchmarks.py`, `set-strategy.sh`, `control-linux.sh`, `rollback.ps1`, `rollback-unit.py`, `test-crash-recovery.py`, `test-manual-restart.ps1`, `test-stop.ps1`, `test-metrics-integration.py` | First-deployment one-offs, superseded by `profile-runtime.py`, `harness/release.py` (rollback manifests) and the ops suites. |
| `install-cuda.sh`, `check-toolchain.sh` | Byte-identical duplicates of the `-13.4` scripts kept in `serving/`. |
| `bench-help.txt`, `serve-help.txt`, `*.log`, `__pycache__/` | Tool help dumps, logs, caches. |

## `outputs/qwen36-api/optimization/` (harness)

| Left behind | Why |
|---|---|
| Superseded run configs: `main-candidate-r2/-r3`, `main-combined*`, `main-layermajor*`, `main-lm-gdn-*`, `r2-attn`, `r2-base`, `long256-layermajor`, `profiles.candidate.json` and their `teacher-*` configs | Intermediate candidates of the release; the final ones are `main-release`, `long256-release`, `baseline64-final`. (Configs that `sync-v5.sh` requires were kept in `harness/`.) |
| `sync-{r1,r1b,r1c,lm1,gdn1,comb1,comb2,refill1,vram1,attn1,r2,r3}-freeze.sha256` | Freeze files of earlier syncs; `sync-final-freeze.sha256` kept. |
| Review / process records: `*-review.md`, `*-review*.json`, `api-guard-v2-*`, `backend-review*`, `engine-metrics-tests*`, `guard-tool-choice-*`, `harness-v5-*`, `keepalive-*-results.json`, `lifetime-job-*`, `lookup-*`, `native-sync-audit.*`, `nsight-prep-results*`, `ops-*-tests.json`, `ops-live-audit.json`, `paired-review-*`, `prefix-*` results, `release-gate-*`, `release-provenance-results.json`, `release-readiness-checklist.*`, `release-safety-results.json`, `remaining-layers-review.*`, `retain-snapshot-*`, `snapshot-and-tools-*`, `soak-gate-*`, `model-baseline-analysis*`, `analyze-model-nsys-test-results*`, `acceptance-cpu-review*`, `final-profile-review.*`, `benchmark-handoff-stop.*`, `relocation-to-E-receipt.json`, `windows-wheel-audit.json`, `gdn-fp32-*`, `long256-fallback64-plan*`, `long256-budget-review.md`, `literature-2026.md`, `phaseA-review.md`, `prep-audit-synthesis.md`, `speculation-cpu-review.md`, `*-acceptance.md`, `WINDOWS-ASSESSMENT.md`, `harness-v5-runbook.md`, `lookup-runbook.md`, `tool-task-v4-diagnostic.json` | Development history of the harness and of the reviews. The owner's state logs (`STATE.md`, `OPERATIONS-release-draft.md`, `HANDOFF-OPS-TO-CLAUDE.md`, `README.md`, `WINDOWS-ASSESSMENT.md`) are copied to `local/notes/harness/`. |
| Applied patches: `backend-review.patch`, `guard-tool-choice.patch` (+README), `retain-chunk-snapshot.patch` (+README), `prefix-quality-README.md` | Already applied to the files in this repository. |
| One-off diagnostics: `diagnose-mtp.py`, `diagnose-prefix-format.py`, `diagnose-prefix-snapshots.py`, `diagnose-quality.py`, `freeze-retain-snapshot.py`, `review-snapshot-and-tools-cpu.py`, `summarize-gdn-snapshot.py`, `prepare-tool-choice-patch.py`, `backend-review-export.py`, `backend-review-tests.py`, `tune-prefill.py`, `spec-bench.py` | Investigation scripts for decisions that are closed. |
| `cuda-*/`, `pcie-transfer/`, `scale-audit/`, `fp32-global/` | Per-component numeric summaries and agent scripts; the shipped kernels' latest sources/tests are in `kernels/`, selected results in `results/`. |
| `eval-v1/` | Private case set: copied to `harness/eval-v1/` (gitignored), never published. |

## `work/`

| Left behind | Why |
|---|---|
| `FreeToken/` | Pristine upstream clone (base `cac247a`). |
| `FreeToken-opt/.git` | Upstream history; the working tree (latest) is `freetoken/`, the diff is listed in `freetoken/MODIFICATIONS.md`. |
| `agents/*` (except the final gate driver) | Agent working directories, drivers `*-driver*.sh`, `before/` backups, reviews/refutations. `agents/final-gates/` scripts are in `harness/gates/`; its `decision-f1.json`/`result.json` in `local/notes/gates/`. |
| `cuda-gdn/patch-src*`, `*-integration.patch`, `wf-fp8gemv/{orig,tree,cache,raw,logs,results}`, `wf-gather*`, `wf-fp8gemv-review`, `gather-vram`, `layer-major`, `tool-choice-guard`, `host-sync` | Integration copies and patches already applied to `freetoken/` (host-sync is superseded by `next-update/synclean-refill`). |
| `cuda-attention`, `cuda-decode-prefetch`, `cuda-gdn-layout`, `cuda-kv`, `cuda-layer-fusion`, `cuda-moe-decode`, `cuda-nvfp4` (pre-bit-exact), `cuda-prefill-transport`, `cuda-transport`, `cuda-fp8-gemv` (v1), `pcie-transfer`, `scale-audit` | Experimental or superseded kernel work; not shipped. Findings are summarized in `docs/`. |
| `functional-fp32`, `l256-deviation`, `fp32-global` | Investigations; conclusions in `docs/` and `results/experiments/`. |
| `next-update/*/` raw outputs: `s1/`, `s2/`, `shim/`, `*.jsonl`, `*.journal.txt`, `orig/`, `tree/`, `patch/` full-file copies, `cpu-smoke/`, `gpu-results/`, logs, one-off `peek*/inspect*/probe*` scripts | Raw run data (may contain prompts/answers) and duplicates; the patches, configs, plans and summaries are in `next-update/`. |
| `next-update/{commit-race,dashboard,issue-drafts,long256-soak,metrics,nsight-final,tool-use,wsl-interop}` | Results and drafts: selected, sanitized copies are in `results/` and `docs/upstream-notes/`. |
| Edge/DNS setup: `cloudflare-*`, `setup-cloudflare.py`, `check-edge-secrets.py`, `discover_*.py`, `inspect_edge.py`, `final-edge-check.py`, `router-login.html`, `forge-before.html`, `check_download_headers.py` | Owner-specific infrastructure; not copied. |
| `lifetime-job-*`, `release-cpu-*`, `*.log`, install/download logs, `historical-install-cuda13.0.sh`, `remove-owned-cuda13.py`, `prepare-64k.sh`, `profile-runtime-before-final-review.py`, `state-before-final.md`, `windows-models-response.json`, `checkpoint-*.json`, `cuda-13.4.1-manifest.json` (dup) | Test scratch, logs, superseded scripts and duplicates. |

## Re-sync needed

- `work/next-update/gather-prefetch/` was still being written when this repository was staged;
  copy its final state into `next-update/gather-prefetch/` (sanitize paths) when that work ends.
