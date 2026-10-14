# MANIFEST

Repository staged on 2026-10-14/15 from the previous workspace, which stays unchanged as the
archive (`${QWEN36_ARCHIVE}`; see `ARCHIVE.md` for everything that was left there). Source
paths below are relative to that archive root. Latest version of each component only.

## Top-level items

| Item | Owner (staging) | Source in the archive | Content |
|---|---|---|---|
| `README.md` | docs | new | Project overview, headline results, build/run instructions, limitations. |
| `LICENSE`, `NOTICE` | layout | `work/FreeToken-opt/LICENSE` (text) / new | Apache-2.0 for the whole repository; NOTICE credits upstream FreeToken. |
| `MANIFEST.md`, `ARCHIVE.md` | layout | new | This file; what stayed in the archive. |
| `.gitignore` | layout | new | Keeps `local/`, `config/local.env`, secrets, private eval data, teacher fixtures/`teacher*.jsonl`, raw reports, traces (`*.nsys-rep`, `*.sqlite`, `*.ncu-rep*`), caches and build outputs out of git. |
| `freetoken/` | layout | `work/FreeToken-opt/` (working tree; no `.git`, caches, `*.so`, build dirs) | The FreeToken fork exactly as qualified for release f1 (upstream base `cac247a`), plus two default-path sanitizations. `freetoken/MODIFICATIONS.md` lists every modified/added file and the env switches. |
| `kernels/` | kernels | `work/cuda-gdn/`, `work/wf-fp8gemv/{src,build.sh,scripts}`, `work/cuda-nvfp4-bitexact/`, attention-gate source (see `kernels/README.md`) | Sources, build scripts and bit-exact test harnesses of the CUDA libraries (GDN ABI 2 incl. FLA fused kernels, FP8 GEMV v2, bit-exact NVFP4 MoE prefill, fused attention gate). No prebuilt `.so`. The decode gather kernel lives in `freetoken/python/freetoken/kernel/csrc/jit/decode_gather.cuh`. |
| `serving/` | layout | `outputs/qwen36-api/` (operational files), `outputs/qwen36-api/optimization/qwen36.cmd` -> `serving/bin/` | Runtime/launcher, profile runtime + `profiles.json` (released f1), API guard, metrics, readiness/keepalive, Windows supervisor/management, installers, `edge/` (Caddy site template + staging helpers). See `serving/README.md`. |
| `harness/` | layout | `outputs/qwen36-api/optimization/` (sync-v5 allowlist, CPU suites, release/ops tooling, run configs), `work/agents/final-gates/` -> `harness/gates/` | Experiment server, teacher-forcing quality gate (quality.py v7), paired matrix, soak, Nsight flow, release and ops tooling, final gate driver. See `harness/README.md`. |
| `harness/eval-v1/` | layout | `outputs/qwen36-api/optimization/eval-v1/` | **Private** eval case set; present locally, gitignored, never published. |
| `next-update/` | layout | `work/next-update/{short-route,fla,cache-reset,fp8kv,synclean-refill,gather-prefetch}` + `AGENT-RULES.md` | Pending patches, configs, plans, pre-registered rules and summaries (raw run data excluded). `gather-prefetch/` was still in progress: re-sync. |
| `results/` | docs | `work/next-update/{dashboard,metrics,nsight-final,tool-use,long256-soak,...}`, `outputs/qwen36-api/optimization` run reports (sanitized) | Release-f1 numbers: dashboard HTML, metrics.json, Nsight summary + levers, matrix comparison, context gates, quality, soak, 64K apples-to-apples, FP8 KV screens, 256K short test, tool-use, findings. |
| `docs/` | docs | new + `work/next-update/issue-drafts` -> `docs/upstream-notes/` | Architecture, results with provenance, methodology, roadmap; upstream notes are local drafts that were never posted. |
| `config/local.env.example` | layout | new | Template of the site values (placeholders). |
| `config/local.env` | layout | new | **Gitignored.** Real site values (paths, hosts, user names). Secrets are referenced by directory only. |
| `tools/scan-publish.py`, `tools/scan-allowlist.txt` | layout | new | Secret/PII scan of `git ls-files` (site literals are read from the gitignored config); justified allowlist. |
| `local/` | layout | see below | **Gitignored.** Owner-only material. |

### `local/` (gitignored, not published)

- `local/serving/**`, `local/harness/**`, `local/next-update/**`: the unchanged originals (real
  paths, hosts, SSH alias) of every tracked file that was sanitized — 283 files. Use these for
  the owner's own deployment until the TODO below is done.
- `local/notes/serving/`, `local/notes/harness/`, `local/notes/gates/`: state logs, handoff
  notes, release receipt (`qualification.json`), f1 decision and result.
- `local/scan-literals.txt`: owner-specific literals checked by `tools/scan-publish.py`.
- `local/stage-log.json`: per-file staging log (source, destination, sanitized or not).

## Sanitization rule

Tracked copies replace site values with documented placeholders (table in `serving/README.md`):
`/home/USER/...`, `USER`, `C:\path\to\LLM\...`, `/path/to/qwen36-workspace`, `llm.example.com`,
`forge.example.com`, `EDGE_SSH_HOST`, `192.0.2.x` / `203.0.113.x`. Secrets were never copied:
the API key stays in `${QWEN36_SECRETS_DIR}` (`api.env`) and is read at run time. Private eval
data, teacher fixture targets, `teacher.jsonl` logit references and raw reports with
prompts/answers are not tracked.

## TODO — path parameterization (not done in this staging)

The tracked scripts still assume the old workspace layout (`outputs/qwen36-api`,
`outputs/qwen36-api/optimization`, `work/FreeToken-opt`, `work/cuda-*`) and carry placeholders
instead of reading `config/local.env`. Until this is done, CPU suites and `sync-v5.sh` are not
expected to run from this repository; the archive + native tree remain the working setup.
Places to repoint (repo layout: `freetoken/`, `serving/`, `harness/`, `kernels/`):

1. `harness/sync-v5.sh`: `NATIVE_DEFAULT`, `WORKSPACE_DEFAULT`, `LOCK_DEFAULT`, `OLD_WORKSPACE`;
   `src_tree=$workspace/work/FreeToken-opt/python/freetoken` -> `freetoken/python/freetoken`;
   `out_dir=$workspace/outputs/qwen36-api` -> `serving`; `opt_dir=$out_dir/optimization` -> `harness`;
   `LIBS` source `work/cuda-nvfp4-bitexact/build/libqwen_nvfp4_bitexact.so` -> `kernels/nvfp4-moe/build/...`;
   `snapshot=$out_dir/HANDOFF-SNAPSHOT.json` (now in `local/notes/serving/`); sandbox guard path;
   manifest entries are workspace-relative (`outputs/qwen36-api/...`), so `CONFIG_JSON` and the
   freeze format change together. Prune `CONFIG_JSON` to the configs still used.
2. Freeze files (`--print-freeze` output, `harness/sync-final-freeze.sha256`): regenerate after (1);
   the archived freeze uses old relative paths.
3. `harness/sync.sh` (blind legacy sync): repoint or drop.
4. `harness/release.py`: `PROJECT`, `OUTPUT=PROJECT/'outputs/qwen36-api'`,
   `RUNTIME_SOURCE=PROJECT/'work/FreeToken-opt/python/freetoken'`, pinned library paths
   (`private-cuda-gdn/pinned`, `private-fp8gemv`, `private-libraries`); mirrored layouts in
   `test-release-cpu.py`, `test-release-qualification-cpu.py`, `make-qualification.py`,
   `export-release-evidence.py` (`WORKSPACE=parents[3]`, `work/FreeToken-opt`) and
   `test-export-release-evidence-cpu.py`.
5. `harness/release-management.ps1`: `$root` (Windows install root), `$output=Split-Path $PSScriptRoot`
   (assumes the harness is a child of the serving dir), `bin\qwen36.cmd` source;
   `test-release-management-cpu.ps1` scratch `work\release-cpu-*`.
6. Harness workspace variables: `bench.py` and `eval-v1.py` `WORKSPACE_OPT`; `eval-v1.py`
   `HERE.parent.parent.parent/'work/FreeToken-opt'`; `qualify.py` `workspace`; `start-experiment.sh`
   `workspace=` (literal asserted by `test-paired-contract-cpu.py`); `lookup-qualification.py`
   `WORKSPACE`/`WORKSPACE_OPT`/`FORK_SOURCE`; `test-engine-metrics.py` and `ops-readiness-tests.py`
   (`parents[3]`, `outputs/qwen36-api`); `test-gdn-snapshot-cpu.py`, `test-retain-snapshot-cpu.py`,
   `test-lookup-qualification-cpu.py` (`here.parents[2]/'work/FreeToken-opt'`); `test-soak-cpu.py`
   (`work/agents/soak-gate/before`, an archive-only backup); `test-ops-live-owner-fault-cpu.ps1`
   (`work\agents\ops-live`); `test-lifetime-job.ps1` (`..\..\..`); `ops-control-tests.ps1`
   (`outputs\qwen36-api`, `work\cuda-transport\ops-test-scratch`); `CPU-checks.sh` (`PYTHONPATH`,
   test paths); `profile-gdn-snapshot.sh`; `test-paired-contract-cpu.py` (2 known config-layout
   failures already in the archive).
7. Gate drivers `harness/gates/*.sh`: `W`, `O=$W/outputs/qwen36-api/optimization`, `N`, `ROOT`, `LOCK`;
   `make-decision-f1.py` and `selftest/make-result.py` paths.
8. Serving (installed files): `start-inference.sh` (`ROOT`, checkpoint receipt path),
   `qwen36-inference.service`, `profile-runtime.py` `ROOT`, `readiness.py` (`ROOT`, public URL,
   `EDGE_SSH_HOST`), `keepalive.py` `RECORD_ROOT`, `metrics_summary.py`, `placement.py` and
   `request_metrics.py` (`QWEN36_DATA_ROOT` default), `supervisor.ps1`/`manage.ps1`/`watch-supervisor.ps1`
   (`$Root` default, Linux paths, tunnel host), `bin/qwen36.cmd`, `install-runtime.sh`/`setup-service.sh`
   (`DELIVERABLES`), `check-kernels.sh`. Suggested: Linux `QWEN36_ROOT` default `~/qwen36-api`,
   Windows `$Root=$PSScriptRoot`, site values from `${QWEN36_ROOT}/config/site.env`.
9. `next-update/*` scripts: same `W`/`N` variables.
10. `kernels/*/build.sh`: owned by the kernels staging; see `kernels/README.md` for its variables.

## Secret / PII scan

`python3 tools/scan-publish.py --json <out>` over `git ls-files` (run from WSL). Rules: private
keys, AWS/GitHub/OpenAI-style/HF/Slack tokens, JWTs, `Bearer` values, secret assignments,
Cloudflare token variables, e-mail addresses, routable IPv4, `/home/<user>`, `C:\Users\<user>`,
`/mnt/<drive>/`, drive-rooted site directories (`<drive>:\` + LLM/Codex/WSL/Projects), plus the site literals from
`config/local.env` and `local/scan-literals.txt` (domain, user names, owner e-mail, public IPs,
archive path). Result and allowlist: see the scan section of the staging report; the allowlist
with justifications is `tools/scan-allowlist.txt` (version numbers that look like IPv4, synthetic
test fixtures, an already-sanitized placeholder path).
