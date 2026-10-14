# Final release gates: plan (DRAFT, not run)

Agent `final-gates`, 2026-10-14. CPU-only drafting. No GPU, no units, no sync, no qualify/matrix/soak/release runs,
no edits under `outputs/` or `work/FreeToken-opt/`. Every GPU number below is an **estimate** taken from earlier logs.
`[UNSURE]` marks each place where I had to guess. The open questions are collected in section 7.

Files in `work/agents/final-gates/`:

| File | Role |
|---|---|
| `final-gates.sh` | Gates A, B, D, C, E, Z. The parent runs it. It works in a run-id namespace and can be re-run. |
| `final-install.sh` | F: production install and live checks. Run it **manually** after reviewing A-E. |
| `final-prep.sh` | Optional pre-steps P1-P3. Check-only by default; `--apply` writes. |
| `final-helpers.py` | Stdlib-only CPU helpers. |
| `parent-lockwrap.py` | Fairness wrapper for matrix, tool-pilot and compare-ab. |

## Handoff state (2026-10-14 12:30Z, after the parent's CPU assembly)

- **Decision: gather is IN.** Run `final-gates.sh --gather in` and later `final-install.sh --gather in`.
- **P1 done by the parent.** Patches 01-08 are at `after_sha256`.
- **P3 done by the parent.** The venv prebuilt `freetoken__decode_gather_tma_8_3_16384.so` (sha256 ad43503a981b…) is bound in the workspace manifest with recipe digest `3a8377b55f3bc6fedb1bad7645860d98e5226b95f7368992674f5af188204441` (83 modules). `final-prep.sh --gather in` (check mode) confirms the module is `prebuilt`.
- **Patch 09 + `layer_major_attn_batch` applied by the parent** (profiles.json sha a144a1919948…). Canonical sha256 matches my builder: main128 542e0fe6…, long256 c6f8ad91…, baseline64 982cd958…. F1 rewrites profiles.json from the draft with the same canonical content plus the qualified flags (only the formatting changes; a backup is kept).
- **STILL MISSING: the allowlist names** `'teacher-main-release'` and `'teacher-long256-release'` are not in `work/FreeToken-opt/python/freetoken/engine/qualification.py`. Add them before A, e.g. with `final-prep.sh --gather in --apply`, which only performs P2 now. A0 stops without them.
- **Dry-run of `final-gates.sh --gather in` against this state (12:29Z, read-only, scratch in `selftest/`).** Every precondition passes except those 2 allowlist names:
  - patches OK; 4 library sha256 OK; gather checks OK;
  - mapping PASS with the patched outputs validator and make-qualification;
  - freeze check: exactly the 6 allowed MISMATCH lines, `RUNTIME 467 files`;
  - `sync-v5 --dry-run` rc 0, 0 freeze mismatches.
- **Expected until the final sync:** the current native tree reports the gather module as `unknown` (native manifest not synced yet), so `kernel_provenance check` on native fails. The A4 sync fixes this, and A6 then requires `prebuilt`. Any experiment started before the final sync sees this state.

## 0. Inputs and names

**Decision input `--gather in|out` (parent).**
- `main-release.json` = `r2-attn.json` plus `FREETOKEN_DISABLE_JIT=1`. With `in` it also gets `QWEN36_DECODE_GATHER=tuned` and `TVM_FFI_CUDA_ARCH_LIST=12.0`.
- `long256-release.json` = the same env with context 262144 and memory_ratio 0.73. This matches the draft "same switches as main128", and it is required: the L4 identity check allows only context, memory_ratio and qualified to differ from the arm.
- The exact bytes for both modes are in `result.json` under `needed_configs`.
- `final-helpers.py configs` derives both files from the pinned `r2-attn.json`. With `in` it also cross-checks against `main-candidate-r3.json`. It never overwrites a file whose content differs.

**Allowlist names** (in `engine/qualification.py`, before the final freeze): `teacher-main-release`, `teacher-long256-release`.

**Run namespace.** `T=final-<run-id>` (default `final-f1`). Fixed names:
- `reports/teacher|functional|quality-{main-release,long256-release}.json` (qualify.py names them after the candidate)
- `reports/$T-qrun-{teacher,functional}-<cand>.json`
- matrix `$T-m`
- comparison `reports/$T-m-main-release-vs-bf128-comparison.json`

Attempt-suffixed names (`-aK`):
- tool pilot `$T-tool-aK`
- gate tags `$T-c128-aK`, `$T-l256-aK`, `$T-b64-aK`
- soaks `soak-pilot-$T-aK`, `soak-$T-aK`
- state file `reports/final-gates-<run>.steps` (append-only; a re-run skips steps that already PASSed)
- log `reports/final-gates.log`

**Matrix baseline arm: `bf128`.**
- Release checklist C1 says compare-ab is run against bf128. compare-ab's default `--baseline` is `bf128`. bf128 was the baseline in final3-v5, final4-r1 and final4-r1b.
- What production runs today is the legacy 64K `start-inference.sh`. It is not an experiment profile, and its 64K context cannot serve the 65536 and 120000 bench lengths.
- `[UNSURE]` whether the parent also wants `main-final` as a third arm (see Q2).

## 1. Parent pre-steps (CPU, about 10 min; `final-prep.sh --gather X`, then `--apply`)

Run these only after r3-driver21 has finished: they change frozen or synced bytes.

- **P1.** Apply release-tooling patches 01-08 with `patch -p1` from the workspace root and verify `after_sha256`. Patch 09 is not applied: F writes `profiles.json`.
- **P2.** Insert both allowlist names after the anchor `'teacher-main-combined',`, preserving line endings (r2/r3 idiom).
- **P3** (`--gather in` only):
  - Copy `~/.cache/freetoken/provenance-jit/freetoken__decode_gather_tma_8_3_16384/<digest24>/…so` into the venv `freetoken_kernel_cache/jit_cache/<mod>/<mod>.so`. `<digest24>` is the CURRENT recipe digest; the private build dir `3a8377b55f3bc6fedb1bad76` exists from r3.
  - Merge `work/wf-gather/scripts/manifest_entry.py --so <that .so>` into `kernel/kernel_cache_manifest.json` (indent=1, sort_keys, LF, as today).
  - Verify with `kernel_provenance.py inventory`: the module status must be `prebuilt`.
  - `[UNSURE]` release-tooling recorded recipe digest `ae842c3e…`, which is the pre-fix `.cuh`. The shared::cta fix changed the digest, so P3 always uses the digest computed from the current tree.

Then run `final-gates.sh --gather X --dry-run`. It is CPU-only and read-only, with scratch under `$FINAL_GATES_SCRATCH` or `/tmp`. It covers preconditions, derived configs, mapping, freeze check, `sync-v5 --dry-run` and current provenance.

## 2. Ordered steps of `final-gates.sh` (real mode)

C runs **after** D. Two things depend on the D matrix manifest, which exists only once the matrix has started:
- the long-context gates' identity check (`run-long256-gates-r1b.sh --identity-manifest/--identity-arm`, required for PASS);
- the soak's `--expected-identity`.

So the execution order is A, B, D, C, E (the brief listed A-E).

GPU minutes are wall minutes holding the GPU lock. The sources are UTC log timestamps.

| # | Step | Kind | GPU min (estimate) | Basis |
|---|---|---|---|---|
| A0 | Preconditions: units inactive; patches 01-08 at `after_sha256`; both allowlist names; sha256 of the 4 ctypes libraries (GDN a8a806ed…, FP8 685c78cf…, NVFP4 b7fea227…, gate f654f87a…); draft profiles sha dce3a145…; tools present; gather: no `.shared::cluster`, manifest entry, venv prebuilt | hard | 0 | - |
| A1 | `configs --write` (O_EXCL, or verify identical) | hard | 0 | - |
| A2 | `mapping`: staged `make-qualification.run_equivalent_profile(run cfg and teacher variant)` == `production_semantics(profile)` for main128/long256/baseline64, plus `required_kernel_modules` (gather only if in) | hard | 0 | self-tested on staging copies: both modes PASS, 3 negative controls FAIL as expected |
| A3 | `sync-v5 --print-freeze` with 2 `--extra-config`, then `make-freeze --check`. Only these MISMATCH lines are allowed, and only when workspace == sync-freeze: profile-runtime.py 0f30ee09c751, release.py 6d8328c3d48e, make-qualification.py 6398c923d3a7 (release-tooling exceptions), run-experiment.sh a06b0b29a178, bench.py 204d28710ef0, screen-chunk.py 798a2728f941 (r3 list). Any other MISMATCH stops the run. | hard | 0 | - |
| A4 | Sync label `final`: dry-run, apply (under the lock), verify. Then check native configs == workspace, allowlist present in native, library sha256 after sync. | hard | 0.3 | r2: 11:14:07→11:14:33 |
| A5 | `test-paired-contract-cpu`, `test-soak-cpu`, `CPU-checks.sh` (guard v2), and the 4 release-tooling suites from /tmp with `-I -B` under nice/ionice. Minimum test counts: profile-runtime 88, release-qualification 56, release 13, export-evidence 13. | hard | 0 (CPU about 3 min) | r1 step3 23 s; suites about 30 s |
| A6 | `kernel_provenance check` on native plus venv cache. All modules must be `prebuilt`, `_pinned_tensor` prebuilt, and with gather in, `freetoken__decode_gather_tma_8_3_16384` prebuilt. | hard | 0 | - |
| A7 | Native `run-experiment.sh` exports the exact `GLIBC_TUNABLES`; `check-glibc-tunables --selftest` | hard | 0 | - |
| B1 | `qualify.py --candidate main-release` (teacher-forced vs bf128-R1 logit reference, functional, PPL). Strict qcheck uses the make-qualification rule: expected == observed > 0, max_abs = max_rel_l2 = 0, finite, forced targets, quality_gate_pass, ppl_gate_pass, no new functional failures. Teacher and functional run receipts are extracted and must equal the config. | hard (STOP) | 4-6 | main-final 5.4, r2 4.6, attn 3.2 |
| B2 | `qualify.py --candidate long256-release` (same checks; the teacher fixture includes the long-120000 case) | hard (STOP) | 7-10 `[UNSURE]` | no earlier 256K qualify; 256K load about 1.5 min ×2 |
| D1 | Tool pilot on bf128 (`--repeats 3`) through `parent-lockwrap.py` | gate (a FAIL stops: no manifest) | 2 | final4-r1b 1.7 |
| D2 | Matrix `--profiles bf128,main-release --pairs 5 --output 128 --tool-pilot …` (10 arms). One automatic resume if an arm fails at preflight, start or readiness. Records D.matrix (rc 0 PASS, 3 INVALID, else FAIL) and D.manifest. | gate | 35-42 | final4-r1b 70 min/20 arms; final3-v5 4.2 min/arm |
| D3 | `compare-ab --candidate main-release --baseline bf128` (V6: 9 gating cases, Bonferroni family 9). Records the verdict plus contract_pass and five_pairs. | gate | <1 | - |
| C0 | main128 context gates: `run-long256-gates-r1b.sh --profile main-release --context 131072 --length 120000 --expected-long-prompt 120067 --expected-boundary-prompt 130396`, identity arm main-release, functional reference `functional-main-release.json` | gate (main128) | 6 `[UNSURE]` | L4 7.4 at 256K; token counts are the CPU-verified long256-plan token budget, and the generator function hashes are unchanged |
| C1 | L4: long256-release, 262144/250000, identity arm main-release | gate (long256) | 6-8 | r1b L4 7.4 |
| C2 | F1: baseline64-final, 65536/60000, identity arm bf128, known limitation long-facts-60000 SHA 93e766db… | gate (baseline64) | 4 | r1b F1 3.6 |
| E1 | Soak pilot 30 s on main-release (start, readiness, `check-glibc-tunables --unit qwen36-experiment`, then soak.py with the manifest identity), then soak-analysis. PASS needs `pilot_view.all_case_types_passed` and tunables PASS. | gate | 2.5 | r1b 2.3 |
| E2 | Soak 7200 s (tunables pre, soak.py, tunables post), then soak-analysis. PASS needs verdict PASS, `two_hour_execution_gate_pass`, and tunables PASS before and after. | gate | 123-126 | r1b 00:24→02:26 = 122 |
| Z | Writes `final-gates-<run>-summary.json` (per-profile gate table, releasable = main128 passes) and the O_EXCL `final-gates-<run>-decision-draft.json` (evidence, run_receipts and the three `cuda_*_evidence` blocks filled in; known_limitations carry a TODO for the parent) | CPU | 0 | - |

**Total: about 196 GPU minutes (3 h 15 min; range 3 h 00 min to 3 h 40 min). Wall time about 3 h 25 min plus lock waits.**

The 2 h soak is about two thirds of that. The longest step before it is the matrix at about 40 min.

**Failure policy**
- **A or B:** STOP (sync, CPU and teacher failures are hard).
- **D1 FAIL:** STOP, because nothing downstream has an identity reference.
- **Matrix fails before writing the manifest:** STOP.
- **C identity mismatch (77):** STOP. Other C failures are logged and the run continues.
- **E** runs only if main128's earlier gates (A, B.main, D.*, C0) are OK. Otherwise E is logged as `NOT_RUN` with the reason; `--soak-anyway` overrides this.
- `--skip-c0` records `SKIPPED_BY_PARENT`, which does not block (see Q6).
- **Exit code:** 0 means main128 is releasable.

**Per-profile gates** (`final-helpers.py summary`):

| Profile | Required |
|---|---|
| main128 | A, B.main, D.pilot, D.matrix=PASS, D.compare ∈ {PASS, INCONCLUSIVE}, C0 ∈ {PASS, SKIPPED_BY_PARENT}, E.pilot, E.soak=PASS |
| long256 | A, B.long256, D.manifest, C1=PASS |
| baseline64 | A, D.manifest, C2 ∈ {PASS, PASS_WITH_KNOWN_LIMITATION} |

## 3. `make-qualification` evidence that the draft fills in

For each CUDA library switch (`cuda_nvfp4_prefill`, `cuda_gdn`, `cuda_fp8_gemv`), the decision needs:
`{"run_receipts": [...], "bitexact_reports": [...]}` under `cuda_nvfp4_evidence`, `cuda_gdn_evidence`, `cuda_fp8_gemv_evidence`. Every path must also be listed in `evidence`.
- **Bit-exact report:** `reports/quality-<cand>.json` (compare-quality output; accepted by patched make-qualification `check_quality_bitexact_report`).
- **Run receipt:** `reports/$T-qrun-teacher-<cand>.json`. This is the `run` object of the teacher report; RUN_DIAGNOSTIC_ENV allows the teacher env.
- **Profile `run_receipts`:**
  - main128: `$T-qrun-functional-main-release.json`, `$T-c128-aK-run.json`, `soak-$T-aK-run.json` (soak.py writes this copy)
  - long256: `$T-qrun-functional-long256-release.json`, `$T-l256-aK-run.json`
  - baseline64: `$T-b64-aK-run.json`

All of them must come from runs on the synced 'final' tree, because make-qualification compares `source_files` against the staged runtime bytes.

## 4. `final-install.sh` (F, manual, about 25-45 min)

The whole window holds the GPU-exclusive lock (ops-live-runbook A1). `--check-only` runs F0-F4 read-only.

| # | Step | Time |
|---|---|---|
| F0 | Summary releasable for this run id and gather; reviewed decision (`decision-check`: no TODO, only profiles whose gates passed, main128 included, all evidence present); speed INCONCLUSIVE accepted only with `--accept-inconclusive-speed`; units inactive; `qwen36-inference` not enabled | <1 min |
| F1 | `outputs/qwen36-api/profiles.json` = draft + `layer_major_attn_batch: 2` (main128, long256; the key and value come from patch 01 `VALUE_FIELDS`), with `decode_gather` deleted if gather is out, and `qualified` exactly = the decided profiles. Validated with the staged validator; the old file is backed up to `work/agents/final-gates/backup/`. | <1 min |
| F2 | `make-qualification --dry-run`, then `--output outputs/qwen36-api/qualification.json` (O_EXCL; an existing identical receipt is reused, a different one causes a STOP) | 1-3 min `[UNSURE]` (hashes venv and checkpoint) |
| F3 | `release.py install --check`, then `install` (prints the rollback manifest), then `release.py rollback --manifest M --check` | 2-5 min |
| F4 | `release-management.ps1 -CheckOnly`, then install, then `-RollbackManifest M -CheckOnly` (Windows interop; `--no-windows` logs NOT_RUN) | 1 min |
| F5 | See the live sequence below | 15-25 min (+10-15 with `--live-extra`) |

**F5 live sequence**
1. `qwen36 start --profile main128` through `cmd.exe` on `C:\path\to\LLM\qwen36-api\bin\qwen36.cmd`.
2. `readiness.py --edge` for up to 600 s; requires ready, backend, tunnel and https ready, model, profile, profile_verified, and the instance id equal to the tunnel's.
3. `ops-live-probe identity --expect-instance`.
4. `check-glibc-tunables --unit qwen36-inference`.
5. `network-acceptance.py` through Caddy: pass, exactly 60 rows, same_instance equal to the instance from step 2, tls_verified, `existing-forge-site` row, `external_e2e=NOT_RUN`.
6. From the LAN: forge.example.com returns 200, llm `/v1/models` without a key returns 401, `/health` returns 404.
7. `qwen36 restart`, then readiness and `identity --different-from`.
8. `qwen36 stop`; `qwen36 status` must exit non-zero; units inactive.
9. Optional `--live-extra`: long256 and baseline64 start, identity, stop, then main128 is re-selected.
10. No autostart: the Linux unit is not enabled and there is no HKCU Run `Qwen36Api` entry.

The model is left stopped. Rollback is never automatic: on failure the script prints the exact rollback commands.

## 5. What is verified now (CPU)

- `bash -n` passes for all three scripts; the Python helpers parse with the ast module; no CRLF.
- The helper was self-tested on real staged files (outputs under `work/agents/final-gates/selftest/`):
  - `configs` (both modes; re-check is idempotent);
  - `mapping` PASS for gather in and out, against the release-tooling staged profile-runtime and make-qualification;
  - negative controls FAIL as expected: gather-in configs against gather-out profiles, `main-candidate-r2` (FLA), and `main-combined` (no attention batch);
  - `profiles` validates with the staged validator; canonical sha256 for gather in: main128 542e0fe6…, long256 c6f8ad91…, baseline64 982cd958… (unchanged);
  - `qcheck` PASS on quality-main-candidate-r2 and quality-main-final;
  - `extract-run` PASS on the r2 functional and teacher reports and FAIL with a wrong config;
  - `provenance` PASS on kernel-provenance-attn1, and FAIL when the gather module is required, which is correct today because it is absent;
  - `summary` and `decision-check` on a synthetic steps file (TODO is refused).
- `final-gates.sh --gather in --dry-run` ran against the post-patch workspace (see the handoff state above). The only failures were the 2 missing allowlist names.
- `final-prep.sh --gather in` ran in check mode: P1 applied, P2 pending, P3 already done.

## 6. NOT_RUN

- Every GPU step (A4 apply, B, D, C, E).
- The real dry-run against the patched tree: the patches are not applied yet.
- `final-prep.sh --apply`, all of `final-install.sh`, Windows interop calls, live ops, network acceptance.
- No STATE.md entry; GitHub untouched.

## 7. Open questions / `[UNSURE]`

1. **Order.** C runs after D because the identity manifest is the D matrix manifest. A cheaper alternative would be a CPU-built identity reference from `bench.current_run_spec`, but `source-identity-check-r1b.py` expects a V6 matrix manifest, so I did not do it.
2. **Baseline arm.** I chose bf128 (checklist C1, compare-ab default, earlier matrices). If the parent wants a "vs previous candidate" claim, add `main-final` as a third arm: about +20 min, and the Bonferroni family becomes 18.
3. **Tool pilot profile.** bf128, following harness-v6-runbook step 7. Piloting main-release would catch a candidate tool regression before the 40-min matrix; the matrix runs tool_task on every arm anyway.
4. **`QWEN36_BENCH_WARMUP_S`.** Not set for the matrix: the V6 contract default is 0. The r2/r3 screens used 60. Confirm.
5. **C1/C2 functional reference.** I use `functional-main-release.json` (driver4 used main-final's functional for baseline64 too). If F1 fails only on `functional_short_hashes_identical_to_reference`, rerunning F1 with `functional-bf128.json` (the bf128 arm's own functional from R1) is an owner call.
6. **C0** (main128 131072/120000 context and boundary gates) is my addition for checklist C5.1 coverage of the default profile; about 6 min. `--skip-c0` drops it without blocking.
7. **VRAM at 256K/.73** with FP8 GEMV (-252 MiB) and attention batching has not been measured. The budget-calc preflight (exit 76 means nothing started) models R1b mechanics only. L4 enforces the 0.5 GiB floor.
8. **Gather recipe digest** changed with the shared::cta fix (`ae842c3e…` became `3a8377b55f3b…`). The parent bound the new one (verified). A6 re-checks it on the native tree after the sync.
9. **Windows interop.** I did not verify that `qwen36` and `powershell.exe` behave the same when started from WSL: the supervisor is launched with Start-Process hidden, and it is unknown whether WSL's interop job kills it. The fallback is `--no-live` plus ops-live-runbook phases B and D in PowerShell.
10. **`profiles.json`** deletes `decode_gather` when gather is out (the draft rationale is "absent = off") instead of setting it to false.
11. **Known limitations.** The decision text needs the parent's wording. The draft carries over the release-prep fp32-basis, special-anchor and API-contract items plus AUTO facts, and starts with a TODO line that F refuses until it is removed.
12. **qualify.py re-runs** overwrite `teacher/functional/quality-<cand>.json`, so B is skipped once it has PASSed. A matrix failure that cannot be resumed needs a new `--run-id`, which repeats B (about 15 min).
13. **Sync label 'final'** can be applied once. After legitimate workspace drift, use `--sync-label final2` and a new `--run-id`.
14. **`network-acceptance`** row count: 60 is assumed from the runbook. Any other count is flagged FAIL ("pass_but_unexpected_shape").
15. **`FREETOKEN_DISABLE_JIT=1`** is in both candidate configs (and so in the matrix and soak arms). A missing prebuilt makes the start fail, which is intended. bf128 and baseline64-final do not carry it, because the identity check against bf128 forbids env changes; make-qualification accepts that ("recommended, not required").
