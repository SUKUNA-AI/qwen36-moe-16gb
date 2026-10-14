# Upstream issue drafts: index (DRAFTS ONLY, nothing posted)

Owner rule: these are local drafts. Nothing has been opened, posted or submitted, and the owner decides the
repository and whether to post. The suggested target for the FreeToken items is `FlashML-org/FreeToken`, the repo
that `outputs/qwen36-api/FP8-KV-NOTES.md` links to.

All file:line references point to FreeToken **`cac247a860e316e06580d05aeb05f2e647bde214`** (0.1.3), read with
`git show cac247a8:<path>` from the local mirror `work/FreeToken-opt` (read-only).
Checkpoint: `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a052410cfd62085d94b58866fd0f2c3c5`.

| # | File | Title (short) | Type | Ready to post? |
|---|---|---|---|---|
| 1 | `01-freetoken-nvfp4-global-scale-fp16.md` | NVFP4 `weight_scale_2` stored as fp16, 94% subnormal | bug / precision | **Yes.** CPU repro re-run today; full patch apply-checked. Caveat stated: 2 of 5 smoke prompts flip the other way (near-ties). |
| 2 | `02-freetoken-qwen35-gemma-norm-bf16-fold.md` | Qwen3.5 `(1+w)` RMSNorm folded into bf16 at load | bug / precision | **Yes, with a stated gap.** Weight- and op-level evidence plus patch; whole-model PPL/eval of the fix NOT run. |
| 3 | `03-freetoken-prefix-cache-history-dependence.md` | Greedy outputs depend on cache history (dedup re-point, bf16 GDN snapshot); ask for a flush endpoint and a deterministic mode | feature / informational | **Yes, as a feature request or discussion.** No bug claimed; options 1, 2 and 4 are not implemented. |
| 4a | `04a-mtp-speculative-decoding.md` | MTP / speculative decoding prototype notes | discussion | **No.** Upstream has no MTP; component-level, synthetic-activation evidence only. |
| 4b | `04b-ram-complement-residency.md` | "RAM complement" residency, 2-3x slower | negative result | **No.** Not an upstream feature; one screening series. |
| 4c | `04c-fp8-kv-cache.md` | Optional FP8 KV cache (Q/KV dtypes tied in `fi.py`) | feature request | **Optional.** Evidence supportive but narrow (component + teacher PPL). |
| 5 | (none) | `cp.async.bulk.shared::cluster` → driver syscall on sm_120 | **skipped** | Not applicable to FreeToken (see below). |

## Item 5: why it was skipped

Upstream FreeToken at `cac247a8` contains **no** `cp.async.bulk`, no `.shared::cluster` and no TMA bulk copy in
its own CUDA sources. `git grep` over the whole tree finds none. The upstream decode gather
(`kernel/csrc/jit/fast_index_copy.cuh`) uses plain `ld.global.L1::no_allocate` / `st.global.wt`. The TMA decode
gather (`decode_gather.cuh`) and the other kernels with that pattern (`work/cuda-attention`,
`work/cuda-decode-prefetch`, `work/pcie-transfer`) are **downstream-only**.

The only TMA path upstream is FLA's Triton `solve_tril` with tensor descriptors. It is opt-in (`FLA_USE_TMA=1`,
`kernel/fla/utils.py:286-295`), off by default, and we did not test it on sm_120. The finding itself (on sm_120,
`cp.async.bulk.shared::cluster...` lowers to `__cuda_syscall_cp_async_bulk_unicast`; on first launch the driver
raises the context stack from 1024 to 14416 B/thread, about 1.37 GiB that is never returned) concerns
NVIDIA's toolchain/driver, not FreeToken. The `.shared::cta` fix is already in our build.

## Patches (local, LF, `-p1`, generated from the base blobs by `scripts/make_patches.sh`)

| Patch | Files | sha256 | Check |
|---|---|---|---|
| `patches/nvfp4-global-fp32.upstream.diff` | 14 | `b6c7f3f827763d45286b6965618717dbf82a4b0e7e219015f799ef4875102869` | `git apply --check` OK on LF-normalised base blobs, and with `--ignore-whitespace` on the raw blobs |
| `patches/gemma-norm-plus-one-fp32.upstream.diff` | 5 (incl. 1 test) | `9bcd960927d547788fad5d15229580bbef0a1cb65978c9cc52fb7f544fb845fe` | same |

- The blobs in the local mirror have CRLF line endings, which is why the second check uses
  `--ignore-whitespace`. The patches themselves are LF.
- Patch 1 is a port of the downstream fp32-global change that passed the verification listed in draft 1. Not
  ported: the downstream bit-exact CUDA kernel, `moe/cuda_nvfp4.py` (downstream-only) and the kernel-provenance
  files. Comments in the Triton kernels are updated; their code is unchanged.
- Patch 2 has the same semantics as `work/functional-fp32/norm-plus-one-fp32.diff`, rebased on the clean base. It
  drops the comment about `kernel/cuda_kv_integration.py`, a downstream-only file, and updates the loader test that
  asserted the fold.
- Neither patch was executed on an unmodified upstream checkout (NOT_RUN: pytest, GPU).

## Scripts and checks run for these drafts (CPU only, no GPU)

- `scripts/repro_global_fp16.py` and `scripts/repro_norm_fold.py` (run through `scripts/run_repros.sh` in WSL,
  read-only checkpoint access). Outputs, quoted in drafts 1 and 2:
  `30841 globals, 28964 fp16-subnormal, 1434 exact, rel err median 4.81e-4 / p99 1.46e-3 / max 1.66e-3`;
  `171008 norm elements, inexact 0.7741, rel err mean 1.38e-3 / max 3.89e-3`.
- `scripts/make_patches.sh <tmpdir>` rebuilds both patches from `git show cac247a8:<path>` and apply-checks them.

## Corrections relative to the earlier draft (`work/functional-fp32/freetoken-issue-draft.txt`)

- That draft cited "FreeToken's own MTP head (`engine/mtp.py:41`)" as applying `(1 + w)` in fp32. `engine/mtp.py`
  does **not** exist upstream; it is our prototype. Draft 2 instead cites upstream's own `GemmaPlusOneRMSNorm`
  docstring (`layers/norm.py:80-82`) and the `qwen4_exp` loader comment (`models/qwen4_exp/weight.py:63-66`).
- The norm-fold mean error is given as 1.4e-3 over all multipliers (1.28-1.43e-3 per norm type). The earlier
  "1.3e-3" was the per-type figure.

## Privacy

The drafts contain no secrets, keys, hostnames, IP addresses, user names, local paths or personal data. They also
contain no prompts or answers from the private eval sets: only counts, categories and aggregate numbers. Case
fixtures are described generically ("a 76-token coding prompt", "a tool-calling request").

## Sources used

- `outputs/qwen36-api/optimization/STATE.md` (history, decisions, eval-v1 basis decision)
- `work/functional-fp32/{freetoken-issue-draft.txt,findings.json,results.json,norm-plus-one-fp32.diff}`
- `outputs/qwen36-api/optimization/fp32-global/REPORT.md`, `work/agents/fp32-global/fp32-global.diff`
- `work/l256-deviation/{results.json,gpu-results/*.json,numerics/*}` and `work/next-update/commit-race/results.json`
  (stream-ordering verdict SAFE)
- `outputs/qwen36-api/FP8-KV-NOTES.md`, `outputs/qwen36-api/optimization/speculation-cpu-review.md`,
  `lookup-qualification.md`, `diagnose-mtp.py`
- Native reports (read-only): `mtp-diagnosis{,-v2}.json`, `mtp-kernel.json`, `spec-gpu.json`,
  `complement-{actual,kernel}.json`, `screening-0-{bf128,complement128,fixed128}.json`, `kv-kernel.json`,
  `kv-scales.json`, `functional-fp128.json`, `quality-comparison.json`, `eval-v1-basis-decision-legacyfp16.json`
