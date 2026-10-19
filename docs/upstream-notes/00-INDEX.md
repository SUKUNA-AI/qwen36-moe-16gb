# Upstream issue drafts: index (READY, NOTHING POSTED)

Owner rule: these are local drafts. Nothing has been opened, posted or submitted; the owner posts them by hand
or tells Claude to. The FreeToken items go to `FlashML-org/FreeToken`, the repo that
`outputs/qwen36-api/FP8-KV-NOTES.md` links to.

Measurements were made on FreeToken **`cac247a860e316e06580d05aeb05f2e647bde214`** (0.1.3), read with
`git show cac247a8:<path>` from the local mirror `work/FreeToken-opt` (read-only).
On 2026-10-19 every FreeToken finding was re-checked on upstream **`main` `4d2e2e2104ab104fa17aa41710cd11afbf604588`**
(68 commits after 0.1.3): all are still present, and the file:line references in drafts 1-3 now point to that commit.
Checkpoint: `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a052410cfd62085d94b58866fd0f2c3c5`.

How to post: copy the file from its `# ` title line down (the HTML comment on line 1 is the routing note, not part
of the post). The title line is the issue title.

| # | File | Where | Title (short) | Type | Status |
|---|---|---|---|---|---|
| 1 | `01-freetoken-nvfp4-global-scale-fp16.md` | FreeToken, new issue, attach `patches/nvfp4-global-fp32.main-4d2e2e2.diff` | NVFP4 `weight_scale_2` stored as fp16, 94% subnormal | bug / precision | **Ready.** Patch rebased on main and apply-checked; not built or run. Caveat stated: 2 of 5 smoke prompts flip the other way (near-ties). |
| 2 | `02-freetoken-qwen35-gemma-norm-bf16-fold.md` | FreeToken, new issue, attach `patches/gemma-norm-plus-one-fp32.upstream.diff` | Qwen3.5 `(1+w)` RMSNorm folded into bf16 at load | bug / precision | **Ready, with a stated gap.** Applies to main unchanged; whole-model PPL/eval of the fix NOT run. |
| 3 | `03-freetoken-prefix-cache-history-dependence.md` | FreeToken, new issue (feature/question) | Greedy outputs depend on cache history; ask for a flush endpoint and a deterministic mode | feature / informational | **Ready.** No bug claimed; options 1, 2 and 4 are not implemented. |
| 4c | `04c-comment-issue-280.md` | FreeToken, comment on #280 | FP8 KV on the FlashInfer path: our evidence | comment | **Ready.** Replaces a new issue: #280 and PR #354 (open, not merged) already cover the feature. `04c-fp8-kv-cache.md` stays as the long version. |
| 5 | `05-nvidia-sm120-bulk-copy-stack.md` | NVIDIA bug portal (developer.nvidia.com/bugs); optional copy on the Developer Forums | `cp.async.bulk.shared::cluster` -> driver syscall on sm_120, +1.37 GiB stack | driver/toolchain bug | **Ready.** Standalone repro in `nvidia-sm120-bulk-copy-repro/`, run on the RTX 5070 Ti on 2026-10-19 (`result-rtx5070ti.txt`). |
| 4a | `04a-mtp-speculative-decoding.md` | - | MTP / speculative decoding prototype notes | discussion | **Do not post.** Upstream has no MTP; component-level, synthetic-activation evidence only. |
| 4b | `04b-ram-complement-residency.md` | - | "RAM complement" residency, 2-3x slower | negative result | **Do not post.** Not an upstream feature; one screening series. |

Suggested order: 1 and 2 first (bugs with patches), then 5 (independent), then 3 and the #280 comment.

## Item 5: why it goes to NVIDIA, not FreeToken

Upstream FreeToken contains **no** `cp.async.bulk`, no `.shared::cluster` and no TMA bulk copy in its own CUDA
sources (`git grep` at `cac247a8`). The upstream decode gather (`kernel/csrc/jit/fast_index_copy.cuh`) uses plain
`ld.global.L1::no_allocate` / `st.global.wt`. The TMA decode gather (`decode_gather.cuh`) and the other kernels
with that pattern are **downstream-only**. The only TMA path upstream is FLA's Triton `solve_tril` with tensor
descriptors, opt-in (`FLA_USE_TMA=1`) and untested on sm_120.

The finding concerns NVIDIA's toolchain/driver. The 2026-10-19 repro (one kernel, both destination forms)
reproduces it exactly: stack limit 1024 -> 14416 B, -1374 MiB VRAM, and `cudaDeviceSetLimit` is undone by the next
launch. The `.shared::cta` fix is already in our build (`docs/ARCHITECTURE.md` 5.3).

## Patches (local, LF, `-p1`, generated from the base blobs by `scripts/make_patches.sh`)

| Patch | Files | sha256 | Check |
|---|---|---|---|
| `patches/nvfp4-global-fp32.main-4d2e2e2.diff` (attach this one) | 15 | `76cb1dd9ab883868b93202c83f80436e9a7a453228ec599cba55279ace74f8c0` | the `cac247a8` patch cherry-picked onto `main` `4d2e2e2`; the CPU-MoE conflict resolved by hand in the new `kernel/csrc/cpu_moe/kernels/nvfp4.cpp` (+ the docstring in `models/muse_glimmer/weight.py`); `git apply --check` OK on `main` |
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

## Scripts and checks run for these drafts

- 2026-10-19, GPU (RTX 5070 Ti, prod unit stopped): `nvidia-sm120-bulk-copy-repro/run.sh` with nvcc 13.4;
  output in `result-rtx5070ti.txt`. `cuobjdump` came from the Triton wheel (12.8), since the local 13.4 toolkit has none.
- 2026-10-19, CPU: both FreeToken patches apply-checked on upstream `main` `4d2e2e2`; cited lines re-mapped there.

Earlier (CPU only, no GPU):

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
