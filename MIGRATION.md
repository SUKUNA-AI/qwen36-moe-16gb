# T6: torch 2.14 / Triton 3.8 migration study

**Decision: feasible candidate stack, not qualified; no upgrade performed.**
Keep the production stack and all f1 references/receipts intact. Target-device
execution, wheel loading, recompilation, bit-exactness and speed are NOT_RUN.
The requested versions exist; the risk is not merely dependency resolution:
Triton's compiler/toolchain and FreeToken's slot ownership changed.

## Exact sources and target

Checked public upstream on 2026-10-15, separately from the working repository:

- Fork basis: [`cac247a860e316e06580d05aeb05f2e647bde214`](https://github.com/FlashML-org/FreeToken/tree/cac247a860e316e06580d05aeb05f2e647bde214), release 0.1.3.
- Study target: [`b416f85c40970c033462b47773020872209978a3`](https://github.com/FlashML-org/FreeToken/tree/b416f85c40970c033462b47773020872209978a3), 2026-10-14, slot-pool change #689.
- Dependency change: [`74a387777708b9650a027494bcb37d2ab9cf2443`](https://github.com/FlashML-org/FreeToken/commit/74a387777708b9650a027494bcb37d2ab9cf2443), #630.
- Triton v3.6.0 `7c56a5e40f7fd928dfd5c72902d5def0097db73a` versus v3.8.0
  `c01b6774b1865984607d89d89d3a10833de92037`.

| Package | Released stack | Study target (Linux x86-64, Python 3.12) |
|---|---|---|
| torch | 2.11.0 | **2.14.1**, PyPI cu130 build |
| torchvision | 0.26.0 | **0.29.1** |
| Triton | 3.6.0 | **3.8.0** |
| sglang-kernel | 0.4.5 | **0.4.9**, requires torch==2.14.1 |
| FlashInfer Python | 0.6.18.post1 | **0.6.18.post1**, explicitly pinned by current upstream |
| flashlib | 0.3.0 | **0.4.0**, slot-cache dependency changed upstream |
| apache-tvm-ffi | 0.1.13.post3 | keep **0.1.13.post3** |
| transformers | 5.16.1 | keep **5.16.1** within upstream's 5.16 range |
| FreeToken | fork of 0.1.3 | pin the **source SHA** above, not floating main or the still-0.1.3 PyPI wheel |

PyPI metadata, exact CPython 3.12 wheel filenames, URLs, SHA256 and dependency
lists are saved in `validation/migration/target-packages.json`. This is a target
set, **not a complete transitive lock**. Resolve and hash a separate owner test
environment before running it; do not feed this table into production pip.
Torch requires CUDA toolkit packages 13.0.3, cuDNN 9.24.0.43, cuSPARSELt 0.8.1,
NCCL 2.30.7 and NVSHMEM 3.4.5. Its Triton constraint is ~=3.8.0. Torch/torchvision
wheels require manylinux_2_28; check glibc >=2.28 in the owner's WSL distro.
Neither the large wheels nor model weights were downloaded in this study.

## Driver 617.14, CUDA and binary compatibility

[NVIDIA's minor-version compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
says CUDA 13.x requires driver family >=580; 617.14 is above this minimum.
The cu130 wheel is therefore compatible at the documented driver-family level;
no driver replacement is proposed. This does **not** prove the owner's WSL CUDA
interop, SM120 kernel images, PTX JIT or package ABI works. Save driver/runtime
versions and run a small device allocation/matmul/capture smoke test first.

Triton 3.8 ships a different assembler for Blackwell: **ptxas-blackwell
13.3.33**, versus **12.9.86** in 3.6. Its general ptxas is 12.9.86, versus 12.8.93.
This is independent of Torch's CUDA 13.0.3 runtime and the owner's CUDA 13.4 build
toolkit. `triton-pins.json` records the source-pinned toolchain. A minimum driver
family alone is not proof that every emitted PTX feature loads; inspect the actual
assembler selected for SM120 and perform module load/launch on the owner device.

Rebuild all Torch-linked extensions and the FreeToken kernel cache in the isolated
new environment. Do not reuse `_pinned_tensor`, `_cpu_moe` or cached .so files
built against Torch 2.11 simply because filenames still match. Hash binaries,
recipes, source trees, wheel RECORDs and ABI probes. Keep the strict loader and
wrapper checks enabled. Custom CUDA builds retain CUDA 13.x / sm_120a, precise
arithmetic flags, and the ban on bulk-copy destination `.shared::cluster`.
Inspect SASS for `CALL.ABS` / `__cuda_syscall`; a successful build is not runtime
validation. This cloud has neither nvcc nor cuobjdump.

## Source differences affecting byte replicas

The source comparison is reproducible, including normalized source hashes and an
AST comparison of the NVFP4 prefill function. See
`kernel-source-comparison.json`, `freetoken-reviewed.diff`, `triton-compiler.diff`
and `flashinfer-plan.json` under `validation/migration/`.

| Component | Actual base -> target difference | Consequence for this fork |
|---|---|---|
| NVFP4 **prefill** | `_prefill_nvfp4_moe_kernel` AST is identical; low/high nibble `tl.dot` accumulation order is unchanged. File changed because decode was rewritten. | Do not rewrite the CUDA prefill replica based on the file diff alone. Re-run raw-bit tests against code compiled by Triton 3.8. |
| NVFP4 **decode** | `_decode_nvfp4_marlin_kernel` changes partial layout [KW,N] -> [N,KW], reduction axis, bit-trick dequant, scale *16384, wide activation loads and unrolling. Launcher changes deep-K threshold `K>2048` -> `K>=2048`, KW128 ->32; Qwen H=2048 is affected. | It is not safe to describe a full upstream rebase as transport-only. Separate decode numerical comparison and teacher gate are required even if prefill passes. |
| FLA GDN | All 17 compared `kernel/fla/*.py` files and Qwen gdn.py are byte-identical after LF normalization. Existing fork-specific snapshot/FLA hooks still need porting. | Source arithmetic order unchanged upstream, but `tl.dot`, reductions, exp/SiLU and state accumulation may compile differently. Test full FP32 state and signed-zero/raw bits, both wu/ho and off. |
| FlashInfer attention | `attention/fi.py` is unchanged upstream; dependency is still 0.6.18.post1. Official header's PrefillPlanInfo has 15 fields; cta_tile_q index3 and split_kv index14 match the fork's reader. | Keep FA2, page size and plan guards. Compare individual/grouped plans for every chunk/window; unchanged version is not proof of identical generated binaries or launch choices under a new environment. Unknown plan shape must fall back. |
| FP8 projections | Upstream `fp8_pertensor_linear.py` is unchanged; the fork's CUDA GEMV is compared against this file. | Torch/cuBLASLt and quantization/reduction selection changed with the stack. Re-run M1..8, normal/heavy/edge, eager and graph tests; keep overflow/fallback coverage. |
| Gather / cache | CUDA branches of basic fast_index_copy keep the same loads/stores; ROCm branches were added. Call sites now use pool-local slot ids, `pool_bank_views(layer_id)`, per-pool copy descriptors and staging views. Hybrid routing also changed. | T1 borrowed tails, tuned gather, T3 metadata/slot ownership and rebuild must be ported together to pool semantics. A textually clean patch can still index the wrong pool. |

Triton source diff is substantive: LLVM PTX feature cap 8.6 ->9.0, changed pass
ordering, prefetch restricted to SM8x, TMEM reduction/barrier passes, dot lowering
and **679 lines of reduction-lowering diff**. The release has an SM90-specific
workaround for LLVM SLP reordering bf16 reductions; do not infer that this fixes
or breaks SM120. It demonstrates why identical Python/Triton source is insufficient
for bit-exact certification. Save generated TTIR/TTGIR/PTX/SASS for the exact
shapes that fail; investigate a component failure before any model benchmark.

## Rebase conflicts and order

No project rebase was performed. A read-only `git merge-file -p` three-way probe
against the two pristine source snapshots found **11 files with textual conflicts**
among 90 changed/added source/build/test files in the current fork. Full per-file
hashes/status are in `fork-rebase.json` (LF normalization only for comparison):

- `core.py`, `engine/engine.py`, `engine/sample.py`;
- `kernel/csrc/cpu_moe/cpu_moe_ext.cpp`, `kernel/utils.py`;
- `message/tokenizer.py`, `moe/cpu_executor.py`, `moe/offload_cache.py`;
- `scheduler/scheduler.py`, `server/generation.py`, `server/openai_api.py`.

`layers/moe.py` merges textually, but pool-bank views make its old gather assumptions
semantically conflicting. Newly added helpers have no common ancestor; a clean
add does not prove integration. The upstream CPU extension was split into common,
format, synchronization and per-format kernel files, so port its hooks to the new
owners instead of restoring the monolithic file.

Recommended port sequence, in a separate future migration branch/environment:

1. Freeze target source and complete dependency/wheel lock; build upstream alone.
   Preserve a clean source/stack baseline before introducing fork optimizations.
2. Port strict provenance, recipe manifests, source receipt/ABI inventory and CPU
   test/import infrastructure. Rebuild every native extension/cache entry.
3. Port telemetry, teacher case/run/reference identity and guard/lifecycle fixes.
   Close REVIEW H1 (functional binding) and replace the frozen final driver first.
4. Port core/scheduler snapshots and layer-major windows **with pool-aware offload
   storage**. Check borrowed region ownership, prefix/cancel, rebuild and slot ids.
   Keep all optional flags off until each transport component passes.
5. Port CUDA attention/GDN/FP8/NVFP4 dispatch one component at a time; keep original
   arithmetic and fallback shapes. If the new oracle differs, record the cause;
   do not silently relax a comparison or modify the old reference.
6. Port T1, then T2, then T3 separately; repeat each gate before combining them.
   T3 must use the new per-pool descriptors and producer fence, not global-slot
   indexing from this branch. Re-capture after any address/layout change.

Reproduce the study using separate read-only checkouts (no model/environment install):

```bash
python tools/migration-source-study.py --fork "$C" \
  --base /absolute/pristine-cac247a --target /absolute/pristine-b416f85 \
  --output /absolute/new-migration-evidence
```

## Owner validation: separate references, components, quality and speed

Use the prerequisites, exclusive lock, experiment-only lifecycle and unique names
in GPU_VALIDATION.md. Keep old and new venvs, source overlays, native libraries,
JIT/graph caches and raw report directories separate. The current site launchers
have fixed owner Python paths: explicitly review their experiment-only copies to
select each venv; setting `$P_NEW` in a shell does not change a systemd service's
interpreter. Do not operate production or use the frozen final-gates.sh.

After dependency/import/ABI/device smoke checks, run the existing component CLIs
with `$P_NEW` and the **ported fork** on FREETOKEN_PYTHON/PYTHONPATH. The flags below
were checked in source; execution remains NOT_RUN. `$NVFP4`, `$GDN`, `$FP8` are
separately recorded libraries; `$UPSTREAM_NEW/python` is the pristine target tree:

```bash
export QWEN36_CUDA_NVFP4_LIBRARY="$NVFP4" QWEN36_CUDA_GDN_LIBRARY="$GDN"
export QWEN36_FP8_GEMV_LIBRARY="$FP8" FREETOKEN_PRISTINE_PYTHON="$UPSTREAM_NEW/python"
flock -w 3600 -E 75 "$LOCK" timeout 1800 "$P_NEW" \
  "$C/kernels/nvfp4-moe/py/test_bitexact.py" --suite all --output "$RAW/nvfp4-new.json"
# test_real acquires QWEN36_GPU_LOCK itself; do not nest another flock.
QWEN36_GPU_LOCK="$LOCK" "$P_NEW" "$C/kernels/nvfp4-moe/py/test_real.py" \
  --layers 0,13,26,39 --m 1,7,17,64,65,256,2048,4096 --output "$RAW/nvfp4-real-new.json"
flock -w 3600 -E 75 "$LOCK" timeout 1800 "$P_NEW" \
  "$C/kernels/gdn/test_gdn_bitexact.py" --lib "$GDN" --out "$RAW/gdn-new.json"
flock -w 3600 -E 75 "$LOCK" timeout 1800 "$P_NEW" \
  "$C/kernels/gdn/fla/test_fla_fused.py" --lib "$GDN" --out "$RAW/fla-new.json"
flock -w 3600 -E 75 "$LOCK" timeout 1800 "$P_NEW" \
  "$C/kernels/fp8-gemv/scripts/step3_bitwise.py" --output "$RAW/fp8-new.json"
```

Then repeat GDN `--fla --fla-variant wu` and `ho`, and the gather/short-route/T3
capture-replay tests from GPU_VALIDATION.md. A rebuilt .so with another hash is
not automatically permitted by the pinned production loader. First use reviewed
component paths; change experiment pins only with a separate recorded binary
provenance step. Any unexpected fallback or zero fused calls invalidates that arm.

Create a **new** teacher-bf128 reference only in the isolated new-stack data root,
with the owner's existing private fixture/tokenizer. Set
`QWEN36_TEACHER_CREATE_REFERENCE=1` explicitly in that experiment config and
`QWEN36_LOGIT_REFERENCE` to a fresh empty directory. The engine exclusively creates
missing reference files; old f1 arrays must not be in that directory. After creation,
remove the create flag, rerun verification, and pin the new schema-2 report and its
clean-cache long-120000 case->run mapping. Each new-stack candidate must use that
report via QWEN36_TEACHER_REFERENCE_REPORT. Preserve old and new bases separately.

Run qualify with explicit new baseline/functional report names as documented in
GPU_VALIDATION.md, with H1 fixed before any acceptance decision. A comparison
against a *common* old reference can already record nonzero max_abs and a separate
quality_gate_pass while returning exit1 for bit-exact failure. The current comparator
cannot compare two independently created reference identities or omit logits:
a separately reviewed PPL-only/cross-reference mode is a prerequisite for that
specific migration analysis. Do not treat exit1/non-bit-exact as invalid evidence,
and do not bypass the identity check to join unrelated references.

For **stack-only** speed use source held constant: A = pristine cac247a on old
stack; B = pristine cac247a on the new stack (isolated build metadata adaptation
required because old pyproject pins torch<2.12). New upstream b416f85 on new stack
is a third arm, C, measuring source changes too. Comparing A directly with C cannot
attribute a gain to Torch/Triton. D = ported fork with individual optional features.
No replica flag is enabled in original-upstream A/B; keep global-scale semantics,
model revision, cache/context/chunk sizes and sampling identical between arms.

Use the project's five-pair rotating A/B protocol, warmup >=60s, no profiler,
fixed output budget, raw paired observations and confidence/noise reporting.
Existing matrix.py assumes one site-configured launcher/Python; multi-venv launch
selection is an explicit owner setup prerequisite, not an invented CLI flag.
Each run receipt must prove the actual imported source and interpreter/packages.
Measure TTFT and decode separately at short/2048/2049/16K/120K; for long256 include
250K, prefix hits/edits, repeated requests and cancellation. A 120K teacher PASS
does not certify 250K; a new 250K fixture requires an explicitly reviewed basis.

## 256K implication

T3's historical +3-4% estimate concerns decode overlap, not 250K prefill. The
checked-in long256 analysis identifies attention and hidden-store window capacity
as the larger TTFT targets. Current data do not show that Torch 2.14/Triton 3.8
improves them. Benchmark the stack-only arm first, inspect grouped FA2 plan
invariants and the one-window VRAM feasibility separately, and change no released
long256 value until teacher/component/paired measurements support it.
