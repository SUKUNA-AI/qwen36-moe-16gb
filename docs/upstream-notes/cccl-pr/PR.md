<!-- POSTED 2026-10-19 as https://github.com/NVIDIA/cccl/pull/12081. Target: NVIDIA/cccl pull request from a fork. -->

# Title

Document the SM_120 cost of `cp.async.bulk` with a `.shared::cluster` destination

# Body

## Description

Related: #6302 (closed by #6362)

This adds one implementation note to the `cuda::ptx::cp_async_bulk` docs page
(`docs/libcudacxx/ptx/instructions/cp_async_bulk.rst`). It recommends the `.shared::cta`
destination (`cuda::ptx::space_shared`, PTX ISA 86) over `.shared::cluster`
(`cuda::ptx::space_cluster`) for global-to-shared copies into the CTA's own shared memory, and
says why that matters on SM_120.

#6302 found that on SM_120 the `.shared::cluster` form is compiled with a driver fallback path, and
#6362 switched `cuda::memcpy_async`, `cuda::device::memcpy_async_tx` and the CUB bulk-copy paths to
`.shared::cta` when PTX ISA 86 is available. The `cuda::ptx` docs page does not mention this yet, so
people who call `cuda::ptx::cp_async_bulk` directly (or copy the first overload on the page, which
is the `.shared::cluster` one) still hit it. We also measured a memory cost that #6302 did not
mention.

What we observed on SM_120 (RTX 5070 Ti, 70 SMs, 16 GB), driver 617.42 used under WSL2,
CUDA 13.4, `nvcc -O3 -arch=sm_120`, one kernel in which thread 0 of each block issues a 4 KiB
global-to-shared bulk copy completed on an mbarrier (no clusters in the launch):

| | `.shared::cluster` | `.shared::cta` |
|---|---|---|
| ptxas | 40 registers, 0-byte stack frame | 40 registers, 0-byte stack frame |
| SASS | `UBLKCP` plus `CALL.ABS.NOINC` to `__cuda_syscall_cp_async_bulk_unicast` | `UBLKCP` only |
| `cudaLimitStackSize` before / after 1st launch | 1024 / 14416 B | 1024 / 1024 B |
| free device memory change after 1st launch | -1374 MiB | 0 MiB |
| after `cudaDeviceSetLimit(cudaLimitStackSize, 1024)` and another launch | 14416 B again | unchanged |
| copied data | correct | correct |

The measured 1374 MiB matches (14416 - 1024) B x 70 SMs x 1536 threads = 1373 MiB. The reservation
stays for the lifetime of the context, and lowering the limit by hand is undone by the next launch.
The ELF of the `.shared::cluster` variant lists `__cuda_syscall_cp_async_bulk_unicast` in
`EIATTR_EXTERNS` and `EIATTR_SYSCALL_OFFSETS`; the `.shared::cta` variant has no syscall
references.

Not tested: SM_100 / SM_121, native Linux, native Windows. The note is worded as an observation for
that driver and toolkit. A standalone repro (one `.cu` file and a build/run script that prints the
numbers above) is available on request.

The tensor (TMA) form looks affected too, but we have not narrowed it down enough to document it,
so this PR does not touch `cp_async_bulk_tensor.rst` (which currently says nothing about this). On
the same GPU and driver, a Triton 3.6.0 kernel that loads one 128x128 bf16 tile through a host
tensor descriptor compiles to `UTMALDG.2D` plus `CALL.ABS.NOINC`, its ELF lists
`__cuda_syscall_cp_async_bulk_tensor_2d_tile_unicast`, and its first launch raised the stack limit
from 1024 to 14432 B (-1376 MiB free device memory). We have not checked which destination form
Triton emitted, and we have not measured whether the `.shared::cta` tensor overloads
(PTX ISA 86) avoid the fallback. A similar loss (about 3.7 GiB on an RTX 5090 on the first Triton
TMA launch) has also been reported publicly elsewhere. If maintainers know whether `.shared::cta`
avoids this for `cp.async.bulk.tensor` on SM_120, we can extend the note to that page.

Side note, not changed here: the deprecated helpers in `libcudacxx/include/cuda/barrier`
(`cuda::device::experimental::cp_async_bulk_global_to_shared` and the
`cp_async_bulk_tensor_{1..5}d_global_to_shared` family) still pass `cuda::ptx::space_cluster`
unconditionally. Happy to send a follow-up that applies the same `__cccl_ptx_isa >= 860` selection
there if that is wanted.

## Checklist
- [ ] New or existing tests cover these changes. (Docs-only change, no code touched.)
- [x] The documentation is up to date with these changes.

# How to submit (owner checklist)

1. Fork `NVIDIA/cccl` on GitHub to your account.
2. Clone the fork (a partial clone is enough):
   `git clone --filter=blob:none https://github.com/<you>/cccl.git && cd cccl`
   On Windows, run `git config core.autocrlf false` in the clone before checking out other branches.
3. Create a branch from current `main`: `git switch -c docs/cp-async-bulk-sm120-shared-cta`
4. Apply the patch: `git am 0001-docs-cp-async-bulk-sm120.patch`
   If your GitHub noreply address has a numeric prefix (`<id>+<user>@users.noreply.github.com`),
   fix the author: `git commit --amend --reset-author --no-edit`.
5. Sign-off: CCCL does not require DCO `Signed-off-by` lines or a CLA for external contributors
   (only NVIDIA enterprise members are asked to sign commits). Adding `-s` is optional.
6. Optional: `pip install pre-commit && pre-commit run --files docs/libcudacxx/ptx/instructions/cp_async_bulk.rst`
   (CI runs pre-commit.ci anyway).
7. Push: `git push -u origin docs/cp-async-bulk-sm120-shared-cta`
8. Open the PR against `NVIDIA/cccl:main` with the title and body above (the body follows the repo's
   PR template: Description + Checklist). The title goes into the CHANGELOG.
9. CI for external contributors starts only after a maintainer comments `/ok to test`; a docs
   preview link is posted automatically. Answer review comments; attach the repro if asked.
