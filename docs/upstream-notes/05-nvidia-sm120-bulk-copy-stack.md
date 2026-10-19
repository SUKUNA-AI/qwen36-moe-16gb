<!-- READY, NOT POSTED. Target: NVIDIA Developer bug portal (developer.nvidia.com/bugs, needs an NVIDIA Developer Program login; private). Optional public copy: NVIDIA Developer Forums, "CUDA Programming and Performance". Attach the folder nvidia-sm120-bulk-copy-repro/ (bulk_stack_repro.cu, run.sh, result-rtx5070ti.txt, sass-cluster-excerpt.txt). Paste from the "# " title line down; the title line is the summary. -->

# sm_120: `cp.async.bulk.shared::cluster` compiles to a driver syscall, and its first launch raises the per-thread stack limit to 14416 B (1.37 GiB VRAM on RTX 5070 Ti)

## Summary

On sm_120 (GeForce Blackwell), a kernel that uses the common
`cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes` form is compiled with a call to
`__cuda_syscall_cp_async_bulk_unicast` next to the native `UBLKCP` path. `ptxas` reports a 0-byte stack frame.
On the first launch, the driver nevertheless raises the context's `cudaLimitStackSize` from 1024 to 14416 bytes
for every resident thread. On an RTX 5070 Ti (70 SMs x 1536 threads) that is
(14416 - 1024) x 70 x 1536 = **1373 MiB of VRAM**, which stays reserved for the life of the context.

The same copy written with a `.shared::cta` destination (PTX ISA 8.6) compiles to `UBLKCP` alone, does not touch
the stack limit and costs 0 MiB. Both forms copy the same bytes. The launch in the repro uses no clusters, so every
destination is the CTA's own shared memory.

On a 16 GB card the 1.37 GiB pushed our workload into WDDM paging. It is easy to hit, because `.shared::cluster`
is the form shown in most examples and the one older PTX ISA versions require.

## Environment

- GPU: NVIDIA GeForce RTX 5070 Ti 16 GB, sm_120, 70 SMs
- Driver: Windows 617.42, used from WSL2 (kernel 6.6.114.1-microsoft-standard-WSL2); `cudaDriverGetVersion` = 13040
- Toolkit: nvcc 13.4 (V13.4.59, `cuda_13.4.r13.4/compiler.38657139_0`), runtime 13040
- Build: `nvcc -O3 -arch=sm_120`
- Tested only under WSL2. Native Linux and native Windows were not tested.

## Reproduction

Attached: `bulk_stack_repro.cu` (one kernel, about 120 lines) and `run.sh`.

```bash
CUDA=/usr/local/cuda bash run.sh ./out
```

`run.sh` builds the kernel twice: `-DDST_CLUSTER=1` (`.shared::cluster` destination) and `-DDST_CLUSTER=0`
(`.shared::cta`). It dumps SASS and ELF and runs both. Each run:

1. creates the context and reads `cudaLimitStackSize` and `cudaMemGetInfo`;
2. launches a full-occupancy grid (420 blocks x 256 threads). In each block, thread 0 issues one 4 KiB bulk copy
   global -> shared, completed on an mbarrier;
3. reads the limit and free memory again, launches a second time and reads free memory;
4. sets the limit back to the original value with `cudaDeviceSetLimit`, reads free memory, launches a third time
   and reads both again;
5. checks the copied data (checksum).

## Results

| | `.shared::cluster` | `.shared::cta` |
|---|---|---|
| ptxas | 40 registers, 0-byte stack frame | 40 registers, 0-byte stack frame |
| SASS | `CALL.ABS.NOINC` x1, `UBLKCP` x1 | `UBLKCP` x1, no `CALL.ABS` |
| ELF | `EIATTR_EXTERNS` and `EIATTR_SYSCALL_OFFSETS` name `__cuda_syscall_cp_async_bulk_unicast` | no syscall references |
| `cudaLimitStackSize` before / after 1st launch | 1024 / **14416** | 1024 / 1024 |
| free VRAM before / after 1st / after 2nd launch | 14955 / 13581 / 13581 MiB (**-1374 MiB**) | 15007 / 15007 / 15007 MiB |
| after `cudaDeviceSetLimit(1024)` | limit 1024, free 14955 MiB | unchanged |
| after the 3rd launch | limit **14416** again, free 13581 MiB | unchanged |
| checksum | OK | OK |

The measured 1374 MiB matches (14416 - 1024) x 70 x 1536 B = 1373.2 MiB.

The SASS of the `.shared::cluster` variant contains both paths: a uniform branch selects either the native
`UBLKCP.S.G` loop or `LEPC` + `CALL.ABS.NOINC` into the syscall (excerpt in `sass-cluster-excerpt.txt`). We did
not check at run time which branch executes. In this launch every destination is CTA-local, so we would expect
the native branch. The stack reservation seems to come from the syscall merely being reachable.

Lowering the limit by hand gives the memory back, but the next launch raises it again. So `cudaDeviceSetLimit` is
not a workaround.

## Expected behaviour

One of:
- for sm_120, `ptxas` lowers `cp.async.bulk.shared::cluster` with a CTA-local destination (or any destination
  when the launch has no cluster) to `UBLKCP` alone, without the syscall fallback; or
- the driver does not raise the per-thread stack limit for a syscall that needs no stack, or the size of the raise
  is documented; or
- the PTX ISA or Blackwell tuning guide documents that, on sm_120, `.shared::cluster` bulk copies cost about
  14 KB of stack per resident thread and that `.shared::cta` (PTX 8.6+) should be used instead.

## Questions

1. Is the 14416 B/thread stack raise intended for `__cuda_syscall_cp_async_bulk_unicast` on sm_120?
2. Does the same happen for the `cp.async.bulk.tensor` (TMA) forms with a `.shared::cluster` destination, and on
   sm_100 / sm_121? We did not test those.
3. Is `.shared::cta` the recommended destination for single-CTA bulk copies on sm_120, so that libraries such as
   CCCL and CUTLASS should prefer it there?

## Workaround we use

We use the `.shared::cta` destination for every bulk copy in sm_120 kernels. Our build scripts fail if the SASS
contains `CALL.ABS` or `__cuda_syscall`, or if the PTX contains `.shared::cluster`. In our production kernel this
also lowered register use from 46 to 41 at unchanged speed, and the copied bytes were identical.
