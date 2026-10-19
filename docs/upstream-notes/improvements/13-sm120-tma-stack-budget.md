<!-- READY, NOT POSTED. Target: FlashML-org/FreeToken, new issue (bug / feature proposal), references #743. Paste from the "# " title line down. Checked against upstream main 4d2e2e2 (2026-10-19). -->

# [Bug] sm_120: the first TMA kernel launch reserves ~14 KB of stack per resident thread (1.4 GiB on 16 GB cards), and the VRAM budget does not account for it

Related: #743, which reports the OOM on an RTX 5090. This issue covers the root cause on all sm_120 cards and
proposes a fix to the memory budget, so the TMA path can stay on.

## Summary

On sm_120, Triton's TMA load (`TensorDescriptor`) compiles to a call into a driver syscall. On the first launch of
such a kernel, the driver raises the context's per-thread stack limit from 1024 B to about 14.4 KB **for every
thread the GPU can hold**. That memory stays reserved until the process exits. It costs about 1.34 GiB on a 70-SM
RTX 5070 Ti and about 3.3 GiB on a 170-SM RTX 5090 (computed with the same formula). Since `4d2e2e2`,
`kernel/nvfp4_moe.py` uses TMA on sm_120 for the NVFP4 MoE prefill GEMM. The reservation therefore happens on the
first prefill, after the expert cache and the KV pool have already been sized from free VRAM, and the server OOMs.

## Setup

- RTX 5070 Ti 16 GB (sm_120, 70 SMs, 1536 threads/SM), driver 617.42 under WSL2, CUDA 13.4
- torch 2.11.0+cu130, Triton 3.6.0
- Reproduced with a standalone 20-line Triton kernel, without any FreeToken code

## Evidence

A Triton kernel loads one 128x128 bf16 tile through a host `TensorDescriptor` and stores it back:

```
stack_limit before=1024 after_1st=14432
free_MiB before=14933.0 after_1st=13557.0 after_2nd=13557.0 delta=1376.0
sass: CALL.ABS=1 UTMALDG=1, ELF extern __cuda_syscall_cp_async_bulk_tensor_2d_tile_unicast
```

- (14432 - 1024) B x 70 SMs x 1536 threads = 1375 MiB, which matches the measured 1376 MiB. For 170 SMs the
  same formula gives 3.26 GiB.
- `ptxas` reports a 0-byte stack frame. The reservation comes from the syscall in the SASS.
- In a CUDA C++ repro with a plain bulk copy (`cp.async.bulk.shared::cluster`), lowering the limit again with
  `cudaDeviceSetLimit` gave the memory back, but the next launch raised it again. A `.shared::cta` destination
  avoided the syscall entirely, but Triton chooses the PTX form itself.
- We are reporting the driver/toolchain side to NVIDIA separately, with a standalone repro.

## Where (upstream main `4d2e2e2`)

- `python/freetoken/kernel/nvfp4_moe.py:281-291`: `_tma_device()` is true on capability (12, 0), and
  `_prefill_use_tma()` enables TMA whenever K % 256 == 0 and the pointers are aligned. There is no memory check
  and no opt-out (`FREETOKEN_DISABLE_GLUON` only covers the Gluon kernels).
- `python/freetoken/kernel/nvfp4_moe.py:356-370`: the descriptors are built at the first prefill.
- `python/freetoken/engine/engine.py:98-102` (`_startup_kv_budget`) and `:410-416` (`_post_weights_free`): the pool
  budgets are taken from free VRAM measured before any TMA kernel has run.
- The Gluon sm_120 kernels (`kernel/gluon/nvfp4/moe/sm120*.py`) also use TMA descriptors. We expect the same
  reservation there but did not test Gluon.

## Proposal

1. **Pay the reservation up front.** On sm_120, when any TMA path is enabled, launch one tiny TMA kernel right
   after context creation and before the baseline free-memory reading that sizes the expert cache and the KV pool.
   The reservation then shows up as already-used memory, and every budget below it stays correct. In our repro,
   later launches did not raise the limit further. Kernels with a different syscall may need a few bytes more:
   we saw 14416 B for the plain bulk copy and 14432 B for the 2D tile load. The warm-up should therefore use the
   largest form the engine uses, or the budget should keep a small margin.
2. **Opt-out switch**, e.g. `FREETOKEN_DISABLE_TMA=1`, so users with tight VRAM can trade prefill speed for about
   1.3-3.3 GiB of cache.
3. **Report it.** Log the stack-limit raise and its size (`cudaDeviceGetLimit(cudaLimitStackSize)` before and
   after the warm-up) at startup, so OOM reports show it directly.

We have not implemented 1-3 in FreeToken. We can send a PR for option 1 plus the switch if that direction suits you.

## Not measured

- Gluon (Triton 3.8) kernels, sm_100 / sm_121, native Linux or Windows (WSL2 only).
- Whether the reservation depends on the driver version.
