<!-- READY, NOT POSTED. Target: comment on FlashML-org/FreeToken#743 (TMA prefill first launch allocates ~3.7 GiB on RTX 5090). Paste everything below this line. Evidence: nvidia-sm120-bulk-copy-repro/result-triton-desc-rtx5070ti.txt and result-rtx5070ti.txt. -->

We hit what looks like the same mechanism on an RTX 5070 Ti (sm_120) and can say where the memory goes: it is
the **per-thread CUDA stack limit** that the driver raises on the first launch, not an allocation by Triton.

On sm_120 the TMA load in a Triton kernel with a host `TensorDescriptor` compiles to a call into a driver syscall.
The kernel's SASS has `UTMALDG.2D` plus a `CALL.ABS.NOINC`, and the ELF lists
`__cuda_syscall_cp_async_bulk_tensor_2d_tile_unicast`. `ptxas` reports no stack frame, but the first launch raises
the context's `CU_LIMIT_STACK_SIZE` from 1024 to about 14.4 KB **for every resident thread** on the GPU, and that
memory stays reserved for the life of the context.

Minimal check: a Triton 3.6.0 kernel that loads one 128x128 bf16 tile through a `TensorDescriptor` and stores it
back (torch 2.11.0+cu130, driver 617.42 under WSL2):

```
stack_limit before=1024 after_1st=14432
free_MiB before=14933.0 after_1st=13557.0 after_2nd=13557.0 delta=1376.0
```

That matches (14432 - 1024) B x 70 SMs x 1536 threads = 1375 MiB. On a 5090 (170 SMs) the same formula gives
about 3.3 GiB, which is in the range you measured. To confirm it on your machine, read
`cuCtxGetLimit(CU_LIMIT_STACK_SIZE)` (or `cudaDeviceGetLimit(cudaLimitStackSize)`) before and after the first
TMA launch. A ctypes snippet is enough:

```python
import ctypes; cu = ctypes.CDLL("libcuda.so.1"); v = ctypes.c_size_t()
cu.cuCtxGetLimit(ctypes.byref(v), 0); print(v.value)   # 0 = CU_LIMIT_STACK_SIZE
```

Some notes that may help with the fix:
- The cost scales with SM count x max threads per SM, not with the kernel or the problem size. That is why even a
  256-token prefill OOMs, and why it happens once per process.
- In a CUDA C++ repro with a plain (non-tensor) bulk copy, `cp.async.bulk.shared::cluster...` showed the same raise
  (1024 -> 14416 B, -1374 MiB), while the `.shared::cta` destination form (PTX 8.6+) compiled to native `UBLKCP`
  only and cost 0 MiB. There, lowering the limit again with `cudaDeviceSetLimit` returned the memory, but the next
  launch raised it again, so that is not a workaround. We did not check whether Triton can be made to emit a
  form without the syscall.
- So a free-memory gate or an opt-out for the TMA path on sm_120, as you suggest, looks like the practical fix.
  The memory budget could also reserve `(14432 - 1024) x SMs x threads/SM` before sizing the pools when TMA is on.

We are reporting the driver/toolchain side to NVIDIA separately, with a standalone repro.
