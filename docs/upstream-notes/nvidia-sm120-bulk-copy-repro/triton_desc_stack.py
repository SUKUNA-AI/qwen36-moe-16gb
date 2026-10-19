"""Does the first launch of a Triton kernel with a host TensorDescriptor raise the CUDA stack limit on sm_120?

Prints the context's CU_LIMIT_STACK_SIZE and free VRAM before and after the first and second launch,
plus whether the compiled SASS calls a driver syscall. Run with a CUDA build of torch + triton.
"""
import ctypes
import sys

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

libcuda = ctypes.CDLL("libcuda.so.1")
CU_LIMIT_STACK_SIZE = 0


def stack_limit() -> int:
    v = ctypes.c_size_t()
    assert libcuda.cuCtxGetLimit(ctypes.byref(v), CU_LIMIT_STACK_SIZE) == 0
    return v.value


@triton.jit
def copy_kernel(in_desc, out_ptr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    x = in_desc.load([pid * BM, 0])
    rows = pid * BM + tl.arange(0, BM)[:, None]
    cols = tl.arange(0, BN)[None, :]
    tl.store(out_ptr + rows * N + cols, x)


def main() -> int:
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    M, N, BM, BN = 64 * 128, 128, 128, 128
    a = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(a)
    desc = TensorDescriptor.from_tensor(a, block_shape=[BM, BN])
    torch.cuda.synchronize()
    s0, f0 = stack_limit(), torch.cuda.mem_get_info()[0]
    k = copy_kernel[(M // BM,)](desc, out, N, BM, BN)
    torch.cuda.synchronize()
    s1, f1 = stack_limit(), torch.cuda.mem_get_info()[0]
    copy_kernel[(M // BM,)](desc, out, N, BM, BN)
    torch.cuda.synchronize()
    f2 = torch.cuda.mem_get_info()[0]
    sass = k.asm.get("sass", "") if hasattr(k, "asm") else ""
    if not sass and "cubin" in getattr(k, "asm", {}):
        import os, subprocess, tempfile
        tool = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")
        with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
            f.write(k.asm["cubin"])
        sass = subprocess.run([tool, "-sass", f.name], capture_output=True, text=True).stdout
        sass += subprocess.run([tool, "-elf", f.name], capture_output=True, text=True).stdout
    mib = 1 << 20
    print(f"gpu={torch.cuda.get_device_name()} cc={torch.cuda.get_device_capability()} torch={torch.__version__} triton={triton.__version__}")
    print(f"stack_limit before={s0} after_1st={s1}")
    print(f"free_MiB before={f0 / mib:.1f} after_1st={f1 / mib:.1f} after_2nd={f2 / mib:.1f} delta={(f0 - f2) / mib:.1f}")
    print(f"sass: CALL.ABS={sass.count('CALL.ABS')} __cuda_syscall={sass.count('__cuda_syscall')} UTMALDG={sass.count('UTMALDG')} (empty if sass unavailable: {not sass})")
    print("copy", "OK" if torch.equal(a, out) else "MISMATCH")
    return 0


if __name__ == "__main__":
    sys.exit(main())
