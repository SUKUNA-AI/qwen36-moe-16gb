"""Free VRAM before and after the first launch of a decode-kernel library, in a fresh process
(ARCHITECTURE.md 5.3: a .shared::cluster bulk copy or a CALL.ABS syscall would raise the context stack
on first launch and permanently take ~1.37 GiB). GPU; run under the GPU lock.

Usage: python vram_first_launch.py --oproj LIB --moe LIB --out F
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import torch

FP8 = torch.float8_e4m3fn


def free_mib():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info()[0] / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oproj", required=True)
    ap.add_argument("--moe", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.zeros(1, device=dev)  # context
    res = {"context_free_mib": free_mib()}
    # operands first, so their allocation is outside the measured window
    qa = torch.zeros((8, 4096), dtype=torch.uint8, device=dev).view(FP8)
    w = torch.zeros((2048, 4096), dtype=torch.uint8, device=dev).view(FP8)
    s = torch.ones((), device=dev)
    out = torch.empty((8, 2048), dtype=torch.bfloat16, device=dev)
    S, H, I = 4, 2048, 512
    a = torch.zeros((8, H), dtype=torch.bfloat16, device=dev)
    packed = torch.zeros((S, 2 * I, H // 2), dtype=torch.uint8, device=dev)
    scale = torch.zeros((S, 2 * I, H // 16), dtype=torch.uint8, device=dev)
    glob = torch.ones((S, 2 * I), device=dev)
    c = torch.empty((1, 8, 2 * I), dtype=torch.bfloat16, device=dev)
    tw = torch.ones((1, 8), device=dev)
    ids = torch.zeros((1, 8), dtype=torch.int32, device=dev)
    a2 = torch.zeros((8, I), dtype=torch.bfloat16, device=dev)
    packed2 = torch.zeros((S, H, I // 2), dtype=torch.uint8, device=dev)
    scale2 = torch.zeros((S, H, I // 16), dtype=torch.uint8, device=dev)
    glob2 = torch.ones((S, H), device=dev)
    c2 = torch.empty((1, 8, H), dtype=torch.bfloat16, device=dev)
    before = free_mib()
    lo = ctypes.CDLL(args.oproj)
    lo.qwen_oproj_init.restype = ctypes.c_int
    assert lo.qwen_oproj_init() == 0
    f = lo.qwen_oproj_tensorwise
    f.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_longlong] * 6 + [ctypes.c_int, ctypes.c_void_p]
    for variant in range(6):
        assert f(qa.data_ptr(), w.data_ptr(), s.data_ptr(), s.data_ptr(), out.data_ptr(), 8, 2048, 4096, 4096, 4096,
                 2048, variant, torch.cuda.current_stream().cuda_stream) == 0
    after_oproj = free_mib()
    lm = ctypes.CDLL(args.moe)
    g = lm.qwen_moe_decode_gemv
    ll, vp = ctypes.c_longlong, ctypes.c_void_p
    g.argtypes = [vp, ll, vp, ll, ll, vp, ll, ll, vp, ll, ll, vp, ll, ll, ll, vp, ll, ll, vp, ll, ll,
                  ll, ll, ll, ll, ctypes.c_int, ctypes.c_int, ctypes.c_int, vp]
    for variant in range(6):
        assert g(a.data_ptr(), H, packed.data_ptr(), packed.stride(0), packed.stride(1), scale.data_ptr(),
                 scale.stride(0), scale.stride(1), glob.data_ptr(), glob.stride(0), glob.stride(1), c.data_ptr(),
                 c.stride(0), c.stride(1), c.stride(2), tw.data_ptr(), 8, 1, ids.data_ptr(), 8, 1, 1, 8, 2 * I, H, 0, 0,
                 variant, torch.cuda.current_stream().cuda_stream) == 0
        # down GEMM (K = 512, the other kernel instantiations), router weight in the epilogue
        assert g(a2.data_ptr(), I, packed2.data_ptr(), packed2.stride(0), packed2.stride(1), scale2.data_ptr(),
                 scale2.stride(0), scale2.stride(1), glob2.data_ptr(), glob2.stride(0), glob2.stride(1), c2.data_ptr(),
                 c2.stride(0), c2.stride(1), c2.stride(2), tw.data_ptr(), 8, 1, ids.data_ptr(), 8, 1, 1, 8, H, I, 1, 1,
                 variant, torch.cuda.current_stream().cuda_stream) == 0
    after_moe = free_mib()
    res.update(before_mib=before, after_oproj_first_launch_mib=after_oproj, after_moe_first_launch_mib=after_moe,
               delta_oproj_mib=before - after_oproj, delta_moe_mib=after_oproj - after_moe)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    ok = res["delta_oproj_mib"] < 64 and res["delta_moe_mib"] < 64
    print("PASS" if ok else "FAIL: first launch took VRAM")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
