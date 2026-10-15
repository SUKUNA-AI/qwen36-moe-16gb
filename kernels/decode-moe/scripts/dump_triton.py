"""Dump the compiled Triton artefacts (TTGIR, LLIR, PTX, SASS-ready cubin) of the production NVFP4
decode GEMV (_decode_nvfp4_marlin_kernel) for the two Qwen3.6 decode GEMMs, exactly as
moe/fused_nvfp4._decode_gemm_marlin launches them. GPU (compiles + one launch each).

Usage: python dump_triton.py --outdir DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    import freetoken.moe.fused_nvfp4 as fn
    from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view
    from freetoken.kernel.triton.nvfp4_fused_moe import _decode_nvfp4_marlin_kernel, _e2m1_lut

    dev = torch.device("cuda", 0)
    S, H, I, TOP_K = 16, 2048, 512, 8
    info = {}
    for name, (N, K, a_rows, mul_w, a_is_route) in {
        "gate_up": (2 * I, H, 1, False, False),
        "down": (H, I, TOP_K, True, True),
    }.items():
        packed = torch.randint(0, 256, (S, N, K // 2), dtype=torch.uint8, device=dev)
        scale = torch.randint(0, 0x7F, (S, N, K // 16), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
        glob = torch.rand((S, N), device=dev)
        a = torch.randn((a_rows, K), device=dev).to(torch.bfloat16)
        topk_ids = torch.randperm(S, device=dev)[:TOP_K].reshape(1, TOP_K).to(torch.int32)
        topk_w = torch.rand((1, TOP_K), device=dev)
        c = torch.empty((1, TOP_K, N), dtype=torch.bfloat16, device=dev)
        packed_i32 = packed.view(torch.int32)
        sc = e4m3_kernel_view(scale)
        deep_k = K > fn._DECODE_MARLIN_DEEPK_THRESHOLD
        block_n = fn._DECODE_MARLIN_DEEPK_BLOCK_N if deep_k else fn._DECODE_MARLIN_BLOCK_N
        block_kw = fn._DECODE_MARLIN_DEEPK_BLOCK_KW if deep_k else fn._DECODE_MARLIN_BLOCK_KW
        grid = (TOP_K, (N + block_n - 1) // block_n)
        k = _decode_nvfp4_marlin_kernel[grid](
            a, packed_i32, sc, glob, c, topk_w, topk_ids, _e2m1_lut(0), TOP_K, N, K,
            a.stride(0), a.stride(1),
            packed_i32.stride(0), packed_i32.stride(1), packed_i32.stride(2),
            sc.stride(0), sc.stride(1), sc.stride(2),
            glob.stride(0), glob.stride(1),
            c.stride(0), c.stride(1), c.stride(2),
            topk_w.stride(0), topk_w.stride(1),
            topk_ids.stride(0), topk_ids.stride(1),
            BLOCK_SIZE_N=block_n, BLOCK_SIZE_KW=block_kw, TOP_K=TOP_K,
            A_ROW_IS_ROUTE=a_is_route, MUL_ROUTED_WEIGHT=mul_w,
            compute_type=fn._tl_dtype(c.dtype), num_warps=fn._DECODE_MARLIN_WARPS,
        )
        torch.cuda.synchronize()
        for ext in ("ttir", "ttgir", "llir", "ptx"):
            if ext in k.asm:
                (out / f"{name}.{ext}").write_text(k.asm[ext])
        if "cubin" in k.asm:
            (out / f"{name}.cubin").write_bytes(k.asm["cubin"])
        info[name] = {"N": N, "K": K, "block_n": block_n, "block_kw": block_kw, "grid": list(grid),
                      "num_warps": fn._DECODE_MARLIN_WARPS, "mul_w": mul_w, "a_is_route": a_is_route,
                      "scale_dtype": str(sc.dtype), "metadata": str(getattr(k, "metadata", ""))[:2000]}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
