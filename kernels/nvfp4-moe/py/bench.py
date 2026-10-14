"""Paired speed benchmark: Triton fused_experts_nvfp4 vs bit-exact CUDA (no profiler).

Synthetic expert banks of the real Qwen3.6 geometry (E=256, H=2048, I=512, top-8),
resident on the GPU (no offload/PCIe), so this is a KERNEL/LAYER microbenchmark and
not a model TTFT measurement.  Each pair runs both arms back to back with alternating
order; CUDA-event timing over `iters` launches per arm; eager and CUDA-graph modes.
Run only through gpu-run.sh (shared GPU lock).
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

from freetoken.moe import fused_nvfp4 as ref
from freetoken.moe.fused import moe_align_block_size

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_bitexact as tb  # noqa: E402
from bitexact_lib import BitExactPrefill  # noqa: E402

H, I, E, TOPK = 2048, 512, 256, 8


def time_fn(fn, iters):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / iters


def graphed(fn, warm=2):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def paired(arms, pairs, iters, mode):
    names = list(arms)
    fns = {}
    for n in names:
        fns[n] = graphed(arms[n]) if mode == "graph" else arms[n]
        for _ in range(3):
            fns[n]()
    torch.cuda.synchronize()
    samples = {n: [] for n in names}
    ratios = []
    for p in range(pairs):
        order = names if p % 2 == 0 else names[::-1]
        got = {}
        for n in order:
            got[n] = time_fn(fns[n], iters)
        for n in names:
            samples[n].append(got[n])
        ratios.append(got[names[0]] / got[names[1]])
    return {"mode": mode, "pairs": pairs, "iters_per_arm": iters,
            "median_ms": {n: statistics.median(samples[n]) for n in names},
            "samples_ms": samples,
            f"median_paired_ratio_{names[0]}_over_{names[1]}": statistics.median(ratios),
            "paired_ratios": ratios}


def breakdown(lib, x, banks, w, ids, iters, variant=None):
    """Per-stage CUDA-event times (eager), CUDA path and Triton path."""
    gu_p, gu_s, gu_g, d_p, d_s, d_g = banks
    m = x.shape[0]
    tw = w.reshape(-1).contiguous()
    out = {}
    s64, e64, n64 = moe_align_block_size(ids, 64, E)
    inter = torch.empty((m * TOPK, I), device=x.device, dtype=x.dtype)
    routes = torch.empty((m, TOPK, H), device=x.device, dtype=x.dtype)
    o = torch.empty_like(x)
    v = -1 if variant is None else variant
    out["cuda_align"] = time_fn(lambda: moe_align_block_size(ids, 64, E), iters)
    out["cuda_gate_up_silu"] = time_fn(lambda: lib.gemm(x, gu_p, gu_s, gu_g, inter, tw, s64, e64, n64, m * TOPK, TOPK,
                                                        False, True, variant=v), iters)
    out["cuda_down"] = time_fn(lambda: lib.gemm(inter, d_p, d_s, d_g, routes, tw, s64, e64, n64, m * TOPK, 1,
                                                True, False, variant=v), iters)
    from freetoken.kernel import moe_sum_reduce_triton
    from freetoken.layers import gated_act_and_mul
    out["sum_triton"] = time_fn(lambda: moe_sum_reduce_triton(routes, o), iters)
    out["sum_cuda"] = time_fn(lambda: lib.moe_sum(routes, o), iters)
    cfg = ref._prefill_config(m)
    s, e_, n = moe_align_block_size(ids, cfg["BLOCK_SIZE_M"], E)
    ic1 = torch.empty((m, TOPK, 2 * I), device=x.device, dtype=x.dtype)
    ic2 = torch.empty((m * TOPK, I), device=x.device, dtype=x.dtype)
    out["triton_align"] = time_fn(lambda: moe_align_block_size(ids, cfg["BLOCK_SIZE_M"], E), iters)
    out["triton_gate_up"] = time_fn(lambda: ref._prefill_gemm(x, gu_p, gu_s, gu_g, ic1, tw, s, e_, n, m * TOPK, TOPK,
                                                              False, cfg), iters)
    out["flashinfer_silu"] = time_fn(lambda: gated_act_and_mul("silu", ic1.view(-1, 2 * I), ic2), iters)
    out["triton_down"] = time_fn(lambda: ref._prefill_gemm(ic2, d_p, d_s, d_g, routes, tw, s, e_, n, m * TOPK, 1,
                                                           True, cfg), iters)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="1,17,64,256,1024,2048,4096")
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--routing", default="uniform")
    ap.add_argument("--sweep", action="store_true", help="per-variant kernel timing")
    ap.add_argument("--skip-paired", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    lib = BitExactPrefill(tile=(64, 64, 64))
    report = {"library": lib.path, "build_id": lib.build_id(), "device": torch.cuda.get_device_name(),
              "routing": args.routing, "note": "resident synthetic banks; kernel/layer microbenchmark, not model TTFT",
              "cases": []}
    # HMMA.16816.F32.BF16 roofline on this GPU (register operands, 8 chains per warp)
    scratch = torch.empty(256, device="cuda")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    peak = []
    for blocks_per_sm in (1, 2, 4):
        blocks, iters = sms * blocks_per_sm, 4096
        lib.mma_peak(scratch, blocks, 16)
        ms = time_fn(lambda: lib.mma_peak(scratch, blocks, iters), 3)
        flops = blocks * 8 * iters * 8 * 4096
        peak.append({"blocks_per_sm": blocks_per_sm, "ms": ms, "tflops": flops / ms / 1e9})
    report["hmma_bf16_f32acc_roofline"] = {"sms": sms, "runs": peak, "tflops": max(r["tflops"] for r in peak)}
    print("HMMA roofline:", json.dumps(report["hmma_bf16_f32acc_roofline"]), flush=True)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(20261013)
    banks = tb.make_banks(gen, "realistic", "realistic")
    for m in [int(v) for v in args.m.split(",")]:
        gen.manual_seed(1000 + m)
        x, ids, w = tb.make_tokens(gen, m, "normal", args.routing)
        case = {"m": m}

        def tri():
            return ref.fused_experts_nvfp4(x, *banks, w, ids, E)

        def cud():
            return lib(x, *banks, w, ids, E)
        # sanity (bitwise) before timing
        case["bit_exact_check"] = bool(torch.equal(tb.bits(tri()), tb.bits(cud())))
        if not args.skip_paired:
            for mode in ("eager", "graph"):
                case[mode] = paired({"triton": tri, "cuda": cud}, args.pairs, args.iters, mode)
                print(f"M{m} {mode}: triton {case[mode]['median_ms']['triton']:.4f} ms, cuda "
                      f"{case[mode]['median_ms']['cuda']:.4f} ms, paired ratio "
                      f"{case[mode]['median_paired_ratio_triton_over_cuda']:.3f}x", flush=True)
        case["breakdown_eager_ms"] = breakdown(lib, x, banks, w, ids, args.iters)
        print(f"M{m} breakdown:", json.dumps({k: round(v, 4) for k, v in case["breakdown_eager_ms"].items()}), flush=True)
        if args.sweep:
            sw = {}
            for v in range(lib.variant_count()):
                bm = lib.variant_bm(v)
                if 64 % bm:
                    continue
                b = breakdown(lib, x, banks, w, ids, args.iters, variant=v)
                sw[v] = {"bm": bm, "gate_up_silu": b["cuda_gate_up_silu"], "down": b["cuda_down"]}
                print(f"   M{m} v{v} (BM{bm}): gate_up_silu {b['cuda_gate_up_silu']:.4f} ms, down {b['cuda_down']:.4f} ms",
                      flush=True)
            case["variant_sweep_ms"] = sw
        report["cases"].append(case)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
