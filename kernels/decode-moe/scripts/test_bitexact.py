"""Component test of libqwen_moe_decode against FreeToken's production NVFP4 decode GEMV (Triton
_decode_nvfp4_marlin_kernel via moe/fused_nvfp4._decode_gemm_marlin), bit for bit, plus the whole
decode MoE (gate_up -> gated act -> down -> sum) with the GEMV swapped, and a micro-benchmark.
GPU; run under the GPU lock with the inference service stopped.

Banks: real Qwen3.6 experts of --layers packed exactly like TritonNvfp4MoEKernel.pack (gate|up rows
concatenated, per-row fp32 globals = weight_scale_2), and synthetic banks (every e2m1 code, every
finite e4m3 scale code incl. negative ones, globals spanning 2^-40..2^30). Activations: gaussian,
heavy-tailed, edge (+-0, bf16 subnormals, huge), routes with repeated slots across tokens.
Every M in 1..8, top_k in {1, 2, 4, 8}, every library variant, eager and CUDA graph.

Usage: python test_bitexact.py --lib build/libqwen_moe_decode.so [--model DIR] [--layers 0,13,26,39]
       [--bench] --out result.json
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import torch

FP8 = torch.float8_e4m3fn
H, I = 2048, 512


class Lib:
    def __init__(self, path):
        L = ctypes.CDLL(path)
        self.lib = L
        L.qwen_moe_decode_abi_version.restype = ctypes.c_int
        L.qwen_moe_decode_variant_count.restype = ctypes.c_int
        L.qwen_moe_decode_error.restype = ctypes.c_char_p
        L.qwen_moe_decode_error.argtypes = [ctypes.c_int]
        f = L.qwen_moe_decode_gemv
        f.argtypes = ([ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong,
                       ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_longlong,
                       ctypes.c_longlong, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
                       ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_longlong,
                       ctypes.c_longlong] + [ctypes.c_longlong] * 4 + [ctypes.c_int] * 3 + [ctypes.c_void_p])
        f.restype = ctypes.c_int
        self.f = f
        self.abi = L.qwen_moe_decode_abi_version()
        self.variants = L.qwen_moe_decode_variant_count()
        self.variant = 0

    def gemm(self, a, packed, scale, glob, c, topk_weights, topk_ids, mul_routed_weight, a_row_is_route):
        """Same calling convention as fused_nvfp4._decode_gemm_marlin."""
        M, top_k = topk_ids.shape
        S, N, KB = packed.shape
        rc = self.f(a.data_ptr(), a.stride(0), packed.data_ptr(), packed.stride(0), packed.stride(1),
                    scale.data_ptr(), scale.stride(0), scale.stride(1), glob.data_ptr(), glob.stride(0),
                    glob.stride(1), c.data_ptr(), c.stride(0), c.stride(1), c.stride(2),
                    topk_weights.data_ptr(), topk_weights.stride(0), topk_weights.stride(1),
                    topk_ids.data_ptr(), topk_ids.stride(0), topk_ids.stride(1),
                    M, top_k, N, KB * 2, int(a_row_is_route), int(mul_routed_weight), self.variant,
                    torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"qwen_moe_decode_gemv rc={rc} ({self.lib.qwen_moe_decode_error(rc)})")


def real_banks(model_dir, layer, experts, dev):
    from safetensors import safe_open

    idx = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}

    def get(name):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(str(Path(model_dir) / f), framework="pt", device="cpu")
        return handles[f].get_tensor(name)

    gu, gus, gug, dn, dns, dng = [], [], [], [], [], []
    for e in experts:
        b = f"model.language_model.layers.{layer}.mlp.experts.{e}."
        g, u, d = (get(b + f"{p}.weight") for p in ("gate_proj", "up_proj", "down_proj"))
        gs, us, ds = (get(b + f"{p}.weight_scale") for p in ("gate_proj", "up_proj", "down_proj"))
        g2, u2, d2 = (get(b + f"{p}.weight_scale_2").float().reshape(()) for p in ("gate_proj", "up_proj", "down_proj"))
        gu.append(torch.cat([g, u], 0))
        gus.append(torch.cat([gs, us], 0))
        gug.append(torch.cat([g2.expand(I), u2.expand(I)]))
        dn.append(d)
        dns.append(ds)
        dng.append(d2.expand(H))
    st = lambda xs: torch.stack(xs).contiguous().to(dev)  # noqa: E731
    return (st(gu).view(torch.uint8), st(gus).view(FP8), st(gug).float(), st(dn).view(torch.uint8), st(dns).view(FP8),
            st(dng).float())


def synthetic_banks(S, gen, dev):
    def scales(shape):
        c = torch.randint(0, 256, shape, generator=gen, device=dev, dtype=torch.uint8)
        return torch.where((c & 0x7F) == 0x7F, c ^ 1, c).view(FP8)

    def glob(shape):
        return torch.exp2(torch.empty(shape, device=dev).uniform_(-40, 30, generator=gen))

    gu = torch.randint(0, 256, (S, 2 * I, H // 2), generator=gen, device=dev, dtype=torch.uint8)
    dn = torch.randint(0, 256, (S, H, I // 2), generator=gen, device=dev, dtype=torch.uint8)
    return gu, scales((S, 2 * I, H // 16)), glob((S, 2 * I)), dn, scales((S, H, I // 16)), glob((S, H))


def activations(kind, shape, gen, dev):
    if kind == "gauss":
        return torch.randn(shape, generator=gen, device=dev).to(torch.bfloat16)
    if kind == "heavy":
        x = torch.randn(shape, generator=gen, device=dev) * torch.exp(torch.randn(shape, generator=gen, device=dev) * 3)
        return x.to(torch.bfloat16)
    # edge: +-0, bf16 subnormals, huge, ordinary
    table = torch.tensor([0.0, -0.0, 1e-39, -1e-39, 9.2e-41, 3.0e38, -3.0e38, 1.0, -2.5, 65504.0],
                         device=dev).to(torch.bfloat16)
    return table[torch.randint(0, len(table), shape, generator=gen, device=dev)]


def routes(M, top_k, S, gen, dev):
    ids = torch.stack([torch.randperm(S, generator=gen, device=dev)[:top_k] for _ in range(M)]).to(torch.int32)
    if M > 1:
        ids[1:, 0] = ids[0, 0]  # the same slot routed by several tokens
    w = torch.rand((M, top_k), generator=gen, device=dev)
    return ids.contiguous(), w.contiguous()


def run_pair(lib, ref_gemm, a, banks, ids, w, which, variant, graph):
    gu, gus, gug, dn, dns, dng = banks
    M, top_k = ids.shape
    if which == "gate_up":
        args = (gu, gus, gug)
        N, mulw, aroute = 2 * I, False, False
    else:
        args = (dn, dns, dng)
        N, mulw, aroute = H, True, True
    c_ref = torch.full((M, top_k, N), float("nan"), dtype=torch.bfloat16, device=a.device)
    c_new = torch.full_like(c_ref, float("nan"))
    ref_gemm(a, *args, c_ref, w, ids, mulw, aroute)
    lib.variant = variant
    if graph:
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            lib.gemm(a, *args, c_new, w, ids, mulw, aroute)
            torch.cuda.synchronize()
            c_new.fill_(float("nan"))
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                lib.gemm(a, *args, c_new, w, ids, mulw, aroute)
        g.replay()
        torch.cuda.synchronize()
    else:
        lib.gemm(a, *args, c_new, w, ids, mulw, aroute)
    return int((c_new.view(torch.int16) != c_ref.view(torch.int16)).sum()), c_ref.numel()


def end_to_end(lib, banks, M, top_k, gen, dev):
    """Whole decode MoE: production function vs the same body with the GEMV swapped."""
    import freetoken.moe.fused_nvfp4 as fn

    ids, w = routes(M, top_k, banks[0].shape[0], gen, dev)
    x = activations("gauss", (M, H), gen, dev)
    ref = fn.fused_experts_decode_nvfp4_marlin(x, *banks, w, ids, "silu", False)
    new = fn._fused_experts_decode_nvfp4(lib.gemm, x, *banks, w, ids, "silu", False)
    return int((new.view(torch.int16) != ref.view(torch.int16)).sum()), ref.numel()


def bench(lib, ref_gemm, banks_list, dev, pairs=5, iters=20):
    """CUDA-graph timing of 40 layers x (gate_up + down) over distinct banks (larger than L2), M=1 and 4."""
    gen = torch.Generator(device=dev)
    gen.manual_seed(5)
    res = {}
    for M in (1, 4):
        cases = []
        for banks in banks_list:
            ids, w = routes(M, 8, banks[0].shape[0], gen, dev)
            x = activations("gauss", (M, H), gen, dev)
            a2 = activations("gauss", (M * 8, I), gen, dev)
            c1 = torch.empty((M, 8, 2 * I), dtype=torch.bfloat16, device=dev)
            c2 = torch.empty((M, 8, H), dtype=torch.bfloat16, device=dev)
            cases.append((banks, ids, w, x, a2, c1, c2))

        def body(fn, part):
            for banks, ids, w, x, a2, c1, c2 in cases:
                if part in ("both", "gate_up"):
                    fn(x, banks[0], banks[1], banks[2], c1, w, ids, False, False)
                if part in ("both", "down"):
                    fn(a2, banks[3], banks[4], banks[5], c2, w, ids, True, True)

        def capture(fn, part, variant=None):
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                if variant is not None:
                    lib.variant = variant
                body(fn, part)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    body(fn, part)
            return g

        graphs = {}
        for part in ("both", "gate_up", "down"):
            graphs[f"triton/{part}"] = capture(ref_gemm, part)
            for v in range(lib.variants):
                graphs[f"v{v}/{part}"] = capture(lib.gemm, part, v)
        times = {k: [] for k in graphs}
        for p in range(pairs):
            for k in (list(graphs) if p % 2 == 0 else list(reversed(list(graphs)))):
                graphs[k].replay()
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(iters):
                    graphs[k].replay()
                e1.record()
                torch.cuda.synchronize()
                times[k].append(e0.elapsed_time(e1) * 1e3 / iters / len(cases))
        med = {k: round(sorted(v)[len(v) // 2], 2) for k, v in times.items()}
        bytes_layer = 8 * M ** 0 * ((2 * I * H // 2) * (1 + 1 / 8) + (H * I // 2) * (1 + 1 / 8))
        res[f"M{M}"] = {"us_per_layer_median": med,
                        "gbps_unique_bytes_M1": {k: round(bytes_layer / (v * 1e-6) / 1e9, 1) for k, v in med.items()},
                        "samples": {k: [round(x, 2) for x in v] for k, v in times.items()}}
        print(f"M={M}", json.dumps(med), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--layers", default="0,13,26,39")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    import freetoken.moe.fused_nvfp4 as fn

    lib = Lib(args.lib)
    ref_gemm = fn._decode_gemm_marlin
    t0 = time.time()
    free0 = torch.cuda.mem_get_info()[0]
    gen = torch.Generator(device=dev)
    gen.manual_seed(0x30E)
    bank_sets = [("synthetic", synthetic_banks(16, gen, dev))]
    if args.model:
        for L in [int(x) for x in args.layers.split(",")]:
            experts = torch.randperm(256, generator=torch.Generator().manual_seed(L))[:16].tolist()
            bank_sets.append((f"real L{L}", real_banks(args.model, L, experts, dev)))
    res = {"abi": lib.abi, "variants": lib.variants, "torch": torch.__version__, "comparisons": 0, "elements": 0,
           "failures": []}
    for bname, banks in bank_sets:
        for top_k in (8, 4, 2, 1):
            for M in range(1, 9):
                if top_k != 8 and M not in (1, 3, 8):
                    continue
                for kind in ("gauss", "heavy", "edge"):
                    ids, w = routes(M, top_k, banks[0].shape[0], gen, dev)
                    x = activations(kind, (M, H), gen, dev)
                    a2 = activations(kind, (M * top_k, I), gen, dev)
                    for which, a in (("gate_up", x), ("down", a2)):
                        for v in range(lib.variants):
                            graph = kind == "gauss" and M in (1, 8) and top_k == 8
                            bad, n = run_pair(lib, ref_gemm, a, banks, ids, w, which, v, graph)
                            res["comparisons"] += 1
                            res["elements"] += n
                            if bad:
                                res["failures"].append({"banks": bname, "which": which, "M": M, "top_k": top_k,
                                                        "kind": kind, "variant": v, "graph": graph, "mismatched": bad})
        print(f"{bname}: comparisons {res['comparisons']}, failures {len(res['failures'])}", flush=True)
    # whole decode MoE with the GEMV swapped (default variant)
    e2e = []
    lib.variant = 0
    for bname, banks in bank_sets:
        for M in (1, 2, 8):
            bad, n = end_to_end(lib, banks, M, 8, gen, dev)
            e2e.append({"banks": bname, "M": M, "mismatched": bad, "elements": n})
    res["end_to_end"] = e2e
    res["end_to_end_pass"] = all(x["mismatched"] == 0 for x in e2e)
    torch.cuda.synchronize()
    res["free_vram_mib_before_after"] = [free0 >> 20, torch.cuda.mem_get_info()[0] >> 20]
    res["seconds"] = round(time.time() - t0, 1)
    res["pass"] = not res["failures"] and res["end_to_end_pass"]
    if res["failures"]:
        print(json.dumps(res["failures"][:20], indent=0))
    if args.bench:
        gen2 = torch.Generator(device=dev)
        gen2.manual_seed(1)
        banks_list = [synthetic_banks(16, gen2, dev) for _ in range(8)]  # 8 x 16 slots x 1.77 MB > L2
        res["bench"] = bench(lib, ref_gemm, banks_list * 5, dev)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("bench", "failures")}, indent=1)[:3000])
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
