"""Component test of libqwen_oproj_gemv against the production cuBLASLt call, bit for bit, plus a
micro-benchmark. GPU; run under the GPU lock with the inference service stopped.

Reference: exactly the uniform-scale branch of fp8_pertensor_linear._scaled_mm (also used verbatim
by qwen3_5_moe/gdn.py and kernel/cuda_attention_gate.py):
    torch._scaled_mm(qa, W.t(), scale_a=input_scale.reshape(()), scale_b=weight_scale[0].reshape(()),
                     out_dtype=torch.bfloat16)

Cases (every M in 1..8, every library variant, eager and inside a CUDA graph):
  * synthetic: uniform e4m3 codes (all 254 finite values), gaussian codes, edge codes (+-0, +-448,
    subnormals, exact cancellation rows), negative-zero activations, wide scales (bf16 overflow to
    inf, subnormal results), a non-contiguous activation row stride;
  * real: out_proj / o_proj weight + weight_scale + input_scale of every layer in --layers, with
    gaussian and heavy-tailed activations.
Comparison is on raw bf16 bits (int16 views): -0 vs +0 and NaN payloads count.

Usage: python test_bitexact.py --lib build/libqwen_oproj_gemv.so [--model DIR] [--layers all]
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
N_OUT, K_IN = 2048, 4096


class Lib:
    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        L = self.lib
        for name in ("qwen_oproj_abi_version", "qwen_oproj_max_m", "qwen_oproj_splits", "qwen_oproj_variant_count",
                     "qwen_oproj_init"):
            getattr(L, name).restype = ctypes.c_int
            getattr(L, name).argtypes = []
        L.qwen_oproj_source_sha256.restype = ctypes.c_char_p
        L.qwen_oproj_error.restype = ctypes.c_char_p
        L.qwen_oproj_error.argtypes = [ctypes.c_int]
        L.qwen_oproj_tensorwise.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_longlong] * 6 + [ctypes.c_int, ctypes.c_void_p]
        L.qwen_oproj_tensorwise.restype = ctypes.c_int
        self.abi = L.qwen_oproj_abi_version()
        self.variants = L.qwen_oproj_variant_count()
        rc = L.qwen_oproj_init()
        if rc:
            raise RuntimeError(f"qwen_oproj_init: {rc}")

    def __call__(self, qa, w, sa, sb, variant, out=None):
        M, K = qa.shape
        N = w.shape[0]
        out = torch.empty((M, N), dtype=torch.bfloat16, device=qa.device) if out is None else out
        rc = self.lib.qwen_oproj_tensorwise(qa.data_ptr(), w.data_ptr(), sa.data_ptr(), sb.data_ptr(), out.data_ptr(),
                                            M, N, K, qa.stride(0), w.stride(0), out.stride(0), variant,
                                            torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"qwen_oproj_tensorwise rc={rc} ({self.lib.qwen_oproj_error(rc)})")
        return out


def reference(qa, w, sa, sb):
    return torch._scaled_mm(qa, w.t(), scale_a=sa.reshape(()), scale_b=sb.reshape(()), out_dtype=torch.bfloat16)


def codes(shape, gen, dev):
    c = torch.randint(0, 256, shape, generator=gen, device=dev, dtype=torch.uint8)
    return torch.where((c & 0x7F) == 0x7F, c ^ 1, c).view(FP8)


def gaussian(shape, gen, dev, scale):
    return (torch.randn(shape, generator=gen, device=dev) * scale).clamp(-448, 448).to(FP8)


def heavy(shape, gen, dev):
    x = torch.randn(shape, generator=gen, device=dev)
    x = x * torch.exp(torch.randn(shape, generator=gen, device=dev) * 2.0)
    return x.clamp(-448, 448).to(FP8)


EDGE = (0x00, 0x80, 0x7E, 0xFE, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88, 0x38, 0xB8)


def edge_act(M, K, gen, dev):
    t = torch.tensor(EDGE, dtype=torch.uint8, device=dev)
    half = t[torch.randint(0, len(EDGE), (M, K // 2), generator=gen, device=dev)]
    return torch.cat([half, half ^ 0x80], 1).contiguous().view(FP8)


def synthetic_cases(gen, dev):
    w_codes = codes((N_OUT, K_IN), gen, dev)
    wc = w_codes.view(torch.uint8)
    wc[:32, K_IN // 2:] = wc[:32, :K_IN // 2]  # exact-cancellation rows against edge_act
    w_gauss = gaussian((N_OUT, K_IN), gen, dev, 40.0)
    scal = [(0.0123, 0.00071), (2.0 ** -7, 3.3e-4), (0.37, 2.0 ** -11), (1.0, 0.051), (2.0 ** 60, 2.0 ** 70),
            (2.0 ** -70, 2.0 ** -60), (5.5e-4, 0.25), (3.1e-3, 7.7e-3)]
    out = []
    for M in range(1, 9):
        for j, (a, b) in enumerate(scal):
            sa = torch.tensor(a, device=dev)
            sb = torch.tensor(b, device=dev)
            kind = j % 4
            if kind == 0:
                out.append((f"codes M={M} s{j}", codes((M, K_IN), gen, dev), w_codes, sa, sb))
            elif kind == 1:
                out.append((f"edge M={M} s{j}", edge_act(M, K_IN, gen, dev), w_codes, sa, sb))
            elif kind == 2:
                out.append((f"gauss M={M} s{j}", gaussian((M, K_IN), gen, dev, 3.0), w_gauss, sa, sb))
            else:
                out.append((f"heavy M={M} s{j}", heavy((M, K_IN), gen, dev), w_gauss, sa, sb))
        negz = torch.full((M, K_IN), 0x80, dtype=torch.uint8, device=dev)
        negz[:, ::7] = 0x00
        out.append((f"neg-zero M={M}", negz.view(FP8), w_codes, torch.tensor(0.5, device=dev), torch.tensor(0.25, device=dev)))
        big = torch.empty((M, K_IN + 64), dtype=torch.uint8, device=dev)
        big.view(FP8)[:, :K_IN] = gaussian((M, K_IN), gen, dev, 5.0)
        out.append((f"strided M={M}", big.view(FP8)[:, :K_IN], w_gauss, torch.tensor(0.02, device=dev),
                    torch.tensor(0.003, device=dev)))
    return out


def real_cases(model_dir, layers, gen, dev):
    from safetensors import safe_open

    idx = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    for L in layers:
        base = f"model.language_model.layers.{L}." + ("self_attn.o_proj" if L % 4 == 3 else "linear_attn.out_proj")
        t = {}
        for suf in ("weight", "weight_scale", "input_scale"):
            name = f"{base}.{suf}"
            with safe_open(str(Path(model_dir) / idx[name]), framework="pt", device="cpu") as f:
                t[suf] = f.get_tensor(name)
        w = t["weight"].to(dev).contiguous()
        ws = t["weight_scale"].float().reshape(-1).to(dev)
        if ws.numel() != 1 and not bool((ws == ws[0]).all()):
            raise RuntimeError(f"{base}: weight_scale is not uniform")
        ws = ws.expand(N_OUT).contiguous()  # FreeToken stores the per-row vector; [0] is the scalar
        xs = t["input_scale"].float().reshape(()).to(dev)
        for M in range(1, 9):
            yield (f"real L{L} gauss M={M}", gaussian((M, K_IN), gen, dev, 60.0), w, xs, ws[0])
            yield (f"real L{L} heavy M={M}", heavy((M, K_IN), gen, dev), w, xs, ws[0])


def compare(lib, label, qa, w, sa, sb, results, graph=False):
    ref = reference(qa, w, sa, sb).view(torch.int16)
    bad = {}
    for v in range(lib.variants):
        if graph:
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                out = torch.empty((qa.shape[0], N_OUT), dtype=torch.bfloat16, device=qa.device)
                lib(qa, w, sa, sb, v, out)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                out.fill_(0)
                with torch.cuda.graph(g, stream=s):
                    lib(qa, w, sa, sb, v, out)
            g.replay()
            torch.cuda.synchronize()
        else:
            out = lib(qa, w, sa, sb, v)
        n = int((out.view(torch.int16) != ref).sum())
        if n:
            bad[v] = n
    results["cases"] += 1
    results["elements"] += ref.numel() * lib.variants
    if bad:
        results["failures"].append({"case": label, "graph": graph, "mismatched_per_variant": bad})
    return not bad


def bench(lib, dev, pairs=5, layers=40, iters=20):
    """CUDA-graph timing of `layers` o-projections over distinct weights (larger than L2), cuBLAS vs
    every library variant, M in (1, 2, 4, 8); alternating order, median per pair."""
    gen = torch.Generator(device=dev)
    gen.manual_seed(99)
    ws = [codes((N_OUT, K_IN), gen, dev) for _ in range(layers)]
    sa = torch.tensor(0.01, device=dev)
    sb = torch.tensor(0.001, device=dev)
    res = {}
    for M in (1, 2, 4, 8):
        qa = codes((M, K_IN), gen, dev)
        outs = [torch.empty((M, N_OUT), dtype=torch.bfloat16, device=dev) for _ in range(layers)]

        def capture(fn):
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                fn()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    fn()
            return g

        graphs = {"cublas": capture(lambda: [outs[i].copy_(reference(qa, ws[i], sa, sb)) for i in range(layers)])}
        graphs["cublas_nocopy"] = capture(lambda: [reference(qa, ws[i], sa, sb) for i in range(layers)])
        for v in range(lib.variants):
            graphs[f"v{v}"] = capture(lambda v=v: [lib(qa, ws[i], sa, sb, v, outs[i]) for i in range(layers)])
        times = {k: [] for k in graphs}
        for p in range(pairs):
            order = list(graphs) if p % 2 == 0 else list(reversed(list(graphs)))
            for k in order:
                g = graphs[k]
                g.replay()
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(iters):
                    g.replay()
                e1.record()
                torch.cuda.synchronize()
                times[k].append(e0.elapsed_time(e1) * 1e3 / iters / layers)
        med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
        res[f"M{M}"] = {"us_per_projection_median": {k: round(v, 3) for k, v in med.items()},
                        "gbps": {k: round(N_OUT * K_IN / (v * 1e-6) / 1e9, 1) for k, v in med.items()},
                        "samples_us": {k: [round(x, 3) for x in v] for k, v in times.items()}}
        print(f"M={M}", json.dumps(res[f"M{M}"]["us_per_projection_median"]), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    lib = Lib(args.lib)
    t0 = time.time()
    free0 = torch.cuda.mem_get_info()[0]
    res = {"abi": lib.abi, "variants": lib.variants, "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
           "cases": 0, "elements": 0, "failures": []}
    gen = torch.Generator(device=dev)
    gen.manual_seed(0x0BB0)
    for label, qa, w, sa, sb in synthetic_cases(gen, dev):
        compare(lib, label, qa, w, sa, sb, res)
        if label.startswith(("codes", "edge")) and "s0" in label or label.startswith("neg-zero"):
            compare(lib, label, qa, w, sa, sb, res, graph=True)
    print(f"synthetic: {res['cases']} cases, failures {len(res['failures'])}", flush=True)
    if args.model:
        layers = list(range(40)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
        for label, qa, w, sa, sb in real_cases(args.model, layers, gen, dev):
            compare(lib, label, qa, w, sa, sb, res, graph=label.endswith("M=1") or label.endswith("M=8"))
        print(f"+ real: {res['cases']} cases, failures {len(res['failures'])}", flush=True)
    torch.cuda.synchronize()
    res["free_vram_mib_before_after"] = [free0 >> 20, torch.cuda.mem_get_info()[0] >> 20]
    res["seconds"] = round(time.time() - t0, 1)
    res["pass"] = not res["failures"]
    if args.bench:
        res["bench"] = bench(lib, dev)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("bench",)}, indent=1)[:3000])
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
