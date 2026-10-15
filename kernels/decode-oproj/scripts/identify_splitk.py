"""Identify, bit for bit, the arithmetic of the tensor-wise FP8 GEMM that the decode out/o-projection
runs today: torch._scaled_mm(qa [M,4096] e4m3, W.t(), scale_a=0-d fp32, scale_b=0-d fp32, bf16)
-> cuBLASLt xmma split-K kernel + cublasLt::splitKreduce_kernel.  GPU; run under the GPU lock.

Method (accept a hypothesis only at 100 % of outputs, every M 1..8, random and real weights):
  1. kernel census: which cuBLASLt kernels run for every M (torch.profiler, eager and graph);
  2. raw FP32 partial sums of one QMMA m16n8k32 chain (k-blocks of 32 ascending, from +0) over
     candidate K slices, from the analysis build of libqwen_fp8_gemv (qwen_fp8_gemm_partial_f32);
  3. candidate combinations of the slices (order, start value) and of the scales, each step one
     IEEE FP32 RN torch op, then RN to bf16; compare with cuBLASLt's bf16 bit for bit.

Usage: python identify_splitk.py --analysis-lib <libqwen_fp8_gemv analysis .so> [--model DIR] --out F
"""
from __future__ import annotations

import argparse
import ctypes
import itertools
import json
import sys
import time
from pathlib import Path

import torch

FP8 = torch.float8_e4m3fn
N_OUT, K_IN = 2048, 4096


def codes(shape, gen, dev):
    c = torch.randint(0, 256, shape, generator=gen, device=dev, dtype=torch.uint8)
    return torch.where((c & 0x7F) == 0x7F, c ^ 1, c).view(FP8)


def gaussian_codes(shape, gen, dev, scale):
    x = torch.randn(shape, generator=gen, device=dev) * scale
    return x.clamp(-448, 448).to(FP8)


def reference(qa, w, sa, sb):
    # exactly fp8_pertensor_linear._scaled_mm's uniform branch (scale_b is a 0-d view of the vector)
    return torch._scaled_mm(qa, w.t(), scale_a=sa.reshape(()), scale_b=sb.reshape(()), out_dtype=torch.bfloat16)


class Partial:
    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        f = self.lib.qwen_fp8_gemm_partial_f32
        f.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_longlong] * 6 + [ctypes.c_void_p]
        f.restype = ctypes.c_int
        self.f = f

    def __call__(self, qa, w, k0, k1):
        M, N = qa.shape[0], w.shape[0]
        out = torch.empty(M, N, dtype=torch.float32, device=qa.device)
        rc = self.f(qa.data_ptr() + k0, w.data_ptr() + k0, out.data_ptr(), M, N, k1 - k0, qa.stride(0), w.stride(0),
                    N, torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"partial_f32 rc={rc}")
        return out


def permute_k(t, order, tile_order=None):
    """Reorder K: 64-byte k-tiles into ``tile_order`` (None: unchanged), then the four 16-byte
    quarters of every tile: new quarter j = old quarter order[j]. The partial kernel's first MMA of
    a tile takes new quarters 0,1 and the second 2,3, and walks tiles in ascending order, so a
    consistent permutation of activation and weight emulates another K grouping per instruction
    and another tile order per chain."""
    if order == "0123" and tile_order is None:
        return t
    rows, K = t.shape
    v = t.view(torch.uint8).reshape(rows, K // 64, 4, 16)
    if tile_order is not None:
        v = v[:, torch.tensor(tile_order, device=t.device)]
    if order != "0123":
        v = v[:, :, torch.tensor([int(c) for c in order], device=t.device), :]
    return v.reshape(rows, K).contiguous().view(t.dtype)


def partitions(K):
    seen, out = set(), []
    for unit in (64, 128, 256):
        T = K // unit
        for S in (1, 2, 3, 4, 5, 6):
            q, r = divmod(T, S)
            ceil = -(-T // S)
            rules = {
                "ceil_first": [ceil] * (S - 1) + [T - ceil * (S - 1)],
                "extra_first": [q + (1 if i < r else 0) for i in range(S)],
                "extra_last": [q + (1 if i >= S - r else 0) for i in range(S)],
            }
            for name, sizes in rules.items():
                if any(s <= 0 for s in sizes) or sum(sizes) != T:
                    continue
                b = [0]
                for s in sizes:
                    b.append(b[-1] + s * unit)
                if tuple(b) not in seen:
                    seen.add(tuple(b))
                    out.append({"unit": unit, "splits": S, "rule": name, "bounds": b})
    # interleaved (round-robin) split-K over 64-byte k-tiles: split z owns tiles z, z+S, z+2S, ...
    T = K // 64
    for S in (2, 3, 4):
        tiles = sorted(range(T), key=lambda t: (t % S, t))
        b = [0]
        for z in range(S):
            b.append(b[-1] + 64 * sum(1 for t in range(T) if t % S == z))
        out.append({"unit": 64, "splits": S, "rule": "interleave", "bounds": b, "tile_order": tiles})
    return out


def combos(parts, sa, sb):
    """(name, bf16 tensor) for each combination hypothesis. Every op is one FP32 RN op."""
    ab = sa * sb
    ba = sb * sa
    zero = torch.zeros_like(parts[0])
    seq0 = zero
    for p in parts:
        seq0 = seq0 + p
    seq = parts[0]
    for p in parts[1:]:
        seq = seq + p
    rev0 = zero
    for p in reversed(parts):
        rev0 = rev0 + p
    sums = {"seq_from_+0": seq0, "seq": seq, "rev_from_+0": rev0}
    if len(parts) <= 4:
        for perm in itertools.permutations(range(len(parts))):
            if list(perm) in (list(range(len(parts))), list(reversed(range(len(parts))))):
                continue
            a0 = zero
            a1 = parts[perm[0]]
            for i in perm:
                a0 = a0 + parts[i]
            for i in perm[1:]:
                a1 = a1 + parts[i]
            tag = "".join(map(str, perm))
            sums[f"order{tag}_from_+0"] = a0
            sums[f"order{tag}"] = a1
    for sname, s in sums.items():
        yield f"{sname}|s*RN(sa*sb)", (s * ab).to(torch.bfloat16)
        yield f"{sname}|s*RN(sb*sa)", (s * ba).to(torch.bfloat16)
        yield f"{sname}|RN(RN(s*sa)*sb)", ((s * sa) * sb).to(torch.bfloat16)
        yield f"{sname}|RN(RN(s*sb)*sa)", ((s * sb) * sa).to(torch.bfloat16)
    pre = zero
    for p in parts:
        pre = pre + p * ab
    yield "pre_scaled_seq_from_+0|p*RN(sa*sb)", pre.to(torch.bfloat16)


def kernel_census(dev):
    from torch.profiler import ProfilerActivity, profile

    gen = torch.Generator(device=dev)
    gen.manual_seed(7)
    w = codes((N_OUT, K_IN), gen, dev)
    sa = torch.tensor(0.0123, device=dev)
    sb = torch.full((N_OUT,), 0.00071, device=dev)
    census = {}
    for M in range(1, 9):
        qa = codes((M, K_IN), gen, dev)
        reference(qa, w, sa, sb[0])
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            reference(qa, w, sa, sb[0])
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
        census[M] = names
    # graph capture: torch picks the algorithm per call the same way; record the graph's kernels
    qa = codes((1, K_IN), gen, dev)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        reference(qa, w, sa, sb[0])
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            out_g = reference(qa, w, sa, sb[0])
    g.replay()
    torch.cuda.synchronize()
    eager = reference(qa, w, sa, sb[0])
    census["graph_M1_equal_eager"] = bool(torch.equal(out_g.view(torch.int16), eager.view(torch.int16)))
    return census


def load_real(model_dir, dev, layers):
    from safetensors import safe_open

    idx = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    out = []
    for L in layers:
        attn = (L % 4) == 3
        base = f"model.language_model.layers.{L}." + ("self_attn.o_proj" if attn else "linear_attn.out_proj")
        t = {}
        for suf in ("weight", "weight_scale", "input_scale"):
            name = f"{base}.{suf}"
            with safe_open(str(Path(model_dir) / idx[name]), framework="pt", device="cpu") as f:
                t[suf] = f.get_tensor(name)
        out.append((base, t["weight"].to(dev), t["weight_scale"].float().reshape(-1)[0].to(dev),
                    t["input_scale"].float().reshape(()).to(dev)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-lib", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--layers", default="0,3,13,26,39")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kperms", default="all", help="'all' or ';'-separated quarter orders like 0123;0213")
    ap.add_argument("--max-splits", type=int, default=6)
    ap.add_argument("--units", default="64,128,256")
    ap.add_argument("--no-gauss", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    part = Partial(args.analysis_lib)
    res = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)}
    t0 = time.time()
    res["kernel_census"] = kernel_census(dev)

    cases = []  # (label, qa, w, sa, sb)
    gen = torch.Generator(device=dev)
    gen.manual_seed(0x5EED0)
    w_rand = codes((N_OUT, K_IN), gen, dev)
    w_gauss = gaussian_codes((N_OUT, K_IN), gen, dev, 30.0)
    sas = (0.0123, 2.0 ** -7, 0.37, 1.0)
    sbs = (0.00071, 3.3e-4, 2.0 ** -11, 0.051)
    for M in range(1, 9):
        for i in range(2):
            cases.append((f"rand_codes M={M} #{i}", codes((M, K_IN), gen, dev), w_rand,
                          torch.tensor(sas[(M + i) % 4], device=dev), torch.tensor(sbs[(M + 2 * i) % 4], device=dev)))
            if not args.no_gauss:
                cases.append((f"gauss M={M} #{i}", gaussian_codes((M, K_IN), gen, dev, 4.0), w_gauss,
                              torch.tensor(sas[(M + i + 1) % 4], device=dev), torch.tensor(sbs[(M + i) % 4], device=dev)))
    if args.model:
        for base, w, ws, xs in load_real(args.model, dev, [int(x) for x in args.layers.split(",")]):
            for M in range(1, 9):
                cases.append((f"real {base} M={M}", gaussian_codes((M, K_IN), gen, dev, 20.0), w, xs, ws))

    cands = [c for c in partitions(K_IN) if c["splits"] <= args.max_splits
             and c["unit"] in {int(u) for u in args.units.split(",")}]
    if args.kperms == "all":
        kperms = ["".join(map(str, p)) for p in itertools.permutations(range(4))]
    else:
        kperms = args.kperms.split(";")
    alive = None  # set of (partition index, combo name) consistent with every case so far
    per_case = []
    for label, qa, w, sa, sb in cases:
        ref = reference(qa, w, sa, sb).view(torch.int16)
        ok_here = set()
        best = (None, -1)
        for kp in kperms:
            permuted = {}
            cache = {}
            for pi, c in enumerate(cands):
                to = c.get("tile_order")
                tkey = tuple(to) if to else None
                if tkey not in permuted:
                    permuted[tkey] = (permute_k(qa, kp, to), permute_k(w, kp, to))
                qp, wp = permuted[tkey]
                b = c["bounds"]
                parts = []
                for k0, k1 in zip(b[:-1], b[1:]):
                    if (tkey, k0, k1) not in cache:
                        cache[(tkey, k0, k1)] = part(qp, wp, k0, k1)
                    parts.append(cache[(tkey, k0, k1)])
                for name, val in combos(parts, sa, sb):
                    eq = int((val.view(torch.int16) == ref).sum())
                    key = (kp, pi, name)
                    if eq == ref.numel():
                        ok_here.add(key)
                    if eq > best[1]:
                        best = (key, eq)
        alive = ok_here if alive is None else alive & ok_here
        per_case.append({"case": label, "n_matching_hypotheses": len(ok_here), "best": [best[0][0], {k: v for k, v in cands[best[0][1]].items() if k != "tile_order"}, best[0][2]],
                         "best_equal": best[1], "elements": ref.numel()})
        print(f"{label:60s} full-match hypotheses {len(ok_here):3d}  surviving {len(alive):3d}", flush=True)
    res["cases"] = per_case
    res["surviving"] = [{"kperm": kp, "partition": {k: v for k, v in cands[pi].items() if k != "tile_order"}, "combine_scale": name} for kp, pi, name in sorted(alive or [])]
    res["n_cases"] = len(cases)
    res["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["surviving"], indent=1))
    print("kernel census M=1:", res["kernel_census"][1])
    return 0 if res["surviving"] else 1


if __name__ == "__main__":
    sys.exit(main())
