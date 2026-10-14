#!/usr/bin/env python3
"""Speed of the fused FLA chunk_h + chunk_o CUDA kernel vs the Triton kernels (no profiler).

kernels : single sequence T in --tokens. before = Triton chunk_h (BV32/nw4/ns2) + chunk_o
          (BK128/BV64/nw4/ns2) exactly as chunk_prenormed launches them; after = fused kernel.
          Also chunk_h / chunk_o / kkt_solve / recompute_w_u alone (context). 5 interleaved pairs,
          median of --iters launches each.
layer   : one GDN prefill layer core (qkvz/ba -> FP8 out_proj input) with QWEN36_CUDA_GDN=1:
          chunk_prenormed (Triton chunk_h/chunk_o) vs chunk_fused, 5 interleaved pairs.
--profile: a few launches of every kernel inside cudaProfilerStart/Stop (for ncu), T=2048/4096.
Microbenchmarks are not model speed. Writes JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qwen36_gdn import GdnLib  # noqa: E402
import test_gdn_bitexact as TB  # noqa: E402
import test_fla_fused as TF  # noqa: E402

HK, HV, D = 16, 32, 128


def timeit(fn, iters, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def pairs(fa, fb, n_pairs, iters):
    out = []
    for i in range(n_pairs):
        if i % 2 == 0:
            ta, tb = timeit(fa, iters), timeit(fb, iters)
        else:
            tb, ta = timeit(fb, iters), timeit(fa, iters)
        out.append((ta, tb))
    lr = [math.log(a / b) for a, b in out]
    m = statistics.mean(lr)
    sd = statistics.stdev(lr) if len(lr) > 1 else 0.0
    t975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(len(lr), 2.776)
    half = t975 * sd / math.sqrt(len(lr)) if len(lr) > 1 else 0.0
    return {"before_ms": [round(a, 5) for a, _ in out], "after_ms": [round(b, 5) for _, b in out],
            "before_median_ms": round(statistics.median(a for a, _ in out), 5),
            "after_median_ms": round(statistics.median(b for _, b in out), 5),
            "speedup_geomean": round(math.exp(m), 4),
            "speedup_ci95": [round(math.exp(m - half), 4), round(math.exp(m + half), 4)]}


def setup(lib, Tn, gen, dev, snapshot=False):
    from freetoken.kernel.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from freetoken.kernel.fla.index import prepare_chunk_indices, prepare_chunk_offsets
    P = TB.synthetic_params(gen, dev)
    q, k, v, g, beta, cu, S = TF.make_qkvgb(lib, P, [Tn], gen, dev, False)
    ci = prepare_chunk_indices(cu, 64)
    co = prepare_chunk_offsets(cu, 64)
    w, u, A = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta, cu_seqlens=cu, chunk_indices=ci)
    idx = torch.tensor([1], dtype=torch.int32, device=dev)
    state = (torch.randn(S, HV, D, D, generator=gen) * 0.05).float().to(dev)
    rows = dst = None
    if snapshot:
        rows = torch.tensor([int(co[0].item()) + max(0, (Tn - 8) // 64)], dtype=torch.int64, device=dev)
        dst = torch.tensor([S - 1], dtype=torch.int64, device=dev)
    return dict(q=q, k=k, v=v, g=g, beta=beta, cu=cu, ci=ci, co=co, w=w, u=u, A=A, idx=idx, state=state,
                rows=rows, dst=dst, Tn=Tn)


def triton_launchers(X):
    import triton
    from freetoken.kernel.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from freetoken.kernel.fla.chunk_o import chunk_fwd_kernel_o
    from freetoken.kernel.fla.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
    from freetoken.kernel.fla.wy_fast import recompute_w_u_fwd
    Tn = X["Tn"]
    st = X["state"].clone()
    box = {}

    def ch():
        box["h"], box["vn"] = chunk_gated_delta_rule_fwd_h(
            k=X["k"], w=X["w"], u=X["u"], g=X["g"], initial_state=st, initial_state_indices=X["idx"],
            cu_seqlens=X["cu"], chunk_indices=X["ci"], snapshot_h_rows=X["rows"],
            snapshot_state_indices=X["dst"])
    ch()
    o = torch.empty_like(X["v"])

    def co():
        chunk_fwd_kernel_o[(2, len(X["ci"]), HV)](X["q"], X["k"], box["vn"], box["h"], X["g"], o, X["cu"],
                                                  X["ci"], D ** -0.5, T=Tn, H=HV, Hg=HK, K=D, V=D, BT=64,
                                                  BK=128, BV=64, USE_G=True, IS_VARLEN=True,
                                                  num_warps=4, num_stages=2)
    A = torch.zeros_like(X["A"])

    def kkt():
        chunk_gated_delta_rule_fwd_kkt_solve_kernel[(len(X["ci"]), HV)](
            k=X["k"], g=X["g"], beta=X["beta"], A=A, cu_seqlens=X["cu"], chunk_indices=X["ci"], T=Tn,
            H=HV, Hg=HK, K=D, BT=64, BC=16)

    def wu():
        recompute_w_u_fwd(k=X["k"], v=X["v"], beta=X["beta"], A=X["A"], g_cumsum=X["g"],
                          cu_seqlens=X["cu"], chunk_indices=X["ci"])

    def both():
        ch()
        co()
    return ch, co, both, kkt, wu


def fused_launcher(lib, X):
    st = X["state"].clone()
    o = torch.empty(1, X["Tn"], HV, D, dtype=torch.bfloat16, device=X["q"].device)
    snap = X["rows"] is not None

    def f():
        lib.fla_chunk_ho(X["q"], X["k"], X["w"], X["u"], X["g"], st, X["idx"], X["cu"], D ** -0.5,
                         chunk_offsets=X["co"] if snap else None, snap_rows_i64=X["rows"],
                         snap_dst_i64=X["dst"], out=o)
    return f


def fused_wu_launcher(lib, X):
    st = X["state"].clone()
    o = torch.empty(1, X["Tn"], HV, D, dtype=torch.bfloat16, device=X["q"].device)
    snap = X["rows"] is not None

    def f():
        lib.fla_wu_chunk_ho(X["q"], X["k"], X["v"], X["A"], X["beta"], X["g"], st, X["idx"], X["cu"],
                            D ** -0.5, chunk_offsets=X["co"] if snap else None, snap_rows_i64=X["rows"],
                            snap_dst_i64=X["dst"], out=o)
    return f


def bench_kernels(lib, Tn, n_pairs, iters, gen, dev, snapshot=False):
    X = setup(lib, Tn, gen, dev, snapshot)
    ch, co, both, kkt, wu = triton_launchers(X)
    fu = fused_launcher(lib, X)
    fw = fused_wu_launcher(lib, X)

    def tri_wuho():
        wu()
        both()
    res = {"T": Tn, "snapshot": snapshot,
           "pair_triton_h+o_vs_fused_ho": pairs(both, fu, n_pairs, iters),
           "pair_triton_wu+h+o_vs_fused_wu": pairs(tri_wuho, fw, n_pairs, iters)}
    res["alone_ms"] = {"triton_chunk_h": round(timeit(ch, iters), 5), "triton_chunk_o": round(timeit(co, iters), 5),
                       "triton_kkt_solve": round(timeit(kkt, iters), 5),
                       "triton_recompute_w_u": round(timeit(wu, iters), 5),
                       "fused_chunk_ho": round(timeit(fu, iters), 5),
                       "fused_wu_chunk_ho": round(timeit(fw, iters), 5)}
    return res


def bench_layer(lib, Tn, n_pairs, iters, gen, dev, variant):
    """One GDN prefill layer core, QWEN36_CUDA_GDN=1 path, chunk_prenormed vs chunk_fused."""
    P = TB.synthetic_params(gen, dev)
    qkvz, ba = TB.make_inputs(P, [Tn], gen, dev, False)
    cu = torch.tensor([0, Tn], dtype=torch.int64, device=dev)
    idx = torch.tensor([1], dtype=torch.int32, device=dev)
    has = torch.tensor([True], device=dev)
    conv0 = (torch.randn(3, TB.CONV_DIM, 3, generator=gen) * 0.8).to(torch.bfloat16).to(dev)
    rec0 = (torch.randn(3, HV, D, D, generator=gen) * 0.05).float().to(dev)
    ca, cb = conv0.clone(), conv0.clone()
    ra, rb = rec0.clone(), rec0.clone()
    fa = lambda: TB.candidate_prefill(lib, P, qkvz, ba, ca, ra, cu, idx, has, None, None, None, want_bf16=False)

    def fb():
        out, path = TF.candidate_prefill_fused(lib, P, qkvz, ba, cb, rb, cu, idx, has, None, None, None,
                                               variant=variant)
        assert path == "fused-" + variant
        return out
    return {"T": Tn, "variant": variant, "pair_layer_prenormed_vs_fused": pairs(fa, fb, n_pairs, iters)}


def profile(lib, gen, dev):
    """Launch every kernel twice per T inside the profiler range (ncu --profile-from-start off).
    Order per T: kkt_solve, recompute_w_u, chunk_h, chunk_o, fused_ho, fused_wu."""
    Xs = [setup(lib, Tn, gen, dev) for Tn in (2048, 4096)]
    ls = [(triton_launchers(X), fused_launcher(lib, X), fused_wu_launcher(lib, X)) for X in Xs]
    for (ch, co, both, kkt, wu), fu, fw in ls:      # warm (JIT / first-launch) outside the range
        kkt(); wu(); ch(); co(); fu(); fw()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for (ch, co, both, kkt, wu), fu, fw in ls:
        for _ in range(2):
            kkt(); wu(); ch(); co(); fu(); fw()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    return {"profiled_T": [2048, 4096], "launches_each": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.environ.get("QWEN36_CUDA_GDN_LIBRARY"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default="2048,4096,8192")
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    lib = GdnLib(args.lib)
    gen = torch.Generator().manual_seed(7)
    rep = {"lib": args.lib, "gpu": torch.cuda.get_device_name(0)}
    if args.profile:
        rep["profile"] = profile(lib, gen, dev)
    else:
        rep["kernels"] = []
        for Tn in [int(x) for x in args.tokens.split(",") if x]:
            r = bench_kernels(lib, Tn, args.pairs, args.iters, gen, dev)
            rep["kernels"].append(r)
            p, q = r["pair_triton_h+o_vs_fused_ho"], r["pair_triton_wu+h+o_vs_fused_wu"]
            print(f"bench T={Tn} h+o {p['before_median_ms']:.4f} -> ho {p['after_median_ms']:.4f} ms "
                  f"x{p['speedup_geomean']:.3f} ci{p['speedup_ci95']} | wu+h+o {q['before_median_ms']:.4f} -> "
                  f"wu {q['after_median_ms']:.4f} ms x{q['speedup_geomean']:.3f} ci{q['speedup_ci95']} "
                  f"alone={r['alone_ms']}", flush=True)
        r = bench_kernels(lib, 2048, args.pairs, args.iters, gen, dev, snapshot=True)
        rep["kernels"].append(r)
        p, q = r["pair_triton_h+o_vs_fused_ho"], r["pair_triton_wu+h+o_vs_fused_wu"]
        print(f"bench T=2048 snapshot h+o {p['before_median_ms']:.4f} -> {p['after_median_ms']:.4f} ms "
              f"x{p['speedup_geomean']:.3f} | wu+h+o {q['before_median_ms']:.4f} -> {q['after_median_ms']:.4f} ms "
              f"x{q['speedup_geomean']:.3f}", flush=True)
        rep["layer"] = []
        for Tn in (2048, 4096):
            for variant in ("wu", "ho"):
                r = bench_layer(lib, Tn, args.pairs, args.iters, gen, dev, variant)
                rep["layer"].append(r)
                p = r["pair_layer_prenormed_vs_fused"]
                print(f"bench layer T={Tn} {variant}: {p['before_median_ms']:.4f} ms -> {p['after_median_ms']:.4f} ms "
                      f"x{p['speedup_geomean']:.3f} ci{p['speedup_ci95']}", flush=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
