#!/usr/bin/env python3
"""Component speed: current FreeToken GDN path vs libqwen36_gdn (no profiler attached).

prefill: one GDN layer core (qkvz/ba GEMM outputs -> FP8 out_proj input), eager (the model runs
         prefill eagerly), T in --prefill-tokens, 5 interleaved pairs.
decode:  one token through 30 GDN layers (distinct states/weights per layer), eager and CUDA graph,
         5 interleaved pairs.
chunk_h: FLA chunk_gated_delta_rule_fwd_h with BV/num_warps variants -> bitwise vs the pinned
         32/4/2 config + time (a config change alone, if bit-identical).
kkt:     the 6 autotune configs of chunk_gated_delta_rule_fwd_kkt_solve_kernel -> bitwise vs the
         cached choice (BK=64, num_warps=1): documents the autotune-cache reproducibility risk.
Microbenchmarks are not model speed. Writes JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qwen36_gdn import GdnLib  # noqa: E402
import test_gdn_bitexact as T  # noqa: E402


def _jit(kernel):
    """The bare JITFunction under @triton.heuristics / @triton.autotune wrappers, so explicit
    meta-parameters (BK, BV, num_warps) are not supplied twice by the autotuner."""
    from triton.runtime.jit import JITFunction
    k = kernel
    while not isinstance(k, JITFunction):
        k = k.fn
    return k


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
    ratios = [a / b for a, b in out]
    lr = [math.log(r) for r in ratios]
    m = statistics.mean(lr)
    sd = statistics.stdev(lr) if len(lr) > 1 else 0.0
    t975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(len(lr), 2.776)
    half = t975 * sd / math.sqrt(len(lr)) if len(lr) > 1 else 0.0
    return {"orig_ms": [a for a, _ in out], "new_ms": [b for _, b in out],
            "speedup_geomean": math.exp(m), "speedup_ci95": [math.exp(m - half), math.exp(m + half)]}


def bench_prefill(lib, P, Tn, n_pairs, iters, dev, gen):
    qkvz, ba = T.make_inputs(P, [Tn], gen, dev, False)
    cu = torch.tensor([0, Tn], dtype=torch.int64, device=dev)
    idx = torch.tensor([1], dtype=torch.int32, device=dev)
    has = torch.tensor([True], device=dev)
    conv0 = (torch.randn(3, T.CONV_DIM, 3, generator=gen) * 0.8).to(torch.bfloat16).to(dev)
    rec0 = (torch.randn(3, T.HV, T.D, T.D, generator=gen) * 0.05).float().to(dev)
    cr, cc = conv0.clone(), conv0.clone()
    rr, rc = rec0.clone(), rec0.clone()
    fa = lambda: T.reference_prefill(P, qkvz, ba, cr, rr, cu, idx, has, None, None, None, diag=False)
    fb = lambda: T.candidate_prefill(lib, P, qkvz, ba, cc, rc, cu, idx, has, None, None, None, want_bf16=False)
    return pairs(fa, fb, n_pairs, iters)


def conv_tile_sweep(lib, P, Tn, dev, gen):
    """conv_qkv alone for tile_tokens x batch_rows variants (all bit-identical by construction:
    the tile only changes which block owns a token, never the arithmetic)."""
    qkvz, _ = T.make_inputs(P, [Tn], gen, dev, False)
    cu = torch.tensor([0, Tn], dtype=torch.int64, device=dev)
    idx = torch.tensor([1], dtype=torch.int32, device=dev)
    has = torch.tensor([True], device=dev)
    conv = (torch.randn(3, T.CONV_DIM, 3, generator=gen) * 0.8).to(torch.bfloat16).to(dev)
    outs = (torch.empty(Tn, 2048, dtype=torch.bfloat16, device=dev),
            torch.empty(Tn, 2048, dtype=torch.bfloat16, device=dev),
            torch.empty(Tn, 4096, dtype=torch.bfloat16, device=dev))
    res = []
    for tt in (16, 32, 64, 128):
        for br in (4, 8):
            f = lambda tt=tt, br=br: lib.conv_qkv(qkvz[:, :T.CONV_DIM], P["conv_w"], conv, cu, idx, has,
                                                  tile_tokens=tt, batch_rows=br, out=outs)
            ms = timeit(f, 30)
            gbs = (Tn * T.CONV_DIM * 2 * 2) / (ms * 1e-3) / 1e9
            res.append({"tile_tokens": tt, "batch_rows": br, "ms": round(ms, 5), "eff_GBps": round(gbs, 1)})
    return res


def build_decode_layers(nl, dev, gen):
    layers = []
    for _ in range(nl):
        P = T.synthetic_params(gen, dev)
        P["conv"] = (torch.randn(2, T.CONV_DIM, 3, generator=gen) * 0.8).to(torch.bfloat16).to(dev)
        P["rec"] = (torch.randn(2, T.HV, T.D, T.D, generator=gen) * 0.05).float().to(dev)
        P["qkvz"], P["ba"] = T.make_inputs(P, [1], gen, dev, False)
        layers.append(P)
    return layers


def bench_decode(lib, nl, n_pairs, iters, dev, gen, variant="cta"):
    layers = build_decode_layers(nl, dev, gen)
    idx = torch.tensor([1], dtype=torch.int32, device=dev)
    cu1 = torch.arange(2, dtype=torch.int32, device=dev)   # persistent like the graph buffer
    for P in layers:
        P["qkvz_ref"] = P["qkvz"].clone()
        P["out"] = {"fp8": torch.empty(1, 4096, dtype=torch.float8_e4m3fn, device=dev)}

    def orig():
        for P in layers:
            T.reference_decode(P, P["qkvz_ref"], P["ba"], P["conv"], P["rec"], idx, lean=True, cu=cu1)

    def new():
        for P in layers:
            lib.decode_fused(P["qkvz"], P["ba"], P["conv_w"], P["conv"], P["A_log"], P["dt_bias"],
                             P["rec"], idx, P["norm_w"], P["in_scale"], T.EPS, T.D ** -0.5, out=P["out"],
                             variant=variant)
    res = {"eager": pairs(orig, new, n_pairs, iters)}
    # CUDA graphs (decode runs in graphs in the server)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        orig(); new()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    go, gn = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(go):
        orig()
    with torch.cuda.graph(gn):
        new()
    res["graph"] = pairs(lambda: go.replay(), lambda: gn.replay(), n_pairs, iters)
    return res


def chunk_h_variants(dev, gen, Tn):
    """Bitwise + time of FLA chunk_h with other BV / num_warps / num_stages via the JIT fn."""
    import triton
    from freetoken.kernel.fla import chunk_delta_h as M
    from freetoken.kernel.fla.index import prepare_chunk_indices, prepare_chunk_offsets
    H, K, V = T.HV, T.D, T.D
    k = torch.nn.functional.normalize(torch.randn(1, Tn, 16, K, generator=gen), dim=-1).to(torch.bfloat16).to(dev)
    w = (torch.randn(1, Tn, H, K, generator=gen) * 0.1).to(torch.bfloat16).to(dev)
    u = (torch.randn(1, Tn, H, V, generator=gen) * 0.5).to(torch.bfloat16).to(dev)
    g = (-torch.rand(1, Tn, H, generator=gen) * 0.3).float().to(dev)
    cu = torch.tensor([0, Tn], dtype=torch.int64, device=dev)
    from freetoken.kernel.fla.cumsum import chunk_local_cumsum
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu)
    ci = prepare_chunk_indices(cu, 64)
    co = prepare_chunk_offsets(cu, 64)
    NT = len(ci)
    init = (torch.randn(3, H, V, K, generator=gen) * 0.05).float().to(dev)
    idx = torch.tensor([1], dtype=torch.int32, device=dev)

    def run(BV, nw, ns, st):
        h = k.new_empty(1, NT, H, V, K)
        vn = torch.empty_like(u)
        _jit(M.chunk_gated_delta_rule_fwd_kernel_h_blockdim64)[(triton.cdiv(V, BV), H)](
            k=k, v=u, w=w, v_new=vn, g=g, gk=None, h=h, initial_state=st, initial_state_indices=idx,
            snapshot_h_rows=None, snapshot_state_indices=None, cu_seqlens=cu, chunk_offsets=co, T=Tn,
            H=H, Hg=16, K=K, V=V, BT=64, BV=BV, USE_G=True, USE_GK=False, USE_INITIAL_STATE=True,
            INPLACE_UPDATE=True, SAVE_NEW_VALUE=True, IS_VARLEN=True, NT_BUCKET=1, NUM_SNAPSHOTS=0,
            num_warps=nw, num_stages=ns)
        return h, vn
    st0 = init.clone()
    h0, v0 = run(32, 4, 2, st0)
    out = []
    configs = [(bv, nw, 2) for bv in (16, 32, 64) for nw in (2, 4, 8)] + [(32, 4, 1), (32, 4, 3)]
    for BV, nw, ns in configs:   # 11 compiles: keeps the step well under its timeout
        try:
            st = init.clone()
            h, vn = run(BV, nw, ns, st)
            eq = bool(torch.equal(h.view(torch.int16), h0.view(torch.int16)) and
                      torch.equal(vn.view(torch.int16), v0.view(torch.int16)) and
                      torch.equal(st.view(torch.int32), st0.view(torch.int32)))
            def f(BV=BV, nw=nw, ns=ns):
                run(BV, nw, ns, init.clone())
            ms = timeit(f, 20)
            out.append({"BV": BV, "num_warps": nw, "num_stages": ns, "bit_identical": eq,
                        "ms": round(ms, 5)})
        except Exception as exc:
            out.append({"BV": BV, "num_warps": nw, "num_stages": ns, "error": repr(exc)[:160]})
    return out


def kkt_variants(dev, gen, Tn):
    import triton
    from freetoken.kernel.fla import chunk_fwd as M
    from freetoken.kernel.fla.index import prepare_chunk_indices
    from freetoken.kernel.fla.cumsum import chunk_local_cumsum
    H = T.HV
    k = torch.nn.functional.normalize(torch.randn(1, Tn, 16, T.D, generator=gen), dim=-1).to(torch.bfloat16).to(dev)
    beta = torch.rand(1, Tn, H, generator=gen).float().to(dev)
    cu = torch.tensor([0, Tn], dtype=torch.int64, device=dev)
    g = chunk_local_cumsum((-torch.rand(1, Tn, H, generator=gen) * 0.3).float().to(dev), chunk_size=64, cu_seqlens=cu)
    ci = prepare_chunk_indices(cu, 64)
    NT = len(ci)

    def run(BK, nw):
        A = torch.zeros(1, Tn, H, 64, device=dev, dtype=torch.bfloat16)
        _jit(M.chunk_gated_delta_rule_fwd_kkt_solve_kernel)[(NT, H)](
            k=k, g=g, beta=beta, A=A, cu_seqlens=cu, chunk_indices=ci, T=Tn, H=H, Hg=16, K=T.D,
            BT=64, BC=16, BK=BK, USE_G=True, IS_VARLEN=True, num_warps=nw, num_stages=3)
        return A
    A0 = run(64, 1)
    out = []
    for BK in (32, 64):
        for nw in (1, 2, 4):
            A = run(BK, nw)
            n = int((A.view(torch.int16) != A0.view(torch.int16)).sum().item())
            out.append({"BK": BK, "num_warps": nw, "mismatching_bf16": n, "numel": A.numel()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.environ.get("QWEN36_CUDA_GDN_LIBRARY"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefill-tokens", default="2048,4096")
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-chunkh-sweep", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(7)
    lib = GdnLib(args.lib)
    P = T.real_params(0, dev) if os.path.isdir(T.CKPT) else T.synthetic_params(gen, dev)
    rep = {"device": torch.cuda.get_device_name(), "library": lib.path, "prefill": {}, "notes":
           "component microbenchmark, no profiler; not model speed"}
    for tn in [int(x) for x in args.prefill_tokens.split(",")]:
        rep["prefill"][str(tn)] = bench_prefill(lib, P, tn, args.pairs, args.iters, dev, gen)
        print("prefill", tn, json.dumps(rep["prefill"][str(tn)]))
    rep["conv_tile_sweep_T4096"] = conv_tile_sweep(lib, P, 4096, dev, gen)
    print("conv sweep", json.dumps(rep["conv_tile_sweep_T4096"]))
    rep["decode_30_layers"] = bench_decode(lib, 30, args.pairs, max(args.iters, 50), dev, gen)
    print("decode", json.dumps(rep["decode_30_layers"]))
    rep["decode_30_layers_cluster"] = bench_decode(lib, 30, args.pairs, max(args.iters, 50), dev, gen,
                                                   variant="cluster")
    print("decode cluster", json.dumps(rep["decode_30_layers_cluster"]))
    if not args.no_chunkh_sweep:
        rep["chunk_h_variants_T4096"] = chunk_h_variants(dev, gen, 4096)
    rep["kkt_autotune_configs_T4096"] = kkt_variants(dev, gen, 4096)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    os.chmod(args.out, 0o600)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
