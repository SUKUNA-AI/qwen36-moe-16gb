#!/usr/bin/env python3
"""Bit-exactness of the fused FLA chunk_h + chunk_o CUDA kernel (libqwen36_gdn ABI 2) against the
Triton 3.6 FLA kernels it replaces.

Reference : chunk_gated_delta_rule_fwd_h (FP32 snapshot, in-place final state) followed by
            chunk_fwd_kernel_o on its bf16 h / v_new -- exactly what chunk_prenormed launches.
Candidate : GdnLib.fla_chunk_ho on the SAME w, u (FLA kkt_solve + recompute_w_u output).
Compared bit for bit (raw bits: -0.0 vs +0.0 and NaN payloads count): o and the whole FP32
state pool (final states, snapshot slots and every untouched slot).

Inputs: q/k/v/g/beta from libqwen36_gdn conv_qkv + gate_cumsum (bit-exact with production) on
synthetic GEMM outputs, and on real-checkpoint activations for layers 0,1,2,13,26,38, plus stress
distributions. GPU required; run under the GPU lock. Writes a JSON report (no tensors).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from qwen36_gdn import GdnLib  # noqa: E402
import test_gdn_bitexact as TB  # noqa: E402

HK, HV, D = 16, 32, 128


def bits(t):
    t = t.contiguous()
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


def neq(a, b):
    a_, b_ = bits(a), bits(b)
    return int((a_ != b_).sum().item())


def make_qkvgb(lib, P, lens, gen, dev, real, stress=None):
    """q,k [1,T,16,128] (L2-normed), v [1,T,32,128], g [1,T,32] (chunk-local cumsum), beta."""
    T = sum(lens)
    qkvz, ba = TB.make_inputs(P, lens, gen, dev, real)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int64, device=dev)
    B = len(lens)
    S = 2 * B + 3
    conv = (torch.randn(S, TB.CONV_DIM, 3, generator=gen) * 0.8).to(torch.bfloat16).to(dev)
    idx = torch.arange(B, dtype=torch.int32, device=dev) + 1
    has_init = torch.ones(B, dtype=torch.bool, device=dev)
    q, k, v = lib.conv_qkv(qkvz[:, :TB.CONV_DIM], P["conv_w"], conv, cu, idx, has_init)
    if stress == "big_v":
        v = (v.float() * 64.0).to(torch.bfloat16)
    if stress == "raw_qk":       # un-normalised q/k: larger HMMA magnitudes / cancellation
        q = (torch.randn(T, HK * D, generator=gen) * 3.0).to(torch.bfloat16).to(dev)
        k = (torch.randn(T, HK * D, generator=gen) * 3.0).to(torch.bfloat16).to(dev)
    A_log, dt_bias = P["A_log"], P["dt_bias"]
    if stress == "strong_decay":
        A_log = A_log + 3.0
    if stress == "weak_decay":
        A_log = A_log - 8.0
    g, beta = lib.gate_cumsum(ba, A_log.contiguous(), dt_bias, cu)
    return (q.view(1, T, HK, D), k.view(1, T, HK, D), v.view(1, T, HV, D), g.view(1, T, HV),
            beta.view(1, T, HV), cu, S)


def reference(q, k, w, u, g, state, idx32, cu, ci, scale, rows, dst):
    import triton
    from freetoken.kernel.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from freetoken.kernel.fla.chunk_o import chunk_fwd_kernel_o
    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=state, initial_state_indices=idx32, cu_seqlens=cu,
        chunk_indices=ci, snapshot_h_rows=rows, snapshot_state_indices=dst)
    T = q.shape[1]
    BT = min(64, max(16, triton.next_power_of_2(T)))
    assert BT == 64
    o = torch.empty_like(v_new)
    grid = (2, len(ci), HV)
    chunk_fwd_kernel_o[grid](q, k, v_new, h, g, o, cu, ci, scale, T=T, H=HV, Hg=HK, K=D, V=D, BT=64,
                             BK=128, BV=64, USE_G=True, IS_VARLEN=True, num_warps=4, num_stages=2)
    return o


def run_case(lib, P, lens, gen, dev, real, results, cont=True, snap="auto", perm=False, stress=None,
             negzero=False, tag_extra=""):
    from freetoken.kernel.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from freetoken.kernel.fla.index import prepare_chunk_indices, prepare_chunk_offsets
    q, k, v, g, beta, cu, S = make_qkvgb(lib, P, lens, gen, dev, real, stress)
    B = len(lens)
    S = S + B
    idx = torch.arange(B, dtype=torch.int32, device=dev) + 1
    if perm:
        idx = (torch.randperm(S - 1, generator=gen)[:B] + 1).to(torch.int32).to(dev)
    state0 = (torch.randn(S, HV, D, D, generator=gen) * (0.05 if stress != "big_state" else 40.0)).float().to(dev)
    if negzero:   # exercise the 0.0 + x initial-state conversion (-0.0 -> +0.0)
        m = torch.rand(S, HV, D, D, generator=gen).to(dev) < 0.05
        state0 = torch.where(m, torch.full_like(state0, -0.0), state0)
    if not cont:
        state0[idx.long()] = 0.0
    ci = prepare_chunk_indices(cu, 64)
    co = prepare_chunk_offsets(cu, 64)
    rows = dst = None
    if snap:
        boh = co.tolist()
        used = set(idx.tolist())
        free = [s for s in range(S) if s not in used]
        rl, dl = [], []
        for i, L in enumerate(lens):
            nt = (L + 63) // 64
            if nt == 0:
                continue
            if snap == "auto":
                c = (L - 8) // 64
                if c < 1:
                    continue
            elif snap == "first":
                c = 0
            elif snap == "last":
                c = nt - 1
            else:   # random chunk, some sequences without snapshot
                if torch.rand(1, generator=gen).item() < 0.3:
                    continue
                c = int(torch.randint(0, nt, (1,), generator=gen).item())
            rl.append(boh[i] + c)
            dl.append(free.pop(int(torch.randint(0, len(free), (1,), generator=gen).item())))
        if rl:
            rows = torch.tensor(rl, dtype=torch.int64, device=dev)
            dst = torch.tensor(dl, dtype=torch.int64, device=dev)
    from freetoken.kernel.fla.wy_fast import recompute_w_u_fwd
    w, u, A_i = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta, cu_seqlens=cu, chunk_indices=ci)
    A = lib.kkt_solve(k, g, beta, cu, ci)               # the wrapper used by the "wu" variant
    w2, u2 = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=g, cu_seqlens=cu, chunk_indices=ci)
    wrapper_same = neq(A, A_i) == 0 and neq(w2, w) == 0 and neq(u2, u) == 0
    scale = D ** -0.5
    st_r, st_c, st_w = state0.clone(), state0.clone(), state0.clone()
    o_r = reference(q, k, w, u, g, st_r, idx, cu, ci, scale, rows, dst)
    kw = dict(chunk_offsets=co if rows is not None else None, snap_rows_i64=rows, snap_dst_i64=dst)
    o_c = lib.fla_chunk_ho(q, k, w, u, g, st_c, idx, cu, scale, **kw)
    o_w = lib.fla_wu_chunk_ho(q, k, v, A, beta, g, st_w, idx, cu, scale, **kw)
    torch.cuda.synchronize()
    n_o, n_s = neq(o_r, o_c), neq(st_r, st_c)
    n_ow, n_sw = neq(o_r, o_w), neq(st_r, st_w)
    empty = 0 in lens
    extra = {"wu_o_mismatch": n_ow, "wu_state_mismatch": n_sw, "kkt_wrapper_identical": wrapper_same}
    if empty:
        # FLA prepare_chunk_indices mislabels every chunk after an empty sequence: kkt_solve /
        # recompute_w_u / chunk_o work on the wrong sequence and leave rows of their torch.empty
        # outputs unwritten, so the reference o (and any w/u-derived value: the "wu" variant and
        # the kkt-wrapper identity) is undefined there; chunk_h itself is per sequence, so the "ho"
        # kernel fed the same w/u must still match the reference state. The production wrapper
        # must fall back (and the patch keeps such batches on the original path).
        st_f = state0.clone()
        fb = lib.chunk_fused(q, k, v, g, beta, st_f, idx, cu, scale, rows, dst)
        torch.cuda.synchronize()
        extra.update({"o_compared": False, "wu_compared": False, "chunk_fused_fell_back": fb is None,
                      "fallback_left_state_untouched": neq(st_f, state0) == 0,
                      "note": "reference o/w/u undefined after an empty sequence (FLA chunk index bug)"})
        ok = n_s == 0 and fb is None and extra["fallback_left_state_untouched"]
    else:
        ok = n_o == 0 and n_s == 0 and n_ow == 0 and n_sw == 0 and wrapper_same
    tag = (f"{'real' if real else 'syn'} lens={lens if len(lens) <= 8 else str(lens[:4])+'..x'+str(len(lens))}"
           f" cont={cont} snap={snap}:{0 if rows is None else rows.numel()} perm={perm}"
           f" stress={stress} negzero={negzero}{tag_extra}")
    results.append({"case": tag, "pass": ok, "o_mismatch": n_o, "o_numel": o_r.numel(),
                    "state_mismatch": n_s, "state_numel": st_r.numel(), "T": sum(lens), **extra})
    msg = ("PASS " if ok else "FAIL ") + tag
    if not ok:
        # first mismatching coordinates (indices only)
        bo = (bits(o_r) != bits(o_c)).nonzero()[:4].tolist()
        bs = (bits(st_r) != bits(st_c)).nonzero()[:4].tolist()
        bow = (bits(o_r) != bits(o_w)).nonzero()[:4].tolist()
        bsw = (bits(st_r) != bits(st_w)).nonzero()[:4].tolist()
        msg += (f" ho: o_mis={n_o} st_mis={n_s} o_at={bo} st_at={bs} | wu: o_mis={n_ow} st_mis={n_sw}"
                f" o_at={bow} st_at={bsw} | kkt_wrapper_identical={wrapper_same}")
    print(msg, flush=True)
    return ok


def candidate_prefill_fused(lib, P, qkvz, ba, conv_states, rec_states, cu, idx, has_init, fresh,
                            snap_rows, snap_dst, variant="wu"):
    """The patched gdn.py prefill branch with QWEN36_CUDA_GDN_FLA=1: chunk_fused, falling back to
    chunk_prenormed (Triton chunk_h/chunk_o) when unsupported; batches with an empty sequence
    take the original gdn.py path (the patch checks this before touching any state)."""
    from qwen36_gdn import has_empty_sequence
    if has_empty_sequence(cu):
        r = TB.reference_prefill(P, qkvz, ba, conv_states, rec_states, cu, idx, has_init, fresh,
                                 snap_rows, snap_dst, diag=False)
        return {"o": r["o"], "y": r["y"], "qa": r["qa"]}, "original"
    total = qkvz.shape[0]
    conv_in = qkvz[:, :TB.CONV_DIM]
    z = qkvz[:, TB.CONV_DIM:]
    q, k, v = lib.conv_qkv(conv_in, P["conv_w"], conv_states, cu, idx, has_init)
    g, beta = lib.gate_cumsum(ba, P["A_log"], P["dt_bias"], cu)
    if fresh is not None:
        rec_states.index_fill_(0, fresh, 0.0)
    a = (q.view(1, total, HK, D), k.view(1, total, HK, D), v.view(1, total, HV, D), g.view(1, total, HV),
         beta.view(1, total, HV), rec_states, idx, cu, D ** -0.5)
    core = lib.chunk_fused(*a, snapshot_h_rows=snap_rows, snapshot_state_indices=snap_dst, variant=variant)
    path = "fused-" + variant if core is not None else "prenormed"
    if core is None:
        core, _h = lib.chunk_prenormed(*a, snapshot_h_rows=snap_rows, snapshot_state_indices=snap_dst)
    y, qa = lib.gated_norm(core, z, P["norm_w"], TB.EPS, in_scale=P["in_scale"], want_bf16=True)
    return {"o": core.contiguous(), "y": y, "qa": qa}, path


def run_e2e_case(lib, P, lens, cont, gen, dev, real, results, tag_extra="", variant="wu"):
    """Original gdn.py prefill (sgl conv, FLA chunk_gated_delta_rule incl. l2norm/cumsum, FP32
    snapshot, rms_norm_gated, _static_quant) vs the patched CUDA path with the fused FLA kernel."""
    from freetoken.kernel.fla.index import prepare_chunk_offsets
    B = len(lens)
    S = B * 2 + 3
    qkvz, ba = TB.make_inputs(P, lens, gen, dev, real)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int64, device=dev)
    idx = torch.arange(B, dtype=torch.int32, device=dev) + 1
    has_init = torch.tensor([cont] * B, dtype=torch.bool, device=dev)
    fresh = None if cont else idx.to(torch.int64)
    conv0 = (torch.randn(S, TB.CONV_DIM, 3, generator=gen) * 0.8).to(torch.bfloat16).to(dev)
    rec0 = (torch.randn(S, HV, D, D, generator=gen) * 0.05).float().to(dev)
    boh = prepare_chunk_offsets(cu, 64).tolist()
    rl, dl = [], []
    for i, L in enumerate(lens):
        c = (L - 8) // 64
        if c >= 1:
            rl.append(boh[i] + c)
            dl.append(B + 1 + i)
    snap_rows = torch.tensor(rl, dtype=torch.int64, device=dev) if rl else None
    snap_dst = torch.tensor(dl, dtype=torch.int64, device=dev) if rl else None
    conv_r, conv_c = conv0.clone(), conv0.clone()
    rec_r, rec_c = rec0.clone(), rec0.clone()
    qkvz_c = qkvz.clone()
    ref = TB.reference_prefill(P, qkvz, ba, conv_r, rec_r, cu, idx, has_init, fresh, snap_rows, snap_dst,
                               diag=False)
    cand, path = candidate_prefill_fused(lib, P, qkvz_c, ba, conv_c, rec_c, cu, idx, has_init, fresh,
                                         snap_rows, snap_dst, variant=variant)
    torch.cuda.synchronize()
    chk = {"o": neq(ref["o"], cand["o"]), "y": neq(ref["y"], cand["y"]),
           "qa": int((ref["qa"].view(torch.uint8) != cand["qa"].view(torch.uint8)).sum().item()),
           "conv_states": neq(conv_r.view(torch.bfloat16), conv_c.view(torch.bfloat16)),
           "recurrent_states": neq(rec_r, rec_c)}
    ok = all(v == 0 for v in chk.values())
    if path == "original":
        # batch with an empty sequence: the patched gdn.py runs the ORIGINAL path, which reads w/u
        # rows FLA never wrote (torch.empty, chunk index bug), so two original-path runs need not
        # be bit-identical. This case only checks that the fallback is taken and records the
        # run-to-run difference of the original path itself as evidence for that fallback.
        chk = {"original_path_run_to_run_mismatch": chk}
        ok = True
    tag = f"e2e {'real' if real else 'syn'} lens={lens} cont={cont} path={path}{tag_extra}"
    results.append({"case": tag, "pass": ok, "path": path, "mismatches": chk, "T": sum(lens),
                    "snapshots": len(rl)})
    print(("PASS " if ok else "FAIL ") + tag + ("" if ok else f" {chk}"), flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.environ.get("QWEN36_CUDA_GDN_LIBRARY"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0,1,2,13,26,38")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    lib = GdnLib(args.lib)
    import triton
    import freetoken
    results = []
    t0 = time.time()
    gen = torch.Generator().manual_seed(20261014)
    P = TB.synthetic_params(gen, dev)
    ok = True
    singles = [33, 40, 63, 64, 65, 127, 128, 129, 200, 511, 1000, 2048, 2049, 4096, 8192]
    if args.quick:
        singles = [33, 64, 65, 129, 1000, 2048, 4096]
    for T in singles:
        ok &= run_case(lib, P, [T], gen, dev, False, results)
        ok &= run_case(lib, P, [T], gen, dev, False, results, cont=False, snap="first")
    varlens = [[1, 63, 64, 65, 130], [2048, 1, 33], [7, 3000, 5], [64] * 8, [100] * 20, [1] * 40,
               [17, 0, 300, 0, 5], [4096, 4096], [8192, 3, 61, 1500]]
    if args.quick:
        varlens = varlens[:5]
    for lens in varlens:
        ok &= run_case(lib, P, lens, gen, dev, False, results)
        ok &= run_case(lib, P, lens, gen, dev, False, results, cont=False, snap="random", perm=True)
        ok &= run_case(lib, P, lens, gen, dev, False, results, snap="last", negzero=True)
    for stress in ["big_v", "raw_qk", "strong_decay", "weak_decay", "big_state"]:
        ok &= run_case(lib, P, [1000, 37], gen, dev, False, results, stress=stress, snap="random")
        ok &= run_case(lib, P, [2048], gen, dev, False, results, stress=stress)
    e2e_syn = [([2048], True), ([33], False), ([1, 63, 64, 65, 130], True), ([5, 7], True), ([1], False),
               ([4096], False), ([3, 0, 90], True)]
    if args.quick:
        e2e_syn = e2e_syn[:5]
    for lens, cont in e2e_syn:
        for var in ("wu", "ho"):
            ok &= run_e2e_case(lib, P, lens, cont, gen, dev, False, results, variant=var)
    for layer in [int(x) for x in args.layers.split(",") if x]:
        Pr = TB.real_params(layer, dev)
        ok &= run_case(lib, Pr, [2048], gen, dev, True, results, tag_extra=f" layer={layer}")
        ok &= run_case(lib, Pr, [700, 1, 1348], gen, dev, True, results, cont=False, snap="random",
                       perm=True, tag_extra=f" layer={layer}")
        for var in ("wu", "ho"):
            ok &= run_e2e_case(lib, Pr, [2048], True, gen, dev, True, results, tag_extra=f" layer={layer}", variant=var)
            ok &= run_e2e_case(lib, Pr, [700, 1, 1348], False, gen, dev, True, results, tag_extra=f" layer={layer}",
                               variant=var)
        if not args.quick:
            ok &= run_case(lib, Pr, [8192], gen, dev, True, results, snap="last", tag_extra=f" layer={layer}")
        del Pr
        torch.cuda.empty_cache()
    from qwen36_gdn import ABI_VERSION
    from triton.backends.nvidia.compiler import get_ptxas
    pt = get_ptxas(120)
    rep = {"pass": bool(ok), "cases": len(results), "failed": sum(not r["pass"] for r in results),
           "seconds": round(time.time() - t0, 1), "abi": ABI_VERSION, "lib": args.lib,
           "triton": triton.__version__, "torch": torch.__version__,
           "freetoken_tree": os.path.dirname(freetoken.__file__),
           "triton_ptxas_sm120": {"path": pt.path, "version": pt.version},
           "triton_ptxas_blackwell_env": os.environ.get("TRITON_PTXAS_BLACKWELL_PATH", "<unset>"),
           "results": results}
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"{'ALL PASS' if ok else 'FAILURES'}: {rep['cases'] - rep['failed']}/{rep['cases']} "
          f"in {rep['seconds']} s", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
