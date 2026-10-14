#!/usr/bin/env python3
"""Bit-exactness tests: libqwen36_gdn kernels vs the CURRENT FreeToken GDN path.

Reference = the exact operations of models/qwen3_5_moe/gdn.py (native tree on PYTHONPATH):
sgl_kernel causal_conv1d_fwd on the transposed copy, FLA chunk_gated_delta_rule (l2norm,
cumsum, kkt_solve, wy, chunk_h with FP32 snapshot, chunk_o), torch gating ops,
rms_norm_gated, FreeToken _static_quant. Candidate = libqwen36_gdn conv/gate/norm kernels
feeding the SAME FLA chunk kernels. Every intermediate and every output is compared bit for bit
(torch.equal on the raw bits, so -0.0 vs +0.0 and NaN payloads count as mismatches).

GPU required; run under the GPU lock. Writes a JSON report (no tensors, no text from data).
"""
from __future__ import annotations

import hashlib
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from qwen36_gdn import GdnLib  # noqa: E402

KEY_DIM, VALUE_DIM, HK, HV, D = 2048, 4096, 16, 32, 128
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
QKVZ = CONV_DIM + VALUE_DIM
EPS = 1e-6
CKPT = os.environ.get("QWEN36_MODEL_DIR", "/path/to/model")  # NVFP4 checkpoint directory
FLA = False        # --fla: candidate prefill uses the fused FLA chunk_h + chunk_o kernel (ABI 2)
FLA_USED = []      # token counts of the prefill calls that actually ran the fused kernel
FLA_VARIANT = "wu" # --fla-variant: wu (recompute_w_u folded in) | ho (chunk_h + chunk_o only)


def bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    if t.dtype in (torch.float32,):
        return t.view(torch.int32)
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16)
    if t.dtype in (torch.float8_e4m3fn,):
        return t.view(torch.uint8)
    return t


def cmp(name, a, b, results):
    a_, b_ = bits(a), bits(b)
    same = a_.shape == b_.shape and torch.equal(a_, b_)
    n = int((a_ != b_).sum().item()) if a_.shape == b_.shape else -1
    results.append({"name": name, "equal": bool(same), "mismatches": n, "numel": int(a_.numel())})
    return same


# ------------------------------------------------------------------------------------------
# Reference path (verbatim op sequence of the current gdn.py prefill branch)
# ------------------------------------------------------------------------------------------
def reference_prefill(P, qkvz, ba, conv_states, rec_states, cu, idx, has_init, fresh, snap_rows, snap_dst,
                      diag=True):
    from freetoken.kernel.causal_conv1d import causal_conv1d_varlen
    from freetoken.kernel.fla import chunk_gated_delta_rule, rms_norm_gated
    from freetoken.kernel.fla.l2norm import l2norm_fwd
    from freetoken.kernel.fla.cumsum import chunk_local_cumsum
    from freetoken.kernel.triton.fp8_pertensor_linear import _static_quant

    total = qkvz.shape[0]
    conv_in, z = torch.split(qkvz, [CONV_DIM, VALUE_DIM], dim=-1)
    b, a = torch.split(ba, [HV, HV], dim=-1)
    z = z.reshape(total, HV, D)
    x = conv_in.transpose(0, 1).contiguous()
    out = causal_conv1d_varlen(x, P["conv_w"], conv_states, cu, idx, has_init)
    mixed = out.transpose(0, 1)
    qf, kf, vf = torch.split(mixed, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    q = qf.reshape(1, total, HK, D).to(torch.bfloat16)
    k = kf.reshape(1, total, HK, D).to(torch.bfloat16)
    v = vf.reshape(1, total, HV, D).to(torch.bfloat16)
    beta = b.sigmoid()
    g = -P["A_log"].exp() * F.softplus(a.float() + P["dt_bias"])
    g = g.reshape(1, total, HV)
    beta = beta.float().reshape(1, total, HV)
    if fresh is not None:
        rec_states.index_fill_(0, fresh, 0.0)
    # diagnostics: the intermediates the fused kernels must reproduce (extra kernels: off for timing)
    inter = {} if not diag else {
        "q_l2": l2norm_fwd(q.contiguous()).reshape(total, KEY_DIM),
        "k_l2": l2norm_fwd(k.contiguous()).reshape(total, KEY_DIM),
        "v": v.reshape(total, VALUE_DIM).contiguous(),
        "g_cumsum": chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu).reshape(total, HV),
        "beta": beta.reshape(total, HV).contiguous(),
    }
    o, _, h = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, scale=D ** -0.5,
        initial_state=rec_states, initial_state_indices=idx.to(torch.int32),
        cu_seqlens=cu.to(torch.int64), head_first=False, use_qk_l2norm_in_kernel=True,
        snapshot_h_rows=snap_rows, snapshot_state_indices=snap_dst)
    core = o[0].reshape(-1, D)
    zz = z.reshape(-1, D)
    y = rms_norm_gated(x=core, weight=P["norm_w"], bias=None, z=zz, eps=EPS, is_rms_norm=True,
                       norm_before_gate=True, activation="silu").reshape(total, -1)
    qa = _static_quant(y, P["in_scale"])
    return {"o": o[0].contiguous(), "h": h, "y": y, "qa": qa, **inter}


def candidate_prefill(lib, P, qkvz, ba, conv_states, rec_states, cu, idx, has_init, fresh,
                      snap_rows, snap_dst, tile=16, rows=4, want_bf16=True):
    """Exactly the patched gdn.py prefill branch (_forward_cuda) up to the out_proj FP8 input."""
    total = qkvz.shape[0]
    conv_in = qkvz[:, :CONV_DIM]
    z = qkvz[:, CONV_DIM:]
    q, k, v = lib.conv_qkv(conv_in, P["conv_w"], conv_states, cu, idx, has_init,
                           tile_tokens=tile, batch_rows=rows)
    g, beta = lib.gate_cumsum(ba, P["A_log"], P["dt_bias"], cu)
    if fresh is not None:
        rec_states.index_fill_(0, fresh, 0.0)
    args = (q.view(1, total, HK, D), k.view(1, total, HK, D), v.view(1, total, HV, D),
            g.view(1, total, HV), beta.view(1, total, HV), rec_states, idx, cu, D ** -0.5)
    core = h = None
    if FLA:   # QWEN36_CUDA_GDN_FLA=1 branch: fused chunk_h + chunk_o, Triton fallback if unsupported
        core = lib.chunk_fused(*args, snapshot_h_rows=snap_rows, snapshot_state_indices=snap_dst,
                               variant=FLA_VARIANT)
        if core is not None:
            FLA_USED.append(total)
    if core is None:
        core, h = lib.chunk_prenormed(*args, snapshot_h_rows=snap_rows, snapshot_state_indices=snap_dst)
    y, qa = lib.gated_norm(core, z, P["norm_w"], EPS, in_scale=P["in_scale"], want_bf16=want_bf16)
    return {"o": core.contiguous(), "h": h, "y": y, "qa": qa, "q_l2": q, "k_l2": k, "v": v,
            "g_cumsum": g, "beta": beta}


# ------------------------------------------------------------------------------------------
def synthetic_params(gen, device):
    P = {
        "conv_w": (torch.randn(CONV_DIM, 4, generator=gen) * 0.35).to(torch.bfloat16).to(device),
        "A_log": (torch.rand(HV, generator=gen) * 3.0 - 1.0).float().to(device),
        "dt_bias": (torch.randn(HV, generator=gen) * 1.5).float().to(device),
        "norm_w": (1.0 + torch.randn(D, generator=gen) * 0.2).to(torch.bfloat16).to(device),
        "in_scale": torch.tensor(0.0125, dtype=torch.float32, device=device),
    }
    return P


def real_params(layer, device):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    pre = f"model.language_model.layers.{layer}.linear_attn."
    need = {"conv_w": "conv1d.weight", "A_log": "A_log", "dt_bias": "dt_bias", "norm_w": "norm.weight",
            "in_scale": "out_proj.input_scale", "wqkv": "in_proj_qkv.weight", "sqkv": "in_proj_qkv.weight_scale",
            "wz": "in_proj_z.weight", "sz": "in_proj_z.weight_scale", "wa": "in_proj_a.weight",
            "wb": "in_proj_b.weight"}
    out = {}
    handles = {}
    for key, name in need.items():
        f = idx[pre + name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(CKPT, f), framework="pt", device="cpu")
        out[key] = handles[f].get_tensor(pre + name)
    P = {
        "conv_w": out["conv_w"].reshape(CONV_DIM, 4).to(torch.bfloat16).contiguous().to(device),
        "A_log": out["A_log"].float().contiguous().to(device),
        "dt_bias": out["dt_bias"].float().contiguous().to(device),
        "norm_w": out["norm_w"].to(torch.bfloat16).contiguous().to(device),
        "in_scale": out["in_scale"].float().reshape(()).contiguous().to(device),
    }
    # projection weights only to make realistic activations (dequantised, any GEMM numerics ok)
    wq = out["wqkv"].to(device).float() * out["sqkv"].to(device).float().reshape(-1, 1)
    wz = out["wz"].to(device).float() * out["sz"].to(device).float().reshape(-1, 1)
    P["w_qkvz"] = torch.cat([wq, wz], 0).to(torch.bfloat16)
    P["w_ba"] = torch.cat([out["wb"].to(device), out["wa"].to(device)], 0).to(torch.bfloat16)
    return P


def make_inputs(P, lens, gen, device, real):
    T = sum(lens)
    if real:
        hid = torch.randn(T, 2048, generator=gen).to(device).to(torch.bfloat16)
        hid = (hid.float() / hid.float().pow(2).mean(-1, keepdim=True).sqrt()).to(torch.bfloat16)
        qkvz = (hid @ P["w_qkvz"].t()).to(torch.bfloat16)
        ba = (hid @ P["w_ba"].t()).to(torch.bfloat16)
    else:
        qkvz = (torch.randn(T, QKVZ, generator=gen) * 1.3).to(torch.bfloat16).to(device)
        ba = (torch.randn(T, 2 * HV, generator=gen) * 2.0).to(torch.bfloat16).to(device)
    return qkvz.contiguous(), ba.contiguous()


def run_case(lib, P, lens, cont, gen, device, real, results, tile=16, rows=4, snapshot=True):
    B = len(lens)
    S = B * 2 + 3   # live slots + snapshot slots
    qkvz, ba = make_inputs(P, lens, gen, device, real)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int64, device=device)
    idx = torch.arange(B, dtype=torch.int32, device=device) + 1
    has_init = torch.tensor([cont] * B, dtype=torch.bool, device=device)
    fresh = None if cont else (idx.to(torch.int64))
    conv0 = (torch.randn(S, CONV_DIM, 3, generator=gen) * 0.8).to(torch.bfloat16).to(device)
    rec0 = (torch.randn(S, HV, D, D, generator=gen) * 0.05).float().to(device)
    snap_rows = snap_dst = None
    if snapshot:
        from freetoken.kernel.fla.index import prepare_chunk_offsets
        boh = prepare_chunk_offsets(cu, 64).tolist()
        rows_l, dst_l = [], []
        for i, L in enumerate(lens):
            c = (L - 8) // 64
            if c >= 1:
                rows_l.append(boh[i] + c)
                dst_l.append(B + 1 + i)
        if rows_l:
            snap_rows = torch.tensor(rows_l, dtype=torch.int64, device=device)
            snap_dst = torch.tensor(dst_l, dtype=torch.int64, device=device)
    conv_r, conv_c = conv0.clone(), conv0.clone()
    rec_r, rec_c = rec0.clone(), rec0.clone()
    # Each path gets its own copy of the GEMM outputs: for a 1-token prefill the original
    # transpose(0,1).contiguous() is a no-op view (size-1 dim), so sgl_kernel writes its conv output
    # into qkvz[:, :conv_dim] in place. Nothing reads those bytes afterwards in gdn.py, but a shared
    # buffer would feed the candidate the reference's output instead of the GEMM output.
    qkvz_c = qkvz.clone()
    qkvz_before = qkvz.clone()
    ref = reference_prefill(P, qkvz, ba, conv_r, rec_r, cu, idx, has_init, fresh, snap_rows, snap_dst)
    ref_modified_input = not torch.equal(qkvz.view(torch.int16), qkvz_before.view(torch.int16))
    cand = candidate_prefill(lib, P, qkvz_c, ba, conv_c, rec_c, cu, idx, has_init, fresh, snap_rows,
                             snap_dst, tile=tile, rows=rows)
    torch.cuda.synchronize()
    cand_modified_input = not torch.equal(qkvz_c.view(torch.int16), qkvz_before.view(torch.int16))
    tag = f"{'real' if real else 'syn'} lens={lens} cont={cont} tile={tile}x{rows}"
    sub = []
    sub.append({"name": "candidate_leaves_qkvz_untouched", "equal": not cand_modified_input,
                "mismatches": int(cand_modified_input), "numel": 1,
                "note": "reference wrote its conv output into qkvz in place" if ref_modified_input else ""})
    for key in ("q_l2", "k_l2", "v", "g_cumsum", "beta", "o", "h", "y", "qa"):
        if key == "h" and cand["h"] is None:
            # fused FLA path: the bf16 h tensor is never materialised (only the BF16 track
            # snapshot needs it, and that configuration keeps the Triton kernels)
            sub.append({"name": "h", "equal": True, "mismatches": 0, "numel": 0,
                        "note": "not materialised by the fused chunk_h+chunk_o kernel"})
            continue
        cmp(key, ref[key], cand[key], sub)
    cmp("conv_states", conv_r, conv_c, sub)
    cmp("recurrent_states(final+snapshots)", rec_r, rec_c, sub)
    ok = all(r["equal"] for r in sub)
    results.append({"case": tag, "pass": ok, "checks": sub})
    print(("PASS " if ok else "FAIL ") + tag + ("" if ok else " " + json.dumps([r for r in sub if not r["equal"]])))
    return ok


# ------------------------------------------------------------------------------------------
# Decode: reference = conv update (sgl) + fused sigmoid-gating recurrence (FLA Triton) +
# rms_norm_gated + _static_quant, exactly as gdn.py's decode branch + out_proj quantisation.
# ------------------------------------------------------------------------------------------
def reference_decode(P, qkvz, ba, conv_states, rec, idx, lean=False, cu=None):
    from freetoken.kernel.causal_conv1d import causal_conv1d_decode
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla
    from freetoken.kernel.fla import rms_norm_gated
    from freetoken.kernel.triton.fp8_pertensor_linear import _static_quant
    B = qkvz.shape[0]
    conv_in, z = torch.split(qkvz, [CONV_DIM, VALUE_DIM], dim=-1)
    b, a = torch.split(ba, [HV, HV], dim=-1)
    z = z.reshape(B, HV, D)
    mixed = causal_conv1d_decode(conv_in, conv_states, P["conv_w"], idx)
    qf, kf, vf = torch.split(mixed, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    q = qf.reshape(1, B, HK, D).to(torch.bfloat16)
    k = kf.reshape(1, B, HK, D).to(torch.bfloat16)
    v = vf.reshape(1, B, HV, D).to(torch.bfloat16)
    if cu is None:  # the server keeps this arange in a persistent graph buffer
        cu = torch.arange(B + 1, dtype=torch.int32, device=qkvz.device)
    core = gdn_decode_fla(q, k, v, a, b, A_log=P["A_log"], dt_bias=P["dt_bias"], state_source=rec,
                          indices=idx, cu_seqlens=cu, scale=D ** -0.5)
    core2 = core.reshape(-1, D)
    zz = z.reshape(-1, D)
    y = rms_norm_gated(x=core2, weight=P["norm_w"], bias=None, z=zz, eps=EPS, is_rms_norm=True,
                       norm_before_gate=True, activation="silu").reshape(B, -1)
    qa = _static_quant(y, P["in_scale"])
    if lean:
        return qa
    return {"mixed": mixed.contiguous().clone(), "core": core2.reshape(B, HV, D).clone(), "y": y, "qa": qa}


def run_decode_case(lib, P, B, steps, gen, dev, real, results, pad_one=False, variant="cta"):
    S = B + 3
    conv_r = (torch.randn(S, CONV_DIM, 3, generator=gen) * 0.8).to(torch.bfloat16).to(dev)
    rec_r = (torch.randn(S, HV, D, D, generator=gen) * 0.05).float().to(dev)
    conv_c, rec_c = conv_r.clone(), rec_r.clone()
    idx = (torch.arange(B, dtype=torch.int32) + 1).to(dev)
    if pad_one and B > 1:
        idx[B - 1] = -1
    tag = f"decode[{variant}] {'real' if real else 'syn'} B={B} steps={steps}{' pad' if pad_one else ''}"
    ok = True
    sub_all = []
    for s in range(steps):
        qkvz, ba = make_inputs(P, [B], gen, dev, real)
        qkvz_c = qkvz.clone()
        ref = reference_decode(P, qkvz, ba, conv_r, rec_r, idx)
        cand = lib.decode_fused(qkvz_c, ba, P["conv_w"], conv_c, P["A_log"], P["dt_bias"], rec_c, idx,
                                P["norm_w"], P["in_scale"], EPS, D ** -0.5, want_fp8=True,
                                want_bf16=True, want_core=True, want_mixed=True, variant=variant)
        torch.cuda.synchronize()
        sub = []
        if pad_one and B > 1:   # padded rows: compare only live rows of the outputs
            live = slice(0, B - 1)
            cmp("mixed", ref["mixed"][live], cand["mixed"][live], sub)
            cmp("core", ref["core"][live], cand["core"][live], sub)
            cmp("y", ref["y"][live], cand["bf16"][live], sub)
            cmp("qa", ref["qa"][live], cand["fp8"][live], sub)
        else:
            cmp("mixed", ref["mixed"], cand["mixed"], sub)
            cmp("core", ref["core"], cand["core"], sub)
            cmp("y", ref["y"], cand["bf16"], sub)
            cmp("qa", ref["qa"], cand["fp8"], sub)
        cmp("conv_states", conv_r, conv_c, sub)
        cmp("recurrent_states", rec_r, rec_c, sub)
        step_ok = all(r["equal"] for r in sub)
        if not step_ok:
            sub_all.append({"step": s, "fails": [r for r in sub if not r["equal"]]})
        ok &= step_ok
    results.append({"case": tag, "pass": ok, "failures": sub_all[:4]})
    print(("PASS " if ok else "FAIL ") + tag + ("" if ok else " " + json.dumps(sub_all[:2])))
    return ok


def run_prefill_then_decode(lib, P, L, steps, gen, dev, real, results):
    """Continuation: prefill L tokens through the candidate prefill path, then decode steps
    through the fused kernel, versus the reference path end to end (states carried)."""
    S = 4
    qkvz, ba = make_inputs(P, [L], gen, dev, real)
    cu = torch.tensor([0, L], dtype=torch.int64, device=dev)
    idx = torch.tensor([1], dtype=torch.int32, device=dev)
    has_init = torch.tensor([False], device=dev)
    fresh = idx.to(torch.int64)
    conv_r = torch.zeros(S, CONV_DIM, 3, dtype=torch.bfloat16, device=dev)
    rec_r = torch.zeros(S, HV, D, D, dtype=torch.float32, device=dev)
    conv_c, rec_c = conv_r.clone(), rec_r.clone()
    qkvz_c = qkvz.clone()   # see run_case: never share the GEMM output buffer between the paths
    reference_prefill(P, qkvz, ba, conv_r, rec_r, cu, idx, has_init, fresh, None, None)
    candidate_prefill(lib, P, qkvz_c, ba, conv_c, rec_c, cu, idx, has_init, fresh, None, None)
    ok = True
    sub = []
    cmp("prefill conv_states", conv_r, conv_c, sub)
    cmp("prefill rec_states", rec_r, rec_c, sub)
    for s in range(steps):
        q1, b1 = make_inputs(P, [1], gen, dev, real)
        ref = reference_decode(P, q1.clone(), b1, conv_r, rec_r, idx)
        cand = lib.decode_fused(q1.clone(), b1, P["conv_w"], conv_c, P["A_log"], P["dt_bias"], rec_c, idx,
                                P["norm_w"], P["in_scale"], EPS, D ** -0.5, want_fp8=True)
        cmp(f"step{s} qa", ref["qa"], cand["fp8"], sub)
    cmp("final conv_states", conv_r, conv_c, sub)
    cmp("final rec_states", rec_r, rec_c, sub)
    torch.cuda.synchronize()
    ok = all(r["equal"] for r in sub)
    tag = f"prefill{L}+decode{steps} {'real' if real else 'syn'}"
    results.append({"case": tag, "pass": ok, "checks": [r for r in sub if not r["equal"]][:6]})
    print(("PASS " if ok else "FAIL ") + tag)
    return ok


def exhaustive_gate(lib, P, results, tag):
    """All 65536 bf16 a-values x every head: g_raw recipe vs torch ops; all b-values: beta."""
    dev = P["A_log"].device
    allbits = torch.arange(65536, dtype=torch.int32, device=dev).to(torch.int16).view(torch.bfloat16)
    a = allbits.reshape(-1, 1).expand(65536, HV)
    ref_g = (-P["A_log"].exp() * F.softplus(a.float() + P["dt_bias"])).t().contiguous()  # [H,65536]
    ref_b = allbits.sigmoid().float()
    g, b = lib.gate_eval(P["A_log"], P["dt_bias"])
    sub = []
    cmp("g_raw_all_bf16_a", ref_g, g, sub)
    cmp("beta_all_bf16_b", ref_b, b, sub)
    ok = all(r["equal"] for r in sub)
    results.append({"case": f"exhaustive gate {tag}", "pass": ok, "checks": sub})
    print(("PASS " if ok else "FAIL ") + f"exhaustive gate {tag}")
    return ok


def exhaustive_silu(lib, results):
    """All 65536 BF16 patterns (inf/NaN included) through both conv kernels with taps (0,0,0,1):
    8 one-token sequences x 8192 channels with zero state, so conv(x) = fma(x,1,+0) and the output
    is bf16(silu(x)). Reference: sgl_kernel causal_conv1d_fwd on the channel-first copy (grid
    8 x 8192). Candidate: qwen_gdn_conv_qkv with the L2 epilogue off. Conv states are compared too."""
    from freetoken.kernel.causal_conv1d import causal_conv1d_varlen
    dev = torch.device("cuda")
    vals = torch.arange(65536, dtype=torch.int32, device=dev).to(torch.int16).view(torch.bfloat16)
    X = vals.reshape(8, CONV_DIM).contiguous()          # token t, channel c
    w = torch.zeros(CONV_DIM, 4, dtype=torch.bfloat16, device=dev)
    w[:, 3] = 1.0
    cu = torch.arange(9, dtype=torch.int64, device=dev)
    idx = torch.arange(8, dtype=torch.int32, device=dev)
    has = torch.zeros(8, dtype=torch.bool, device=dev)
    cs_r = torch.zeros(8, CONV_DIM, 3, dtype=torch.bfloat16, device=dev)
    cs_c = cs_r.clone()
    ref = causal_conv1d_varlen(X.t().contiguous(), w, cs_r, cu, idx, has).t()   # [8, 8192]
    q, k, v = lib.conv_qkv(X, w, cs_c, cu, idx, has, norm_qk=False)
    cand = torch.cat([q, k, v], dim=1)
    sub = []
    cmp("silu_all_bf16_patterns_sgl_conv_vs_conv_qkv", ref, cand, sub)
    cmp("conv_states_all_bf16_patterns", cs_r, cs_c, sub)
    bad = (ref.contiguous().view(torch.int16) != cand.view(torch.int16)).reshape(-1)
    if bool(bad.any()):
        pats = vals.view(torch.int16)[bad][:32].to(torch.int32) & 0xffff
        sub.append({"name": "first_mismatching_input_bf16_bits", "equal": False, "mismatches": int(bad.sum()),
                    "numel": 65536, "patterns": [hex(int(p)) for p in pats.tolist()]})
    # full fp32 domain: pinned PTX sequence vs the plain CUDA expression x/(1+expf(-x))
    mism = lib.silu_full_range_mismatches()
    sub.append({"name": "silu_pinned_vs_plain_expf_all_2^32", "equal": mism == 0, "mismatches": mism,
                "numel": 1 << 32})
    ok = all(r["equal"] for r in sub)
    results.append({"case": "exhaustive silu", "pass": ok, "checks": sub})
    print(("PASS " if ok else "FAIL ") + "exhaustive silu")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.environ.get("QWEN36_CUDA_GDN_LIBRARY"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0,1,2,13,26,38")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fla", action="store_true",
                    help="candidate prefill = QWEN36_CUDA_GDN_FLA=1 branch (fused chunk_h + chunk_o)")
    ap.add_argument("--fla-variant", default="wu", choices=("wu", "ho"))
    args = ap.parse_args()
    global FLA, FLA_VARIANT
    FLA = args.fla
    FLA_VARIANT = args.fla_variant
    torch.manual_seed(0)
    import freetoken
    ft_root = os.environ.get("FREETOKEN_PYTHON")
    if ft_root and not os.path.dirname(freetoken.__file__).startswith(os.path.abspath(ft_root)):
        raise SystemExit("reference must be the FreeToken runtime tree (PYTHONPATH=$FREETOKEN_PYTHON)")
    dev = torch.device("cuda")
    library_path = args.lib or os.path.join(os.path.dirname(__file__),"build/libqwen36_gdn.so")
    with open(library_path,"rb") as source:library_sha256 = hashlib.sha256(source.read()).hexdigest()
    lib = GdnLib(library_path)
    with open(lib.path,"rb") as source:
        if hashlib.sha256(source.read()).hexdigest() != library_sha256:
            raise RuntimeError("GDN library changed while loading")
    results, t0 = [], time.time()
    gen = torch.Generator().manual_seed(4170)
    allok = True
    # synthetic: lengths 1/17/2048/4096/8192, fresh and continuation, varlen batch, tiles
    Ps = synthetic_params(gen, dev)
    lens_list = [[1], [2], [3], [17], [64], [65], [2048], [4096]] + ([] if args.quick else [[8192], [17, 2048, 5], [1, 2, 3, 300]])
    for lens in lens_list:
        for cont in (False, True):
            allok &= run_case(lib, Ps, lens, cont, gen, dev, False, results)
    for tile, rows in ((64, 8), (32, 8), (128, 4)):
        allok &= run_case(lib, Ps, [4096], True, gen, dev, False, results, tile=tile, rows=rows)
    allok &= exhaustive_gate(lib, Ps, results, "synthetic")
    for variant in ("cta", "cluster"):
        for B in (1, 2, 4, 5, 8):
            allok &= run_decode_case(lib, Ps, B, 3 if args.quick else 8, gen, dev, False, results,
                                     variant=variant)
        allok &= run_decode_case(lib, Ps, 2, 3, gen, dev, False, results, pad_one=True, variant=variant)
    allok &= run_prefill_then_decode(lib, Ps, 300, 6, gen, dev, False, results)
    # real checkpoint weights (activations through the real dequantised in_proj weights)
    for layer in [int(x) for x in args.layers.split(",") if x]:
        Pr = real_params(layer, dev)
        allok &= exhaustive_gate(lib, Pr, results, f"layer{layer}")
        for lens in ([17], [2048], [4096]) if args.quick else ([1], [17], [2048], [4096], [8192]):
            for cont in (False, True):
                allok &= run_case(lib, Pr, lens, cont, gen, dev, True, results)
        for variant in ("cta", "cluster"):
            allok &= run_decode_case(lib, Pr, 1, 4 if args.quick else 16, gen, dev, True, results,
                                     variant=variant)
        allok &= run_prefill_then_decode(lib, Pr, 2048 if args.quick else 4100, 4, gen, dev, True, results)
        del Pr
        torch.cuda.empty_cache()
    try:
        allok &= exhaustive_silu(lib, results)
    except Exception as exc:  # report, do not hide
        results.append({"case": "exhaustive silu", "pass": False, "error": repr(exc)[:300]})
        allok = False
    import freetoken
    from triton.backends.nvidia.compiler import get_ptxas
    pt = get_ptxas(120)
    if hashlib.sha256(open(lib.path,"rb").read()).hexdigest() != library_sha256:
        raise RuntimeError("GDN library changed during component run")
    rep = {"library_sha256":library_sha256, "all_pass": bool(allok), "seconds": round(time.time() - t0, 1), "library": lib.path,
           "freetoken_tree": os.path.dirname(freetoken.__file__),
           "triton_ptxas_sm120": {"path": pt.path, "version": pt.version},
           "torch": torch.__version__, "device": torch.cuda.get_device_name(),
           "fla_fused_candidate": FLA, "fla_variant": FLA_VARIANT if FLA else None,
           "fla_fused_calls": len(FLA_USED),
           "fla_fused_tokens": sorted(set(FLA_USED)),
           "cases": len(results), "results": results}
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    os.chmod(args.out, 0o600)
    print("ALL_PASS" if allok else "SOME_FAIL", args.out)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
