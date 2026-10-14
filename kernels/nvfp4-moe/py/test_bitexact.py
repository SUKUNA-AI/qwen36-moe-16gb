"""Bit-exactness tests: CUDA bit-exact NVFP4 MoE prefill vs the pinned Triton path.

GPU required; must be started through gpu-run.sh (non-blocking flock on the shared GPU
lock).  Synthetic data only here (real checkpoint weights: test_real.py).  Every
comparison is on raw BF16 bit patterns (int16 view), never on tolerances.
"""
import argparse
import json
import os
import sys
import time

import torch

from freetoken.moe import fused_nvfp4 as ref
from freetoken.moe.fused import moe_align_block_size
from freetoken.layers import gated_act_and_mul
from freetoken.kernel import moe_sum_reduce_triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bitexact_lib import BitExactPrefill  # noqa: E402

H, I, E, TOPK = 2048, 512, 256, 8
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def bits(t):
    return t.contiguous().view(torch.int16)


def compare(actual, expected, name):
    a, e = bits(actual).reshape(-1), bits(expected).reshape(-1)
    eq = a == e
    n = eq.numel()
    bad = int((~eq).sum())
    res = {"name": name, "elements": n, "mismatch": bad, "bit_equal_fraction": (n - bad) / n if n else 1.0}
    if bad:
        idx = torch.nonzero(~eq).reshape(-1)
        af = actual.reshape(-1).float()[idx]
        ef = expected.reshape(-1).float()[idx]
        res["first_mismatch_flat_index"] = int(idx[0])
        res["first_mismatch_bits"] = [int(a[idx[0]]) & 0xFFFF, int(e[idx[0]]) & 0xFFFF]
        res["max_abs_diff"] = float((af - ef).abs().max())
        res["max_ulp_diff"] = int((a[idx].int() - e[idx].int()).abs().max())
    return res


def make_banks(gen, scale_mode, global_mode, dev="cuda"):
    def bank(n, k):
        packed = torch.randint(0, 256, (E, n, k // 2), device=dev, dtype=torch.uint8, generator=gen)
        if scale_mode == "allcodes":
            # all 254 finite E4M3 codes (0x7F/0xFF are NaN) incl. zero, subnormals, negatives
            codes = torch.randint(0, 254, (E, n, k // 16), device=dev, dtype=torch.int32, generator=gen)
            codes = codes + (codes >= 0x7F).int()
            scales = codes.to(torch.uint8).view(torch.float8_e4m3fn)
        elif scale_mode == "positive":
            codes = torch.randint(1, 0x7F, (E, n, k // 16), device=dev, dtype=torch.int32, generator=gen)
            scales = codes.to(torch.uint8).view(torch.float8_e4m3fn)
        else:  # realistic
            scales = (torch.rand(E, n, k // 16, device=dev, generator=gen) * .4 + .02).to(torch.float8_e4m3fn)
        if global_mode == "wide":
            glob = torch.exp2(torch.rand(E, n, device=dev, generator=gen) * 10 - 12)
        else:
            glob = torch.rand(E, n, device=dev, generator=gen) * .06 + .01
        return packed, scales, glob
    return (*bank(2 * I, H), *bank(H, I))


def make_tokens(gen, m, act_mode, routing, dev="cuda"):
    x = torch.randn(m, H, device=dev, generator=gen)
    if act_mode == "heavy":
        x = x * torch.exp2(torch.randint(-8, 9, (m, H), device=dev, generator=gen).float())
    x = x.to(torch.bfloat16)
    if routing == "zipf":
        # skewed popularity: a few experts receive most routes (multi-tile experts)
        pop = 1.0 / torch.arange(1, E + 1, device=dev).float() ** 1.1
        pop = pop[torch.randperm(E, device=dev, generator=gen)]
        ids = torch.multinomial(pop.expand(m, E), TOPK, replacement=False, generator=gen).int()
    else:
        ids = torch.argsort(torch.rand(m, E, device=dev, generator=gen), dim=1)[:, :TOPK].int().contiguous()
    w = torch.softmax(torch.randn(m, TOPK, device=dev, generator=gen), dim=-1).float().contiguous()
    return x, ids, w


def triton_stages(x, banks, w, ids, on_input=False):
    gu_p, gu_s, gu_g, d_p, d_s, d_g = banks
    m = x.shape[0]
    cfg = ref._prefill_config(m)
    sorted_ids, expert_ids, ntpp = moe_align_block_size(ids, cfg["BLOCK_SIZE_M"], E)
    tw = w.reshape(-1).contiguous()
    ic1 = torch.empty((m, TOPK, 2 * I), device=x.device, dtype=x.dtype)
    ref._prefill_gemm(x, gu_p, gu_s, gu_g, ic1, tw, sorted_ids, expert_ids, ntpp, m * TOPK, TOPK, on_input, cfg)
    ic2 = torch.empty((m * TOPK, I), device=x.device, dtype=x.dtype)
    gated_act_and_mul("silu", ic1.view(-1, 2 * I), ic2)
    ic3 = torch.empty((m, TOPK, H), device=x.device, dtype=x.dtype)
    ref._prefill_gemm(ic2, d_p, d_s, d_g, ic3, tw, sorted_ids, expert_ids, ntpp, m * TOPK, 1, not on_input, cfg)
    out = torch.empty_like(x)
    moe_sum_reduce_triton(ic3, out)
    full = ref.fused_experts_nvfp4(x, *banks, w, ids, E, apply_router_weight_on_input=on_input)
    assert torch.equal(bits(full), bits(out)), "stage decomposition differs from fused_experts_nvfp4"
    return ic1.view(m * TOPK, 2 * I), ic2, ic3, out, cfg


def cuda_checks(lib, x, banks, w, ids, stages, on_input, variants, use_oracle, tag):
    gu_p, gu_s, gu_g, d_p, d_s, d_g = banks
    ic1, ic2, ic3, out, _ = stages
    m = x.shape[0]
    tw = w.reshape(-1).contiguous()
    results = []
    align = lib.tile[0]
    sorted_ids, expert_ids, ntpp = moe_align_block_size(ids, align, E)
    if use_oracle:
        o1 = torch.full((m * TOPK, 2 * I), -1, device=x.device, dtype=torch.int16).view(torch.bfloat16)
        lib.oracle(x, gu_p, gu_s, gu_g, o1, tw, sorted_ids, expert_ids, ntpp, m * TOPK, TOPK, on_input)
        results.append(compare(o1, ic1, f"{tag}/oracle/gate_up"))
        o3 = torch.full((m, TOPK, H), -1, device=x.device, dtype=torch.int16).view(torch.bfloat16)
        lib.oracle(ic2, d_p, d_s, d_g, o3, tw, sorted_ids, expert_ids, ntpp, m * TOPK, 1, not on_input)
        results.append(compare(o3, ic3, f"{tag}/oracle/down"))
    s2 = torch.empty_like(ic2)
    lib.silu_and_mul(ic1, s2)
    results.append(compare(s2, ic2, f"{tag}/silu_replica"))
    for v in variants:
        va = max(align, lib.variant_bm(v))
        if va != align:
            sids, eids, nt = moe_align_block_size(ids, va, E)
        else:
            sids, eids, nt = sorted_ids, expert_ids, ntpp
        c1 = torch.full((m * TOPK, 2 * I), -1, device=x.device, dtype=torch.int16).view(torch.bfloat16)
        lib.gemm(x, gu_p, gu_s, gu_g, c1, tw, sids, eids, nt, m * TOPK, TOPK, on_input, False, variant=v, align=va)
        results.append(compare(c1, ic1, f"{tag}/v{v}/gate_up"))
        c2 = torch.full_like(ic2, -1.0)
        lib.gemm(x, gu_p, gu_s, gu_g, c2, tw, sids, eids, nt, m * TOPK, TOPK, on_input, True, variant=v, align=va)
        results.append(compare(c2, ic2, f"{tag}/v{v}/gate_up_silu_fused"))
        c3 = torch.full((m, TOPK, H), -1, device=x.device, dtype=torch.int16).view(torch.bfloat16)
        lib.gemm(ic2, d_p, d_s, d_g, c3, tw, sids, eids, nt, m * TOPK, 1, not on_input, False, variant=v, align=va)
        results.append(compare(c3, ic3, f"{tag}/v{v}/down"))
    # full chain through the old-ABI entry point (automatic variant), both reductions
    full = lib(x, *banks, w, ids, E, apply_router_weight_on_input=on_input)
    results.append(compare(full, out, f"{tag}/full_layer_triton_sum"))
    lib.cuda_sum = True
    full2 = lib(x, *banks, w, ids, E, apply_router_weight_on_input=on_input)
    lib.cuda_sum = False
    results.append(compare(full2, out, f"{tag}/full_layer_cuda_sum"))
    so = torch.empty_like(out)
    lib.moe_sum(ic3.view(m, TOPK, H), so)
    results.append(compare(so, out, f"{tag}/sum_replica"))
    # strided activation rows (pool with larger row stride)
    backing = torch.empty((m, 2 * H), device=x.device, dtype=x.dtype)
    backing[:, :H].copy_(x)
    full3 = lib(backing[:, :H], *banks, w, ids, E, apply_router_weight_on_input=on_input)
    results.append(compare(full3, out, f"{tag}/full_layer_strided_input"))
    return results


def suite_synthetic(lib, ms, variants, oracle_max_m):
    out = []
    configs = [
        ("allcodes", "wide", "heavy", "uniform", False),
        ("realistic", "realistic", "normal", "zipf", False),
        ("positive", "wide", "heavy", "zipf", True),
    ]
    for ci, (sm, gm, am, rt, on_input) in enumerate(configs):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(1234 + ci)
        banks = make_banks(gen, sm, gm)
        for m in ms:
            gen.manual_seed(99991 * (ci + 1) + m)
            x, ids, w = make_tokens(gen, m, am, rt)
            t0 = time.time()
            stages = triton_stages(x, banks, w, ids, on_input)
            tag = f"syn[{sm},{gm},{am},{rt},on_input={int(on_input)}]/M{m}"
            res = cuda_checks(lib, x, banks, w, ids, stages, on_input, variants, m <= oracle_max_m, tag)
            torch.cuda.synchronize()
            bad = [r for r in res if r["mismatch"]]
            print(f"{tag}: {len(res) - len(bad)}/{len(res)} bit-exact ({time.time() - t0:.1f}s)", flush=True)
            for r in bad:
                print("   MISMATCH", json.dumps(r), flush=True)
            out.extend(res)
        del banks
        torch.cuda.empty_cache()
    return out


def suite_dequant(lib):
    dev = "cuda"
    # every code at every nibble position, plus random words, against every scale code
    base = []
    for c in range(16):
        w = 0
        for j in range(8):
            w |= ((c + 3 * j) % 16) << (4 * j)
        base.append(w)
    g = torch.Generator(device=dev)
    g.manual_seed(7)
    rnd = torch.randint(0, 2**32, (240,), device=dev, dtype=torch.int64, generator=g)
    words = torch.cat([torch.tensor(base, device=dev, dtype=torch.int64), rnd])
    nw = words.numel()
    words = words.repeat(256)
    scales = torch.arange(256, device=dev, dtype=torch.int32).repeat_interleave(nw).to(torch.uint8)
    w32 = words.to(torch.int64).bitwise_and(0xFFFFFFFF)
    w32 = torch.where(w32 >= 2**31, w32 - 2**32, w32).to(torch.int32)
    got = lib.dequant_fixture(w32, scales)
    nib = torch.stack([(words >> (4 * j)) & 15 for j in range(8)], dim=1)
    lut = torch.tensor(E2M1, device=dev, dtype=torch.float32)
    sc = scales.view(torch.float8_e4m3fn).float()
    expect = (lut[nib] * sc[:, None]).to(torch.bfloat16)
    finite = torch.isfinite(sc)
    res = compare(got.view(torch.bfloat16)[finite], expect[finite], "dequant/all_finite_scale_codes_x_all_nibbles")
    nan_rows = ~finite
    nan_ok = bool(torch.isnan(got.view(torch.bfloat16)[nan_rows].float()).all())
    res["nan_scale_codes_give_nan"] = nan_ok
    res["nan_bits_equal_to_triton_style"] = bool(torch.equal(got[nan_rows], bits(expect[nan_rows])))
    print(json.dumps(res), flush=True)
    return [res]


def suite_silu(lib):
    dev = "cuda"
    allbits = torch.arange(-32768, 32768, device=dev, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    g = torch.Generator(device=dev)
    g.manual_seed(11)
    ups = {
        "up=1": torch.ones(65536, device=dev, dtype=torch.bfloat16),
        "up=-3.5": torch.full((65536,), -3.5, device=dev, dtype=torch.bfloat16),
        "up=allbits_perm": allbits[torch.randperm(65536, device=dev, generator=g)],
        "up=normal": torch.randn(65536, device=dev, generator=g).to(torch.bfloat16),
    }
    out = []
    d = 4096
    rows = 65536 // d
    for name, up in ups.items():
        x = torch.cat([allbits.view(rows, d), up.view(rows, d)], dim=1).contiguous()
        e = torch.empty((rows, d), device=dev, dtype=torch.bfloat16)
        gated_act_and_mul("silu", x, e)
        a = torch.empty_like(e)
        lib.silu_and_mul(x, a)
        r = compare(a, e, f"silu/all_65536_gate_bits/{name}")
        out.append(r)
        print(json.dumps(r), flush=True)
    return out


def suite_sum(lib):
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(5)
    out = []
    for m in (1, 17, 2048):
        x = torch.randn(m, TOPK, H, device=dev, generator=g)
        x = (x * torch.exp2(torch.randint(-20, 21, x.shape, device=dev, generator=g).float())).to(torch.bfloat16)
        e = torch.empty((m, H), device=dev, dtype=torch.bfloat16)
        moe_sum_reduce_triton(x, e)
        a = torch.empty_like(e)
        lib.moe_sum(x, a)
        r = compare(a, e, f"sum/adversarial/M{m}")
        out.append(r)
        print(json.dumps(r), flush=True)
    return out


def suite_triton_selfcheck():
    """Is the HMMA result independent of the row position inside the tile?  Run the
    Triton kernel itself with BLOCK_M=16 and BLOCK_M=32 on the same data."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(4242)
    banks = make_banks(gen, "allcodes", "wide")
    x, ids, w = make_tokens(gen, 300, "heavy", "zipf")
    tw = w.reshape(-1).contiguous()
    outs = []
    for bm in (16, 32, 64):
        cfg = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32, GROUP_SIZE_M=8 if bm > 16 else 1,
                   num_warps=8, num_stages=4)
        s, e_, n = moe_align_block_size(ids, bm, E)
        c = torch.empty((300, TOPK, 2 * I), device="cuda", dtype=torch.bfloat16)
        ref._prefill_gemm(x, banks[0], banks[1], banks[2], c, tw, s, e_, n, 300 * TOPK, TOPK, False, cfg)
        outs.append(c)
    r1 = compare(outs[1], outs[0], "triton_self/BM32_vs_BM16")
    r2 = compare(outs[2], outs[0], "triton_self/BM64_vs_BM16")
    print(json.dumps(r1), json.dumps(r2), flush=True)
    return [r1, r2]


def suite_negative(lib):
    """Negative controls: the same tensors with the K placement of the previous CUDA
    variants must NOT be bit-exact; otherwise the test would be insensitive."""
    out = []
    # optional: the earlier (~99.9 % bit-equal) "matched" CUDA variant, see kernels/README.md
    old = os.environ.get("QWEN36_NVFP4_PREVIOUS_LIBRARY", "")
    for ci, (sm, gm, am, rt) in enumerate([("realistic", "realistic", "normal", "zipf"),
                                           ("allcodes", "wide", "heavy", "uniform")]):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(777 + ci)
        banks = make_banks(gen, sm, gm)
        x, ids, w = make_tokens(gen, 256, am, rt)
        ic1, ic2, ic3, ref_out, _ = triton_stages(x, banks, w, ids, False)
        tw = w.reshape(-1).contiguous()
        s, e_, n = moe_align_block_size(ids, 64, E)
        for mode, label in ((0, "triton_kwidth4_mapping"), (1, "even32_odd32_natural_k16"), (2, "natural_k"),
                            (3, "triton_set_halves_swapped"), (4, "triton_set_lane_pairs_reversed"),
                            (5, "triton_set_pair_elements_swapped")):
            o1 = torch.empty_like(ic1)
            lib.oracle(x, banks[0], banks[1], banks[2], o1, tw, s, e_, n, 256 * TOPK, TOPK, False, mode=mode)
            o3 = torch.empty_like(ic3)
            lib.oracle(ic2, banks[3], banks[4], banks[5], o3, tw, s, e_, n, 256 * TOPK, 1, True, mode=mode)
            out.append(compare(o1, ic1, f"negative[{sm}]/oracle_mode{mode}_{label}/gate_up"))
            out.append(compare(o3, ic3, f"negative[{sm}]/oracle_mode{mode}_{label}/down"))
        if os.path.exists(old):
            prev = BitExactPrefill(library=old, tile=(64, 64, 64))
            o2 = torch.empty_like(ic2)
            prev.gemm_abi(x, banks[0], banks[1], banks[2], o2, tw, s, e_, n, 256 * TOPK, TOPK, False, True)
            out.append(compare(o2, ic2, f"negative[{sm}]/previous_matched_library/gate_up_silu"))
        for r in out[-13:]:
            print(json.dumps(r), flush=True)
        del banks
    return out


def suite_high_slot(lib):
    """Bank offsets beyond int32 (slot >= 2048 of a 3932-slot cache), metadata built
    directly (as in the previous high-slot fixture); Triton is the reference."""
    e, n, k, m = 3932, 1024, 2048, 17
    p = torch.empty((e, n, k // 2), device="cuda", dtype=torch.uint8)
    s = torch.empty((e, n, k // 16), device="cuda", dtype=torch.float8_e4m3fn)
    g = torch.empty((e, n), device="cuda", dtype=torch.float32)
    selected = [0, 17, 511, 2047, 2048, 3071, 3929, 3931]
    gen = torch.Generator(device="cuda")
    gen.manual_seed(31)
    for slot in selected:
        p[slot].copy_(torch.randint(0, 256, p[slot].shape, device="cuda", dtype=torch.uint8, generator=gen))
        s[slot].copy_((torch.rand(s[slot].shape, device="cuda", generator=gen) * .4 + .02).to(s.dtype))
        g[slot].copy_(torch.rand(g[slot].shape, device="cuda", generator=gen) * .06 + .01)
    x = torch.randn(m, k, device="cuda", generator=gen).to(torch.bfloat16)
    w = torch.full((m, 8), .125, device="cuda")
    out = []
    for bm in (16, 64):
        route_rows, slot_rows = [], []
        for position, slot in enumerate(selected):
            row = [token * 8 + position for token in range(m)]
            row += [m * 8] * ((-len(row)) % bm)
            route_rows.extend(row)
            slot_rows.extend([slot] * (len(row) // bm))
        sorted_ids = torch.tensor(route_rows, device="cuda", dtype=torch.int32)
        experts = torch.tensor(slot_rows, device="cuda", dtype=torch.int32)
        ntpp = torch.tensor([len(route_rows)], device="cuda", dtype=torch.int32)
        expected = torch.empty((m, 8, n), device="cuda", dtype=torch.bfloat16)
        cfg = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32, GROUP_SIZE_M=8, num_warps=8, num_stages=4)
        ref._prefill_gemm(x, p, s, g, expected, w.flatten(), sorted_ids, experts, ntpp, m * 8, 8, False, cfg)
        for v in range(lib.variant_count()):
            if bm % lib.variant_bm(v):
                continue
            got = torch.full_like(expected, -1.0)
            lib.gemm(x, p, s, g, got, w.flatten(), sorted_ids, experts, ntpp, m * 8, 8, False, False, variant=v, align=bm)
            out.append(compare(got, expected, f"high_slot/align{bm}/v{v}"))
    bad = [r for r in out if r["mismatch"]]
    print(f"high_slot: {len(out) - len(bad)}/{len(out)} bit-exact", flush=True)
    del p, s, g
    torch.cuda.empty_cache()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all")
    ap.add_argument("--m", default="1,7,17,64,65,256,2048,4096")
    ap.add_argument("--variants", default="all")
    ap.add_argument("--oracle-max-m", type=int, default=2048)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    assert os.environ.get("QWEN36_PREFILL_TILE") is None
    torch.backends.cuda.matmul.allow_tf32 = False
    lib = BitExactPrefill(tile=(64, 64, 64))
    variants = list(range(lib.variant_count())) if args.variants == "all" else [int(v) for v in args.variants.split(",")]
    ms = [int(m) for m in args.m.split(",")]
    report = {"library": lib.path, "build_id": lib.build_id(), "device": torch.cuda.get_device_name(),
              "torch": torch.__version__, "variants": variants, "results": []}
    suites = args.suite.split(",") if args.suite != "all" else ["dequant", "silu", "sum", "triton_self", "high_slot",
                                                                 "synthetic"]
    for s in suites:
        if s == "dequant":
            report["results"] += suite_dequant(lib)
        elif s == "silu":
            report["results"] += suite_silu(lib)
        elif s == "sum":
            report["results"] += suite_sum(lib)
        elif s == "triton_self":
            report["results"] += suite_triton_selfcheck()
        elif s == "synthetic":
            report["results"] += suite_synthetic(lib, ms, variants, args.oracle_max_m)
        elif s == "negative":
            report["negative_controls"] = suite_negative(lib)
        elif s == "high_slot":
            report["results"] += suite_high_slot(lib)
    positives = [r for r in report["results"]]
    tot = sum(r["elements"] for r in positives)
    bad = sum(r["mismatch"] for r in positives)
    report["summary"] = {"comparisons": len(report["results"]), "elements": tot, "mismatched_elements": bad,
                         "bit_equal_fraction": (tot - bad) / tot if tot else 1.0,
                         "all_bit_exact": bad == 0}
    print(json.dumps(report["summary"]), flush=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
