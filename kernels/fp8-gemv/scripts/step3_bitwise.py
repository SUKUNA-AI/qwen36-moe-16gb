"""Step 3 (GPU, under the lock): bitwise equality of the INTEGRATED module path on real weights.

For every dense FP8 projection of the checkpoint (40 layers x {fused input, output} = 80; the 39
row-wise fused input projections are the GEMV's targets) and M = 1..8:
  pristine = unmodified FreeToken fp8_pertensor_linear ($FREETOKEN_PRISTINE_PYTHON)
  off      = patched TorchFp8TensorLinearKernel, layers finalized with QWEN36_FP8_GEMV unset
  on       = patched TorchFp8TensorLinearKernel, layers finalized with QWEN36_FP8_GEMV=1 (GEMV)
  direct   = torch._scaled_mm exactly as the production W8A8 branch issues it
are compared bit for bit, eager and under CUDA graph capture (replays with fresh inputs), with
hidden-state-like activations (normal, heavy-tailed, edge values). The current stream is a side
stream (as in Engine), so a launch on the wrong stream would break capture or ordering.
Also: gating negatives (rowwise must decline), prefill-sized M, a multi-threaded hot path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (FP8, LIB, Checkpoint, GemvCounter, activation, env_info, load_orig_fpl,  # noqa: E402
                    load_projections, make_layer, mismatches, scaled_mm_reference)

import torch  # noqa: E402

COLS = ("on_vs_pristine", "off_vs_pristine", "direct_vs_pristine", "on_vs_direct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--ms", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--kinds", default="normal,heavy,edge")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--graph-kinds", default="normal,edge")
    args = ap.parse_args()
    ms = [int(m) for m in args.ms.split(",")]
    kinds = args.kinds.split(",")
    os.environ.pop("QWEN36_FP8_GEMV", None)
    os.environ.pop("QWEN36_FP8_GEMV_LIBRARY", None)
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    side = torch.cuda.Stream(dev)
    torch.cuda.set_stream(side)  # Engine runs on its own (non-default) stream

    orig = load_orig_fpl()
    import freetoken.kernel.triton.fp8_pertensor_linear as fpl
    from freetoken.layers.quantization.linear.fp8_tensor import TorchFp8TensorLinearKernel

    kern = TorchFp8TensorLinearKernel()
    report = {"args": vars(args)}
    report["module_imported_before_finalize"] = "freetoken.kernel.cuda_fp8_gemv" in sys.modules
    t0 = time.perf_counter()
    ck = Checkpoint()
    projs = load_projections(ck)
    report["checkpoint_load_s"] = round(time.perf_counter() - t0, 2)
    for p in projs:
        p["xs_value"] = float(p["xs"].item())
        p["off"] = make_layer(p["W"], p["S"], p["xs"])
        kern.finalize(p["off"])
    # must be checked before anything in this harness imports the module (env_info does)
    report["unset_module_imported_after_finalize"] = "freetoken.kernel.cuda_fp8_gemv" in sys.modules
    report["unset_gemv_flags"] = sorted({p["off"]._fp8_gemv for p in projs})
    report["env"] = env_info()

    os.environ["QWEN36_FP8_GEMV"] = "1"
    os.environ["QWEN36_FP8_GEMV_LIBRARY"] = str(LIB)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()  # as Engine._sync_get_memory does before measuring
    free_before = torch.cuda.mem_get_info()[0]
    t0 = time.perf_counter()
    for p in projs:
        p["on"] = make_layer(p["W"], p["S"], p["xs"])
        kern.finalize(p["on"])
    torch.cuda.synchronize()
    report["enabled_finalize_s"] = round(time.perf_counter() - t0, 3)
    torch.cuda.empty_cache()
    report["resident_device_bytes_after_enable"] = free_before - torch.cuda.mem_get_info()[0]
    import freetoken.kernel.cuda_fp8_gemv as g

    report["status_after_finalize"] = g.status()
    report["projections"] = [dict(layer=p["layer"], kind=p["kind"], type=p["type"], N=p["N"], K=p["K"],
                                  uniform=p["uniform"], gemv=p["on"]._fp8_gemv,
                                  segments=p["off"]._fp8_scale_segments) for p in projs]
    expected_gemv = [p for p in projs if not p["uniform"] and (p["N"], p["K"]) in g.SHAPES]
    report["layers_gemv_enabled"] = sum(1 for p in projs if p["on"]._fp8_gemv)
    report["layers_gemv_expected"] = len(expected_gemv)
    counter = GemvCounter()

    # ------------------------------------------------------------------ eager, all kinds/seeds
    eager = {}
    for M in ms:
        cases = [(k, s) for k in kinds for s in range(args.seeds)]
        bad = torch.zeros((len(cases), len(projs), len(COLS)), dtype=torch.int64, device=dev)
        counter.reset()
        for ci, (kind, seed) in enumerate(cases):
            gen = torch.Generator().manual_seed(100003 * M + 7919 * seed + 31 * kinds.index(kind))
            for pi, p in enumerate(projs):
                x = activation(kind, M, p["K"], p["xs_value"], gen)
                pristine = orig.fp8_pertensor_linear(x, p["W"], p["S"], None, p["xs"], p["off"]._fp8_uniform_scale,
                                                     scale_segments=p["off"]._fp8_scale_segments)
                off = kern.apply(p["off"], x)
                on = kern.apply(p["on"], x)
                direct = scaled_mm_reference(fpl, x, p["W"], p["S"], p["xs"], p["uniform"])
                bad[ci, pi, 0] += mismatches(on, pristine)
                bad[ci, pi, 1] += mismatches(off, pristine)
                bad[ci, pi, 2] += mismatches(direct, pristine)
                bad[ci, pi, 3] += mismatches(on, direct)
        counts = bad.cpu()
        per_shape = {}
        for pi, p in enumerate(projs):
            key = f"{p['N']}x{p['K']}{'-uniform' if p['uniform'] else ''}"
            row = per_shape.setdefault(key, {c: 0 for c in COLS} | {"elements": 0, "projections": 0})
            for j, c in enumerate(COLS):
                row[c] += int(counts[:, pi, j].sum())
            row["elements"] += M * p["N"] * len(cases)
            row["projections"] += 1
        eager[M] = {"cases": len(cases), "per_shape": per_shape,
                    "mismatches_total": {c: int(counts[:, :, j].sum()) for j, c in enumerate(COLS)},
                    "gemv_served": {f"{n}xM{m}": v for (n, m), v in sorted(counter.served.items())},
                    "gemv_declined": {f"{n}xM{m}": v for (n, m), v in sorted(counter.declined.items())},
                    "gemv_served_expected": len(expected_gemv) * len(cases)}
        print(json.dumps({"eager_M": M, **eager[M]["mismatches_total"], "served": counter.total(),
                          "expected": len(expected_gemv) * len(cases)}), flush=True)
    report["eager"] = eager

    # ------------------------------------------------------------------ CUDA graph capture
    graph = {}
    names = ("pristine", "off", "on")
    for M in ms:
        static_x = [torch.zeros(M, p["K"], dtype=torch.bfloat16, device=dev) for p in projs]
        outs = {n: [None] * len(projs) for n in names}

        def seq(name):
            for pi, p in enumerate(projs):
                x = static_x[pi]
                if name == "pristine":
                    outs[name][pi] = orig.fp8_pertensor_linear(x, p["W"], p["S"], None, p["xs"],
                                                               p["off"]._fp8_uniform_scale,
                                                               scale_segments=p["off"]._fp8_scale_segments)
                else:
                    outs[name][pi] = kern.apply(p[name], x)

        gen = torch.Generator().manual_seed(4242 + M)
        for pi, p in enumerate(projs):
            static_x[pi].copy_(activation("normal", M, p["K"], p["xs_value"], gen))
        for n in names:  # eager warm-up before capture (as GraphRunner does)
            seq(n)
        torch.cuda.synchronize()
        counter.reset()
        graphs, capture_error = {}, None
        try:
            for n in names:
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr, stream=side):
                    seq(n)
                graphs[n] = gr
        except Exception as exc:  # noqa: BLE001 - recorded as a failure
            capture_error = f"{type(exc).__name__}: {exc}"
        served_in_capture = counter.total()
        res = {"capture_error": capture_error, "gemv_served_during_capture": served_in_capture,
               "gemv_expected_during_capture": len(expected_gemv), "replays": {}}
        if capture_error is None:
            for kind in args.graph_kinds.split(","):
                for rep in range(2):
                    gen = torch.Generator().manual_seed(9000 + 17 * M + rep + 101 * kinds.index(kind))
                    for pi, p in enumerate(projs):
                        static_x[pi].copy_(activation(kind, M, p["K"], p["xs_value"], gen))
                    for n in names:
                        graphs[n].replay()
                    torch.cuda.synchronize()
                    counter.reset()
                    bad = torch.zeros(4, dtype=torch.int64, device=dev)
                    for pi, p in enumerate(projs):
                        ref = orig.fp8_pertensor_linear(static_x[pi], p["W"], p["S"], None, p["xs"],
                                                        p["off"]._fp8_uniform_scale,
                                                        scale_segments=p["off"]._fp8_scale_segments)
                        eager_on = kern.apply(p["on"], static_x[pi])
                        bad[0] += mismatches(outs["on"][pi], ref)
                        bad[1] += mismatches(outs["off"][pi], ref)
                        bad[2] += mismatches(outs["pristine"][pi], ref)
                        bad[3] += mismatches(outs["on"][pi], eager_on)
                    c = bad.tolist()
                    res["replays"][f"{kind}#{rep}"] = {"graph_on_vs_eager_pristine": c[0],
                                                       "graph_off_vs_eager_pristine": c[1],
                                                       "graph_pristine_vs_eager_pristine": c[2],
                                                       "graph_on_vs_eager_on": c[3],
                                                       "elements": sum(M * p["N"] for p in projs)}
        graph[M] = res
        print(json.dumps({"graph_M": M, "capture_error": capture_error, "served_in_capture": served_in_capture,
                          "replays": {k: v["graph_on_vs_eager_pristine"] for k, v in res["replays"].items()}}),
              flush=True)
        del graphs, outs, static_x
    report["graph"] = graph

    # ------------------------------------------------------------------ gating negatives
    rowwise = counter.orig
    gdn = next(p for p in projs if p["on"]._fp8_gemv and p["N"] == 12288)
    attn = next(p for p in projs if p["on"]._fp8_gemv and p["N"] == 9216)
    outp = next(p for p in projs if p["kind"] == "out")
    gen = torch.Generator().manual_seed(77)
    x4 = activation("normal", 4, 2048, gdn["xs_value"], gen)
    qa4 = fpl._static_quant(x4, gdn["xs"])
    good = rowwise(qa4, gdn["W"], gdn["xs"], gdn["S"])
    ref = scaled_mm_reference(fpl, x4, gdn["W"], gdn["S"], gdn["xs"], False)
    gating = {"good_call_served": good is not None,
              "good_call_mismatches": int(mismatches(good, ref)) if good is not None else None}
    wide = torch.zeros(4, 4096, dtype=torch.uint8, device=dev)
    wide[:, :2048] = qa4.view(torch.uint8)
    raw = torch.zeros(4 * 2048 + 64, dtype=torch.uint8, device=dev)
    mis = raw[1:1 + 4 * 2048].view(4, 2048)
    mis.copy_(qa4.view(torch.uint8))
    strided_s = torch.zeros(2 * gdn["N"], dtype=torch.float32, device=dev)[::2]
    strided_s.copy_(gdn["S"])
    x9 = activation("normal", 9, 2048, gdn["xs_value"], gen)
    qa_out = fpl._static_quant(activation("normal", 2, 4096, outp["xs_value"], gen), outp["xs"])
    negatives = {
        "M9": (fpl._static_quant(x9, gdn["xs"]), gdn["W"], gdn["xs"], gdn["S"]),
        "M0": (torch.empty(0, 2048, dtype=FP8, device=dev), gdn["W"], gdn["xs"], gdn["S"]),
        "qa_row_stride_4096": (wide[:, :2048].view(FP8), gdn["W"], gdn["xs"], gdn["S"]),
        "qa_misaligned_ptr": (mis.view(FP8), gdn["W"], gdn["xs"], gdn["S"]),
        "qa_bf16": (x4, gdn["W"], gdn["xs"], gdn["S"]),
        "qa_3d": (qa4.view(1, 4, 2048), gdn["W"], gdn["xs"], gdn["S"]),
        "weight_out_proj_shape": (qa_out, outp["W"], outp["xs"], outp["S"]),
        "weight_rows_sliced": (qa4, gdn["W"][:-16], gdn["xs"], gdn["S"][:-16]),
        "weight_transposed_layout": (qa4, gdn["W"].t().contiguous().t(), gdn["xs"], gdn["S"]),
        "ws_fp16": (qa4, gdn["W"], gdn["xs"], gdn["S"].half()),
        "ws_strided": (qa4, gdn["W"], gdn["xs"], strided_s),
        "ws_wrong_length": (qa4, attn["W"], attn["xs"], gdn["S"]),
        "xs_cpu": (qa4, gdn["W"], gdn["xs"].cpu(), gdn["S"]),
        "xs_fp16": (qa4, gdn["W"], gdn["xs"].half(), gdn["S"]),
        "xs_two_elements": (qa4, gdn["W"], torch.stack([gdn["xs"], gdn["xs"]]), gdn["S"]),
    }
    for name, a in negatives.items():
        try:
            out = rowwise(*a)
            gating[name] = "declined" if out is None else "SERVED"
        except Exception as exc:  # noqa: BLE001
            gating[name] = f"raised {type(exc).__name__}: {exc}"[:200]
    gating["all_negatives_declined"] = all(gating[k] == "declined" for k in negatives)
    gating["status_still_enabled"] = g.status()["enabled"]
    report["gating"] = gating
    print(json.dumps({"gating_all_declined": gating["all_negatives_declined"], "good": gating["good_call_mismatches"]}),
          flush=True)

    # ------------------------------------------------------------------ prefill-sized M
    prefill = {}
    for M in (9, 16, 64, 512):
        counter.reset()
        bad = torch.zeros(1, dtype=torch.int64, device=dev)
        gen = torch.Generator().manual_seed(555 + M)
        for p in projs[:8]:
            x = activation("normal", M, p["K"], p["xs_value"], gen)
            pristine = orig.fp8_pertensor_linear(x, p["W"], p["S"], None, p["xs"], p["off"]._fp8_uniform_scale,
                                                 scale_segments=p["off"]._fp8_scale_segments)
            bad[0] += mismatches(kern.apply(p["on"], x), pristine)
        prefill[M] = {"mismatches": int(bad.item()), "gemv_served": counter.total(),
                      "rowwise_called": sum(counter.declined.values())}
    report["prefill"] = prefill

    # ------------------------------------------------------------------ concurrent hot path
    pairs = []
    for p in (gdn, attn):
        for M in range(1, 9):
            x = activation("normal", M, 2048, p["xs_value"], gen)
            qa = fpl._static_quant(x, p["xs"])
            pairs.append((qa, p, scaled_mm_reference(fpl, x, p["W"], p["S"], p["xs"], False)))
    torch.cuda.synchronize()
    thread_bad, thread_none, errors = [0] * 4, [0] * 4, []

    def worker(t):
        try:
            s = torch.cuda.Stream(dev)
            with torch.cuda.stream(s):
                acc = torch.zeros(1, dtype=torch.int64, device=dev)
                for _ in range(25):
                    for qa, p, refp in pairs:
                        out = rowwise(qa, p["W"], p["xs"], p["S"])
                        if out is None:
                            thread_none[t] += 1
                            continue
                        acc += mismatches(out, refp)
                s.synchronize()
                thread_bad[t] = int(acc.item())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    report["threads"] = {"threads": 4, "calls_per_thread": 25 * len(pairs), "mismatches": thread_bad,
                         "declined": thread_none, "errors": errors}
    print(json.dumps({"threads": report["threads"]}), flush=True)

    report["status_final"] = g.status()
    ok = (not report["unset_module_imported_after_finalize"]
          and report["layers_gemv_enabled"] == report["layers_gemv_expected"] == 39
          and all(v["mismatches_total"][c] == 0 for v in eager.values() for c in COLS)
          and all(v["gemv_served_expected"] == sum(v["gemv_served"].values()) for v in eager.values())
          and all(v["capture_error"] is None and v["gemv_served_during_capture"] == 39 for v in graph.values())
          and all(r[k] == 0 for v in graph.values() for r in v["replays"].values()
                  for k in ("graph_on_vs_eager_pristine", "graph_off_vs_eager_pristine",
                            "graph_pristine_vs_eager_pristine", "graph_on_vs_eager_on"))
          and gating["all_negatives_declined"] and gating["good_call_mismatches"] == 0
          and all(v["mismatches"] == 0 and v["gemv_served"] == 0 and v["rowwise_called"] == 0 for v in prefill.values())
          and not errors and sum(thread_bad) == 0 and sum(thread_none) == 0)
    report["verdict"] = "PASS" if ok else "FAIL"
    with open(args.output, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps({"verdict": report["verdict"]}), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
