"""Step 4 (GPU, under the lock, NO profiler): speed of the 80-projection decode sequence through the
INTEGRATED wrapper, M = 1..8, eager and CUDA graph, 5 alternating pairs, CUDA events, medians.

baseline = pristine FreeToken fp8_pertensor_linear (current behaviour)
gemv     = patched TorchFp8TensorLinearKernel.apply on layers finalized with QWEN36_FP8_GEMV=1
off      = patched apply on layers finalized with the switch unset (3 pairs: must equal baseline)
All 80 projections (1.26 GB of FP8 weights >> 48 MB L2) per iteration, real checkpoint weights.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LIB, Checkpoint, GemvCounter, activation, env_info, load_orig_fpl, load_projections, make_layer, mismatches  # noqa: E402

import torch  # noqa: E402


def events_ms(fn, iters: int) -> float:
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / iters


def summarize(rows, a, b):
    ratio = [r[a] / r[b] for r in rows]
    return {"pairs_ms": rows, "median_ms": {a: statistics.median(r[a] for r in rows),
                                            b: statistics.median(r[b] for r in rows)},
            "speedup_median": statistics.median(ratio), "speedup_min": min(ratio), "speedup_max": max(ratio),
            "saved_ms_median": statistics.median(r[a] - r[b] for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--ms", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--eager-iters", type=int, default=20)
    ap.add_argument("--graph-iters", type=int, default=50)
    args = ap.parse_args()
    os.environ.pop("QWEN36_FP8_GEMV", None)
    os.environ.pop("QWEN36_FP8_GEMV_LIBRARY", None)
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    side = torch.cuda.Stream(dev)
    torch.cuda.set_stream(side)
    orig = load_orig_fpl()
    from freetoken.layers.quantization.linear.fp8_tensor import TorchFp8TensorLinearKernel

    kern = TorchFp8TensorLinearKernel()
    projs = load_projections(Checkpoint())
    for p in projs:
        p["xs_value"] = float(p["xs"].item())
        p["off"] = make_layer(p["W"], p["S"], p["xs"])
        kern.finalize(p["off"])
    os.environ["QWEN36_FP8_GEMV"] = "1"
    os.environ["QWEN36_FP8_GEMV_LIBRARY"] = str(LIB)
    for p in projs:
        p["on"] = make_layer(p["W"], p["S"], p["xs"])
        kern.finalize(p["on"])
    import freetoken.kernel.cuda_fp8_gemv as g

    report = {"env": env_info(), "args": vars(args), "status": g.status(),
              "weights_bytes": sum(p["W"].numel() for p in projs), "projections": len(projs),
              "gemv_layers": sum(1 for p in projs if p["on"]._fp8_gemv), "per_M": {}}
    counter = GemvCounter()
    for M in [int(m) for m in args.ms.split(",")]:
        gen = torch.Generator().manual_seed(2026 + M)
        xs = [activation("normal", M, p["K"], p["xs_value"], gen) for p in projs]
        outs = {n: [None] * len(projs) for n in ("baseline", "gemv", "off")}

        def run(name):
            o = outs[name]
            if name == "baseline":
                for i, p in enumerate(projs):
                    o[i] = orig.fp8_pertensor_linear(xs[i], p["W"], p["S"], None, p["xs"], p["off"]._fp8_uniform_scale,
                                                     scale_segments=p["off"]._fp8_scale_segments)
            else:
                key = "on" if name == "gemv" else "off"
                for i, p in enumerate(projs):
                    o[i] = kern.apply(p[key], xs[i])

        for n in outs:  # warm-up + bit check before timing
            for _ in range(3):
                run(n)
        torch.cuda.synchronize()
        counter.reset()
        run("gemv")
        served = counter.total()
        bad = torch.zeros(2, dtype=torch.int64, device=dev)
        run("baseline")
        run("off")
        for i in range(len(projs)):
            bad[0] += mismatches(outs["gemv"][i], outs["baseline"][i])
            bad[1] += mismatches(outs["off"][i], outs["baseline"][i])
        bits = bad.tolist()
        graphs = {}
        for n in outs:
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=side):
                run(n)
            graphs[n] = gr
        for gr in graphs.values():
            for _ in range(3):
                gr.replay()
        torch.cuda.synchronize()
        res = {"M": M, "bit_mismatches_gemv_vs_baseline": bits[0], "bit_mismatches_off_vs_baseline": bits[1],
               "gemv_calls_served_per_sequence": served}
        for mode in ("eager", "graph"):
            fns = ({n: (lambda n=n: run(n)) for n in outs} if mode == "eager"
                   else {n: graphs[n].replay for n in outs})
            iters = args.eager_iters if mode == "eager" else args.graph_iters
            rows = []
            for i in range(args.pairs):
                order = ("baseline", "gemv") if i % 2 == 0 else ("gemv", "baseline")
                rows.append({n: events_ms(fns[n], iters) for n in order})
            res[mode] = summarize(rows, "baseline", "gemv")
            rows_off = []
            for i in range(3):
                order = ("baseline", "off") if i % 2 == 0 else ("off", "baseline")
                rows_off.append({n: events_ms(fns[n], iters) for n in order})
            res[mode + "_off_vs_baseline"] = summarize(rows_off, "baseline", "off")
        report["per_M"][M] = res
        print(json.dumps({"M": M, "bits": bits, "served": served,
                          "eager_ms": {k: round(v, 4) for k, v in res["eager"]["median_ms"].items()},
                          "graph_ms": {k: round(v, 4) for k, v in res["graph"]["median_ms"].items()},
                          "graph_speedup": round(res["graph"]["speedup_median"], 4),
                          "eager_speedup": round(res["eager"]["speedup_median"], 4),
                          "off_graph_ratio": round(res["graph_off_vs_baseline"]["speedup_median"], 4)}), flush=True)
        del graphs
    report["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(args.output, "w") as f:
        json.dump(report, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
