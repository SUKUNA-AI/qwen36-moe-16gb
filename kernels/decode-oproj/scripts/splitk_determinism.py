"""Is cuBLASLt's tensor-wise split-K result for the decode out/o-projection a deterministic function of
its inputs?  GPU; run under the GPU lock.

* repeats the same torch._scaled_mm call R times (eager, and replays of one captured graph) and
  counts distinct bf16 outputs;
* for every output element checks which permutation of the 3 split-K partial sums (each summed from
  +0, then times RN(sa*sb), then RN to bf16) reproduces each observed value -- an arrival-order
  (atomic) reduction is explained by "some permutation per element per run", a fixed-order
  reduction by one permutation everywhere;
* records the kernels of the eager call and of the graph (torch.profiler).

Usage: python splitk_determinism.py --analysis-lib <analysis .so> [--bounds 0,1408,2816,4096] --out F
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identify_splitk import FP8, K_IN, N_OUT, Partial, codes, reference  # noqa: E402


def kernels_of(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name[:120] for e in prof.events() if e.device_type.name == "CUDA"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-lib", required=True)
    ap.add_argument("--bounds", default="0,1408,2816,4096")
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    part = Partial(args.analysis_lib)
    b = [int(x) for x in args.bounds.split(",")]
    gen = torch.Generator(device=dev)
    gen.manual_seed(1234)
    w = codes((N_OUT, K_IN), gen, dev)
    sa = torch.tensor(0.0123, device=dev)
    sb = torch.tensor(0.00071, device=dev)
    res = {"bounds": b}
    for M in (1, 2, 4, 8):
        qa = codes((M, K_IN), gen, dev)
        parts = [part(qa, w, k0, k1) for k0, k1 in zip(b[:-1], b[1:])]
        ab = sa * sb
        perms = {}
        for perm in itertools.permutations(range(len(parts))):
            s = torch.zeros_like(parts[0])
            for i in perm:
                s = s + parts[i]
            perms[perm] = (s * ab).to(torch.bfloat16).view(torch.int16)
        outs = [reference(qa, w, sa, sb).view(torch.int16).clone() for _ in range(args.repeats)]
        # graph replays
        s_ = torch.cuda.Stream()
        with torch.cuda.stream(s_):
            reference(qa, w, sa, sb)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s_):
                og = reference(qa, w, sa, sb)
        gouts = []
        for _ in range(args.repeats):
            g.replay()
            gouts.append(og.view(torch.int16).clone())
        torch.cuda.synchronize()
        allo = torch.stack(outs + gouts)
        distinct = torch.unique(allo, dim=0).shape[0]
        explained = torch.zeros_like(allo, dtype=torch.bool)
        per_perm = {}
        for perm, v in perms.items():
            eq = allo == v[None]
            explained |= eq
            per_perm["".join(map(str, perm))] = int(eq.sum())
        first = allo[0]
        res[f"M{M}"] = {
            "runs": int(allo.shape[0]), "distinct_outputs": int(distinct),
            "elements_varying_across_runs": int((allo != first[None]).any(0).sum()),
            "elements_per_run": int(first.numel()),
            "explained_by_some_permutation": int(explained.sum()), "total": int(allo.numel()),
            "matches_per_permutation": per_perm,
            "eager_kernels": kernels_of(lambda: reference(qa, w, sa, sb)),
            "graph_kernels": kernels_of(lambda: g.replay()),
        }
        print(M, json.dumps({k: v for k, v in res[f"M{M}"].items() if "kernels" not in k}), flush=True)
    print(json.dumps(res["M1"]["eager_kernels"]), json.dumps(res["M1"]["graph_kernels"]))
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
