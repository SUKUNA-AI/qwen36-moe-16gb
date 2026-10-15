"""Structure probes for cuBLASLt's tensor-wise FP8 split-K GEMM (decode out/o-projection, N=2048,
K=4096). GPU; run under the GPU lock.

Absorption probes: every output row n is a sparse dot product. Big products P = 448*448 = 200704
(fp32 ulp 2^-6) and a tiny product s = 2^-9 * 2^-9 = 2^-18. s survives to the output only if it is
never added (with an FP32 rounding) to an accumulator that holds +-P before +P and -P cancel.

  probe "pairs": x = 2^-9 on tile 0 (k 0..63), 448 on tile T (k 64T..64T+63), 0 elsewhere.
     row (a, d): w[0] = 2^-9 (s), w[64T+a] = +448, w[64T+d] = -448  -> survives iff a and d are
     cancelled before they meet the accumulator, i.e. iff they are in the same MMA K-set (if a K-set
     is summed exactly). The 64x64 matrix gives the K-set partition of a 64-byte k-tile.
  probe "tiles": x = 2^-9 on tile 0, 448 on tiles 1..63. row (i, j), i < j: w[0] = 2^-9, w[64i] =
     +448, w[64j] = -448. survives iff +P and -P cancel before s is merged: reveals which tiles share
     a split-K chain and the order the partial sums are reduced in.
  probe "tiles_last": same with s placed in the last tile (63) instead of tile 0.

Usage: python probe_splitk.py --out F [--tile 1] [--M 1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identify_splitk import FP8, K_IN, N_OUT, reference  # noqa: E402

E448, EM448, ETINY = 0x7E, 0xFE, 0x01  # e4m3 codes: +448, -448, 2^-9


def run(x_codes, rows, M=1):
    """rows: list of {k: code}; returns per-row float of the bf16 output (row m=0)."""
    dev = torch.device("cuda", 0)
    w = torch.zeros((N_OUT, K_IN), dtype=torch.uint8)
    for n, entries in enumerate(rows):
        for k, c in entries.items():
            w[n, k] = c
    x = torch.zeros((M, K_IN), dtype=torch.uint8)
    x[:] = torch.tensor(x_codes, dtype=torch.uint8)
    w = w.to(dev).view(FP8)
    x = x.to(dev).view(FP8)
    one = torch.tensor(1.0, device=dev)
    out = reference(x, w, one, one).float().cpu()
    return out  # [M, N]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=1)
    ap.add_argument("--M", type=int, default=1)
    args = ap.parse_args()
    res = {}
    s = 2.0 ** -18
    # ---- pairs inside one tile
    T = args.tile
    x = [0] * K_IN
    for k in range(64):
        x[k] = ETINY
        x[64 * T + k] = E448
    pairs = [(a, d) for a in range(64) for d in range(64) if a < d]
    rows = [{0: ETINY, 64 * T + a: E448, 64 * T + d: EM448} for a, d in pairs]
    out = run(x, rows, args.M)
    surv = {}
    for idx, (a, d) in enumerate(pairs):
        surv[(a, d)] = float(out[0, idx]) / s
    mat = [["." for _ in range(64)] for _ in range(64)]
    for (a, d), v in surv.items():
        ch = "1" if v == 1.0 else ("0" if v == 0.0 else "?")
        mat[a][d] = mat[d][a] = ch
    res["pairs_matrix"] = ["".join(r) for r in mat]
    res["pairs_values"] = sorted({v for v in surv.values()})
    # K-set classes (connected by "1")
    parent = list(range(64))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (a, d), v in surv.items():
        if v == 1.0:
            parent[find(a)] = find(d)
    classes = {}
    for k in range(64):
        classes.setdefault(find(k), []).append(k)
    res["pairs_classes"] = sorted(classes.values())
    print("pairs values (x s):", res["pairs_values"])
    print("K-set classes in tile", T, ":", res["pairs_classes"])
    # ---- tiles
    for name, sk in (("tiles", 0), ("tiles_last", 63 * 64 + 1)):
        x = [E448] * K_IN
        stile = sk // 64
        for k in range(64):
            x[64 * stile + k] = ETINY
        tiles = [t for t in range(64) if t != stile]
        tp = [(i, j) for i in tiles for j in tiles if i < j]
        out_all = []
        for start in range(0, len(tp), N_OUT):
            chunk = tp[start:start + N_OUT]
            rows = [{sk: ETINY, 64 * i: E448, 64 * j: EM448} for i, j in chunk]
            rows += [{}] * (N_OUT - len(rows))
            o = run(x, rows, args.M)
            out_all += [float(o[0, r]) / s for r in range(len(chunk))]
        tm = [["." for _ in range(64)] for _ in range(64)]
        vals = set()
        for (i, j), v in zip(tp, out_all):
            vals.add(v)
            ch = "1" if v == 1.0 else ("0" if v == 0.0 else "?")
            tm[i][j] = tm[j][i] = ch
        res[name + "_matrix"] = ["".join(r) for r in tm]
        res[name + "_values"] = sorted(vals)
        print(name, "values:", sorted(vals))
        for r in tm:
            print("".join(r))
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("pairs matrix:")
    for r in res["pairs_matrix"]:
        print(r)


if __name__ == "__main__":
    main()
