"""D3b: replay the S4 route traces (moe/route_trace.py) through the exact flashlib LRU with the
speculative plan moe/pregate.py issues, for several per-layer caps, and estimate the copy time left
on the critical path.

Speculative query (as pregate's select kernel builds it): the first ``cap`` predicted misses in score
order, padded with the first of them, or with the top-1 prediction when none misses. Time model per
layer: a copy of n experts costs C0 + C1 * n us; a speculative copy is hidden up to the window W of
the mixer (minus the predictor/LRU launches), the rest stalls the join; real misses cost in full.

Usage: d3b_replay.py TRACE_DIR OUT_JSON  (TRACE_DIR: req*.npz from QWEN36_ROUTE_TRACE; req0000 = warm-up)
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "transport/scripts"))
from d3a_replay import Lru  # noqa: E402

C0, C1 = 5.0, 42.0  # us: gather launch + one 1.78 MB NVFP4 expert at ~42 GB/s


def tcopy(n):
    return C0 + C1 * n if n else 0.0


def attention_layer(layer):
    return layer % 4 == 3  # Qwen3.6-35B-A3B: full attention every 4th layer


def replay(path, cap_linear, cap_attn, window_linear, window_attn):
    z = np.load(path)
    actual, pred = z["actual"], z["predicted"]
    E = int(z["num_experts"])
    n, L, _ = actual.shape
    base = Lru(z["slot_for_id"], z["id_of_slot"], z["usage"], z["step"])
    spec = Lru(z["slot_for_id"], z["id_of_slot"], z["usage"], z["step"])
    r = dict(steps=n, misses=0, residual=0, spec=0, wasted=0, t_base=0.0, t_spec=0.0, rows=[])
    for s in range(n):
        for layer in range(L):
            a = actual[s, layer]
            if (a < 0).all():
                continue
            b = layer * E
            m = base.ensure(a, b)
            r["misses"] += len(m)
            r["t_base"] += tcopy(len(m))
            p = [int(x) for x in pred[s, layer] if x >= 0]
            cap = cap_attn if attention_layer(layer) else cap_linear
            if layer >= 1 and p and cap:
                miss = [x for x in p if spec.slot_of_id[x + b] < 0][:cap]
                q = (miss + [miss[0]] * cap)[:max(2, cap)] if miss else [p[0]] * max(2, cap)
                fetched = spec.ensure(np.asarray(q), b)
                window = window_attn if attention_layer(layer) else window_linear
                r["t_spec"] += max(0.0, tcopy(len(fetched)) - window)
                r["spec"] += len(fetched)
                r["wasted"] += len(fetched - set(int(x) for x in a))
                r["rows"].append(len(fetched))
            left = spec.ensure(a, b)
            r["residual"] += len(left)
            r["t_spec"] += tcopy(len(left))
    return r


def main():
    trace_dir, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    paths = sorted(trace_dir.glob("req*.npz"))[1:]
    rows = []
    for caps in ((8, 8), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)):
        for windows in ((55.0, 90.0), (30.0, 60.0), (55.0, 340.0)):
            t = {}
            for path in paths:
                for k, v in replay(path, *caps, *windows).items():
                    t[k] = t.get(k, 0) + v if k != "rows" else t.get(k, []) + v
            st = t["steps"]
            w = np.array(t["rows"] or [0])
            rows.append({"caps_linear_attn": caps, "window_us_linear_attn": windows,
                         "misses_per_token": t["misses"] / st, "critical_misses_per_token": t["residual"] / st,
                         "spec_rows_per_token": t["spec"] / st, "spec_wasted_per_token": t["wasted"] / st,
                         "spec_rows_per_layer_p90": float(np.percentile(w, 90)),
                         "copy_ms_per_token_base": t["t_base"] / st / 1000,
                         "copy_ms_per_token_pregate": t["t_spec"] / st / 1000,
                         "saved_ms_per_token": (t["t_base"] - t["t_spec"]) / st / 1000})
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}))
    out.write_text(json.dumps({"schema": 1, "what": "D3b capped pre-gate replay (exact LRU) and copy-time model",
                               "model": {"C0_us": C0, "C1_us_per_expert": C1}, "traces": [p.name for p in paths],
                               "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
