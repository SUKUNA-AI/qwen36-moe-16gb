"""D3a: replay route traces (freetoken/moe/route_trace.py) through the exact flashlib LRU.

Per decode request the trace holds the LRU state at its first step and, per (step, layer), the
real routing, which of it missed, and the pre-gate prediction (layer L+1's router on layer L's
output). The replay:
  1. re-runs the real routing through a numpy copy of flashlib's lru_ensure from the snapshot and
     requires its misses to equal the observed ones (otherwise the trace or the model is wrong);
  2. coverage = |predicted & missed| / |missed| over layers >= 1 (layer 0 has no previous layer);
  3. replays again with a speculative ensure(predicted) before every real ensure (what D3b would
     do), counting the misses left on the critical path and the extra PCIe rows fetched.

Usage: d3a_replay.py TRACE_DIR META_JSON OUT_JSON
"""
import json
import pathlib
import sys

import numpy as np


class Lru:
    """flashlib.kernels.slot_cache lru_ensure: clock +1 per call; hits get usage=step; distinct
    misses in ascending id order take victims in ascending (usage, slot) among slots not touched
    by this call."""

    def __init__(self, slot_for_id, id_of_slot, usage, step):
        self.slot_of_id = slot_for_id.reshape(-1).astype(np.int64).copy()
        self.id_of_slot = id_of_slot.astype(np.int64).copy()
        self.usage = usage.astype(np.int64).copy()
        self.step = int(step)

    def ensure(self, ids, base):
        self.step += 1
        q = np.unique(ids[ids >= 0].astype(np.int64)) + base
        slots = self.slot_of_id[q]
        self.usage[slots[slots >= 0]] = self.step
        miss = q[slots < 0]
        if miss.size:
            order = np.argsort(self.usage, kind="stable")
            order = order[self.usage[order] != self.step][:miss.size]
            for e, v in zip(miss, order):
                old = self.id_of_slot[v]
                if old >= 0:
                    self.slot_of_id[old] = -1
                self.id_of_slot[v] = e
                self.slot_of_id[e] = v
                self.usage[v] = self.step
        return set((miss - base).tolist())

    def resident(self, ids, base):
        return set(int(i) for i in ids if i >= 0 and self.slot_of_id[int(i) + base] >= 0)


def replay(path):
    z = np.load(path)
    actual, missed, pred = z["actual"], z["missed"], z["predicted"]
    E = int(z["num_experts"])
    n, L, _ = actual.shape
    base = Lru(z["slot_for_id"], z["id_of_slot"], z["usage"], z["step"])
    spec = Lru(z["slot_for_id"], z["id_of_slot"], z["usage"], z["step"])
    r = dict(steps=n, mismatched_layers=0, misses=0, misses_l1=0, covered=0, actual_l1=0, predicted_hits_actual=0,
             spec_fetched=0, spec_wasted=0, residual_misses=0, spec_window_rows=[])
    per_layer_miss = np.zeros(L, np.int64)
    per_layer_cov = np.zeros(L, np.int64)
    for s in range(n):
        for layer in range(L):
            a = actual[s, layer]
            if (a < 0).all():
                continue
            b = layer * E
            m = base.ensure(a, b)
            observed = set(int(x) for x in a[missed[s, layer]])
            if m != observed:
                r["mismatched_layers"] += 1
            r["misses"] += len(m)
            p = pred[s, layer]
            if layer >= 1 and (p >= 0).any():
                ps = set(int(x) for x in p if x >= 0)
                aset = set(int(x) for x in a)
                r["misses_l1"] += len(m)
                r["covered"] += len(ps & m)
                r["actual_l1"] += len(aset)
                r["predicted_hits_actual"] += len(ps & aset)
                per_layer_miss[layer] += len(m)
                per_layer_cov[layer] += len(ps & m)
                fetched = spec.ensure(p, b)
                r["spec_fetched"] += len(fetched)
                r["spec_wasted"] += len(fetched - aset)
                r["spec_window_rows"].append(len(fetched))
                r["residual_misses"] += len(spec.ensure(a, b))
            else:
                r["residual_misses"] += len(spec.ensure(a, b))
    w = np.array(r.pop("spec_window_rows") or [0])
    r["spec_rows_per_layer"] = {"mean": float(w.mean()), "p50": float(np.median(w)), "p90": float(np.percentile(w, 90)),
                                "max": int(w.max())}
    r["per_layer_misses"] = per_layer_miss.tolist()
    r["per_layer_covered"] = per_layer_cov.tolist()
    return r


def summarize(rows):
    keys = ("steps", "mismatched_layers", "misses", "misses_l1", "covered", "actual_l1", "predicted_hits_actual",
            "spec_fetched", "spec_wasted", "residual_misses")
    t = {k: sum(r[k] for r in rows) for k in keys}
    t["coverage"] = t["covered"] / t["misses_l1"] if t["misses_l1"] else None
    t["routing_recall"] = t["predicted_hits_actual"] / t["actual_l1"] if t["actual_l1"] else None
    t["misses_per_token"] = t["misses"] / t["steps"] if t["steps"] else None
    t["residual_misses_per_token"] = t["residual_misses"] / t["steps"] if t["steps"] else None
    t["spec_fetched_per_token"] = t["spec_fetched"] / t["steps"] if t["steps"] else None
    t["spec_wasted_per_token"] = t["spec_wasted"] / t["steps"] if t["steps"] else None
    t["spec_precision"] = (t["spec_fetched"] - t["spec_wasted"]) / t["spec_fetched"] if t["spec_fetched"] else None
    return t


def main():
    trace_dir, meta_path, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    meta = json.loads(meta_path.read_text())
    labels = ["warm"] + [f"{r['case']}-{r.get('prompt_tokens')}" for r in meta.get("requests", [])]
    rows = []
    for i, path in enumerate(sorted(trace_dir.glob("req*.npz"))):
        r = replay(path)
        r["request"] = labels[i] if i < len(labels) else path.stem
        rows.append(r)
    measured = [r for r in rows if r["request"] != "warm"]
    result = {"schema": 1, "what": "D3a pre-gate coverage under exact LRU replay", "requests": rows,
              "total": summarize(measured),
              "by_context": {c: summarize([r for r in measured if r["request"].startswith(c)])
                             for c in ("short", "4k", "16k")},
              "decision_rule": "D3b only if coverage >= 0.6 (and T3 qualifies)"}
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["total"], indent=1))
    print(json.dumps({k: v["coverage"] for k, v in result["by_context"].items()}))


if __name__ == "__main__":
    main()
