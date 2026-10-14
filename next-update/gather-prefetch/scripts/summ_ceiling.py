"""Summarize ceiling-s1.json: best median GB/s per mechanism family x memory kind x pattern, + today's gather."""
import json, sys, collections
src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
rows = d["rows"]
def fam(m):
    if m.startswith("ce_"): return m
    if m.startswith("tma:8:1:3:16384"): return "gather_today(tma 8x1x3x16K)"
    return m.split(":")[0] + ("_sink" if m.endswith("sink") else "")
best = {}
for r in rows:
    key = (fam(r["mech"]), r["kind"], r["pattern"], r["n"])
    if r["mismatches"]:
        continue
    b = best.get(key)
    if b is None or r["GBps_median"] > b["GBps_median"]:
        best[key] = r
summary = collections.defaultdict(dict)
for (f, kind, pat, n), r in sorted(best.items()):
    summary[f"{f}"][f"{kind}/{pat}{'' if pat=='flat' else n}"] = {
        "GBps_median": round(r["GBps_median"], 2), "GBps_p10best": round(r["GBps_best"], 2),
        "config": r["mech"], "median_us": round(r["median_us"], 1), "inflight_bytes": r["inflight"]}
overall = {}
for f, m in summary.items():
    k, v = max(m.items(), key=lambda kv: kv[1]["GBps_median"])
    overall[f] = {"max_GBps_median": v["GBps_median"], "at": k, "config": v["config"], "p10best": v["GBps_p10best"]}
res = {"source": src, "device": d["device"], "async_engines": d["async_engines"], "overall_max_per_mechanism": overall,
       "per_mechanism": summary, "total_rows": len(rows), "rows_with_mismatches": sum(1 for r in rows if r["mismatches"])}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(overall, indent=1))
print("rows", len(rows), "mismatch rows", res["rows_with_mismatches"])
