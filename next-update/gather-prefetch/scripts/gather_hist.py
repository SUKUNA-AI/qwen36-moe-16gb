"""Histogram of decode_gather_tma durations per decode step class + per-layer windows (f1 nsys sqlite).
Outputs JSON with: duration histogram (5 us bins), per-step gather sum, shared-expert window per layer
(router GEMV end -> topk start), and the inferred per-call miss counts via duration clustering."""
import sqlite3, sys, bisect, collections, json, statistics
db, out = sys.argv[1], sys.argv[2]
con = sqlite3.connect(db)
S = dict(con.execute("select id, value from StringIds"))
nv = [(s, e, (t if t is not None else S.get(tid, ""))) for s, e, t, tid in
      con.execute("select start, end, text, textId from NVTX_EVENTS where end is not null")]
dec = sorted((s, e) for s, e, t in nv if t == "Qwen36/decode")
pref = sorted((s, e) for s, e, t in nv if t == "Qwen36/prefill")
api = {c: s for c, s in con.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME")}
ks = con.execute("select start, end, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
dstarts = [s for s, e in dec]
pstarts = [s for s, e in pref]
labels = ["warmup_short", "warmup_chunk", "warm_decode", "fresh_16k", "fresh_120k", "filler"]
per = collections.defaultdict(list)
for s, e, c, sn in ks:
    t = api.get(c)
    if t is None: continue
    i = bisect.bisect_right(dstarts, t) - 1
    if i >= 0 and dec[i][0] <= t <= dec[i][1]:
        per[i].append((s, e, S.get(sn, "")))
res = {}
for lab in labels:
    res[lab] = {"gather_us": [], "step_us": [], "shared_window_us": [], "ensure_us": [], "gather_per_step": [],
                "gather_call_us_by_layer": collections.defaultdict(list), "gather_start_after_router_us": []}
for i, kl in per.items():
    lab = labels[bisect.bisect_right(pstarts, dec[i][0]) - 1]
    kl.sort()
    r = res[lab]
    r["step_us"].append((kl[-1][1] - kl[0][0]) / 1e3)
    tot = 0.0
    layer = -1
    # walk: router GEMV is the 'kernel' with name 'kernel' right after the post-attn norm (flashinf_3)
    last_norm_end = None
    for j, (s, e, n) in enumerate(kl):
        if n.startswith("kernel_cutlass_kernel_flashinf"):
            last_norm_end = e
        if n == "_router_triton_kernel":
            if last_norm_end is not None:
                # time from the post-attention norm end to the top-k start = router GEMV + shared expert block
                r["shared_window_us"].append((s - last_norm_end) / 1e3)
        if n == "_lru_ensure_kernel":
            r["ensure_us"].append((e - s) / 1e3)
        if n == "decode_gather_tma_kernel":
            layer += 1
            d = (e - s) / 1e3
            r["gather_us"].append(d)
            r["gather_call_us_by_layer"][layer].append(d)
            tot += d
    r["gather_per_step"].append(tot)
summary = {}
for lab, r in res.items():
    g = r["gather_us"]
    if not g: continue
    hist = collections.Counter(int(x // 10) * 10 for x in g)
    miss = [x for x in g if x > 5]
    summary[lab] = {
        "steps": len(r["step_us"]),
        "step_us_median": statistics.median(r["step_us"]),
        "gather_calls": len(g),
        "gather_calls_with_misses": len(miss),
        "gather_us_sum_per_step_mean": sum(g) / len(r["step_us"]),
        "zero_miss_call_us_median": statistics.median([x for x in g if x <= 5]) if any(x <= 5 for x in g) else None,
        "miss_call_us_quantiles": [round(q, 1) for q in statistics.quantiles(miss, n=10)] if len(miss) > 10 else None,
        "miss_call_us_mean": statistics.mean(miss) if miss else None,
        "hist_10us": dict(sorted(hist.items())),
        "router_plus_shared_window_us_median": statistics.median(r["shared_window_us"]) if r["shared_window_us"] else None,
        "lru_ensure_us_median": statistics.median(r["ensure_us"]) if r["ensure_us"] else None,
        "per_layer_gather_us_mean": {k: round(statistics.mean(v), 1) for k, v in sorted(r["gather_call_us_by_layer"].items())},
    }
json.dump(summary, open(out, "w"), indent=1)
for lab, s in summary.items():
    print(lab, {k: v for k, v in s.items() if k not in ("per_layer_gather_us_mean", "hist_10us")})
    print("  hist", s["hist_10us"])
