"""FLA on/off structure analysis of one Nsight Systems SQLite export (layer-major prefill).

Diagnostic only: timings are taken under a profiler and are NOT speed evidence. Kernel sums are
taken on the compute stream only (one stream = serialised, no overlap); cross-stream activity is
reported as interval unions, never summed.

For every layer-major 'Qwen36/prefill' NVTX range (one per long prompt) it reports:
  * chunks (= Layer_0 sub-ranges), GPU window, compute-stream busy union and idle gaps split into
    host-late gaps (the next op was launched after the stream went idle: CPU/launch bound) and
    dependency gaps (launched earlier but waited, e.g. on the expert-copy event);
  * GDN core kernels (kkt_solve / recompute_w_u / chunk_h / chunk_o / fused CUDA) per call;
  * fused vs Triton fallback call counts (FLA path engagement) and per-call T (chunk rows);
  * per GDN-layer task: host (NVTX) duration vs GPU kernel time, core share;
  * copy engine: H2D bytes and busy union inside the window; host-blocking API calls.
Usage: python fla_nsys.py report.sqlite [--label fla-on] [--out result.json]
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import re
import sqlite3
import statistics

CORE = {
    "kkt": re.compile(r"chunk_gated_delta_rule_fwd_kkt_solve_kernel"),
    "wu": re.compile(r"recompute_w_u_fwd_kernel"),
    "h": re.compile(r"chunk_gated_delta_rule_fwd_kernel_h"),
    "o": re.compile(r"chunk_fwd_kernel_o"),
    "fused_wu_ho": re.compile(r"chunk_wu_ho_fused_kernel"),
    "fused_ho": re.compile(r"chunk_ho_fused_kernel"),
}
GDN_CUDA = {
    "conv_qkv": re.compile(r"gdn_conv_qkv_kernel"),
    "gate_cumsum": re.compile(r"gdn_gate_cumsum_kernel"),
    "gated_norm": re.compile(r"gdn_gated_norm_kernel"),
}
SYNC_APIS = re.compile(r"^(cudaStreamSynchronize|cudaDeviceSynchronize|cudaEventSynchronize|cudaMemcpy(_v\d+)?|"
                       r"cudaStreamWaitEvent|cuStreamSynchronize|cuCtxSynchronize|cuEventSynchronize|cuMemcpyDtoH.*)")
LAYER_RE = re.compile(r"^Layer_(\d+)$")


def merged(iv):
    out = []
    for s, e in sorted(iv):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def union_ns(iv):
    return sum(e - s for s, e in merged(iv))


def ms(ns):
    return round(ns / 1e6, 3)


def stats(vals_ns):
    if not vals_ns:
        return None
    v = sorted(vals_ns)
    return {"n": len(v), "sum_ms": ms(sum(v)), "mean_us": round(sum(v) / len(v) / 1e3, 1),
            "median_us": round(statistics.median(v) / 1e3, 1), "p10_us": round(v[len(v) // 10] / 1e3, 1),
            "p90_us": round(v[(len(v) * 9) // 10] / 1e3, 1)}


def cols(conn, table):
    return {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}


def load(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    S = dict(conn.execute("SELECT id, value FROM StringIds"))
    kc = cols(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
    name_col = "demangledName" if "demangledName" in kc else "shortName"
    kernels = [dict(start=s, end=e, stream=st, corr=c, name=S.get(n, ""), grid=(gx, gy, gz))
               for s, e, st, c, n, gx, gy, gz in conn.execute(
                   f'SELECT start,end,streamId,correlationId,{name_col},gridX,gridY,gridZ '
                   f'FROM CUPTI_ACTIVITY_KIND_KERNEL')]
    memcpy = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        memcpy = [dict(start=s, end=e, stream=st, corr=c, bytes=b, kind=k)
                  for s, e, st, c, b, k in conn.execute(
                      "SELECT start,end,streamId,correlationId,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY")]
    memset = []
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        memset = [dict(start=s, end=e, stream=st, corr=c, bytes=b)
                  for s, e, st, c, b in conn.execute(
                      "SELECT start,end,streamId,correlationId,bytes FROM CUPTI_ACTIVITY_KIND_MEMSET")]
    apis = []
    for t in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if t in tables:
            apis += [dict(start=s, end=e, tid=g, corr=c, name=S.get(n, ""))
                     for s, e, g, c, n in conn.execute(f"SELECT start,end,globalTid,correlationId,nameId FROM {t}")]
    nc = cols(conn, "NVTX_EVENTS")
    text = "COALESCE(text, (SELECT value FROM StringIds WHERE id=textId))" if "textId" in nc else "text"
    nvtx = [dict(start=s, end=e, tid=g, name=n or "")
            for s, e, g, n in conn.execute(f"SELECT start,end,globalTid,{text} FROM NVTX_EVENTS "
                                           f"WHERE end IS NOT NULL AND end > start")]
    conn.close()
    return kernels, memcpy, memset, apis, nvtx


def analyze(path, label):
    kernels, memcpy, memset, apis, nvtx = load(path)
    api_by_corr = {a["corr"]: a for a in apis if a["corr"]}
    ops = [dict(k, op="kernel") for k in kernels] + [dict(m, op="memcpy") for m in memcpy] + \
          [dict(m, op="memset") for m in memset]
    for o in ops:
        o["api"] = api_by_corr.get(o["corr"])
    prefills = sorted([n for n in nvtx if n["name"] == "Qwen36/prefill"], key=lambda n: n["start"])
    layers = sorted([n for n in nvtx if LAYER_RE.match(n["name"])], key=lambda n: n["start"])
    layer_starts = [n["start"] for n in layers]
    results = []
    for idx, r in enumerate(prefills):
        lo = bisect.bisect_left(layer_starts, r["start"])
        hi = bisect.bisect_right(layer_starts, r["end"])
        sub = [n for n in layers[lo:hi] if n["tid"] == r["tid"] and n["end"] <= r["end"]]
        chunks = sum(1 for n in sub if n["name"] == "Layer_0")
        if chunks < 2:
            continue  # not layer-major (short prompt)
        mine = [o for o in ops if o["api"] is not None and o["api"]["tid"] == r["tid"]
                and r["start"] <= o["api"]["start"] <= r["end"]]
        if not mine:
            continue
        g0 = min(o["start"] for o in mine)
        g1 = max(o["end"] for o in mine)
        kern = [o for o in mine if o["op"] == "kernel"]
        per_stream = collections.defaultdict(int)
        for k in kern:
            per_stream[k["stream"]] += k["end"] - k["start"]
        compute = max(per_stream, key=per_stream.get)
        cs = sorted([o for o in mine if o["stream"] == compute], key=lambda o: o["start"])
        busy = union_ns([(o["start"], o["end"]) for o in cs])
        host_gap = dep_gap = 0
        host_gaps, dep_gaps = [], []
        for a, b in zip(cs, cs[1:]):
            gap = b["start"] - a["end"]
            if gap <= 0:
                continue
            launch_end = b["api"]["end"] if b["api"] else b["start"]
            if launch_end > a["end"]:
                # launched after the stream drained: the part after the launch is driver latency,
                # the part before it is host/CPU time
                host_gap += gap
                host_gaps.append(gap)
            else:
                dep_gap += gap
                dep_gaps.append(gap)
        # GDN core per name (compute stream only)
        core = {k: [] for k in CORE}
        core_T = collections.Counter()
        for k in kern:
            if k["stream"] != compute:
                continue
            for key, rx in CORE.items():
                if rx.search(k["name"]):
                    core[key].append(k["end"] - k["start"])
        gdn_cuda = {k: [] for k in GDN_CUDA}
        for k in kern:
            if k["stream"] == compute:
                for key, rx in GDN_CUDA.items():
                    if rx.search(k["name"]):
                        gdn_cuda[key].append(k["end"] - k["start"])
        # chunk rows per GDN call: kkt_solve grid.x = number of 64-row chunks of the call
        for k in kern:
            if k["stream"] == compute and CORE["kkt"].search(k["name"]):
                core_T[k["grid"][0]] += 1
        fused_calls = len(core["fused_wu_ho"]) + len(core["fused_ho"])
        triton_calls = len(core["o"])
        # per GDN-layer task (Layer_N NVTX on the prefill thread; attention layers are grouped
        # outside Layer_N under attention batching)
        api_starts = sorted((o["api"]["start"], i) for i, o in enumerate(mine))
        astarts = [x[0] for x in api_starts]
        task_rows = []
        for n in sub:
            m = LAYER_RE.match(n["name"])
            layer = int(m.group(1))
            a = bisect.bisect_left(astarts, n["start"])
            b = bisect.bisect_right(astarts, n["end"])
            tk = [mine[api_starts[j][1]] for j in range(a, b)]
            tk = [o for o in tk if o["stream"] == compute and o["op"] == "kernel"]
            gpu = sum(o["end"] - o["start"] for o in tk)
            coreg = sum(o["end"] - o["start"] for o in tk if any(rx.search(o["name"]) for rx in CORE.values()))
            span = (max(o["end"] for o in tk) - min(o["start"] for o in tk)) if tk else 0
            task_rows.append(dict(layer=layer, host=n["end"] - n["start"], gpu=gpu, core=coreg, span=span,
                                  launches=len(tk)))
        gdn_tasks = [t for t in task_rows if t["layer"] % 4 != 3]
        host_apis = [a for a in apis if a["tid"] == r["tid"] and r["start"] <= a["start"] <= r["end"]
                     and SYNC_APIS.match(a["name"])]
        sync_by = collections.defaultdict(lambda: [0, 0])
        for a in host_apis:
            sync_by[a["name"]][0] += 1
            sync_by[a["name"]][1] += a["end"] - a["start"]
        # copy engine inside the GPU window (all memcpys, any thread)
        h2d = [m for m in memcpy if m["kind"] == 1 and m["end"] > g0 and m["start"] < g1]
        copy_busy = union_ns([(max(m["start"], g0), min(m["end"], g1)) for m in h2d])
        # top kernel families on the compute stream
        fam = collections.defaultdict(int)
        famn = collections.Counter()
        for k in kern:
            if k["stream"] == compute:
                key = re.sub(r"<.*", "", k["name"])[:90]
                fam[key] += k["end"] - k["start"]
                famn[key] += 1
        top = sorted(fam.items(), key=lambda x: -x[1])[:25]
        res = {
            "index": idx, "chunks": chunks, "host_range_ms": ms(r["end"] - r["start"]),
            "gpu_window_ms": ms(g1 - g0), "compute_stream": compute,
            "compute_busy_ms": ms(busy), "compute_busy_frac": round(busy / max(1, g1 - g0), 4),
            "idle_host_late_ms": ms(host_gap), "idle_host_late_gaps": len(host_gaps),
            "idle_host_late_gt50us": sum(1 for g in host_gaps if g > 50_000),
            "idle_dependency_ms": ms(dep_gap), "idle_dependency_gaps": len(dep_gaps),
            "idle_dependency_top_ms": [ms(g) for g in sorted(dep_gaps, reverse=True)[:5]],
            "idle_host_late_top_ms": [ms(g) for g in sorted(host_gaps, reverse=True)[:5]],
            "launches_compute_stream": len(cs),
            "gdn_core": {k: stats(v) for k, v in core.items() if v},
            "gdn_core_sum_ms": ms(sum(sum(v) for v in core.values())),
            "gdn_cuda_other": {k: stats(v) for k, v in gdn_cuda.items() if v},
            "fused_calls": fused_calls, "triton_chunk_o_calls": triton_calls,
            "kkt_grid_x_hist (64-row chunks per GDN call)": dict(sorted(core_T.items())),
            "gdn_layer_tasks": {
                "n": len(gdn_tasks),
                "host_ms_median": ms(statistics.median([t["host"] for t in gdn_tasks])) if gdn_tasks else None,
                "gpu_kernel_ms_median": ms(statistics.median([t["gpu"] for t in gdn_tasks])) if gdn_tasks else None,
                "gpu_span_ms_median": ms(statistics.median([t["span"] for t in gdn_tasks])) if gdn_tasks else None,
                "core_ms_median": ms(statistics.median([t["core"] for t in gdn_tasks])) if gdn_tasks else None,
                "host_gt_gpu_tasks": sum(1 for t in gdn_tasks if t["host"] > t["gpu"]),
                "sum_host_ms": ms(sum(t["host"] for t in gdn_tasks)),
                "sum_gpu_kernel_ms": ms(sum(t["gpu"] for t in gdn_tasks)),
                "sum_core_ms": ms(sum(t["core"] for t in gdn_tasks)),
                "launches_median": statistics.median([t["launches"] for t in gdn_tasks]) if gdn_tasks else None,
            },
            "attention_layer_tasks_in_layer_nvtx": sum(1 for t in task_rows if t["layer"] % 4 == 3),
            "host_blocking_apis": {k: {"n": v[0], "ms": ms(v[1])} for k, v in sorted(sync_by.items())},
            "copy_h2d": {"n": len(h2d), "GB": round(sum(m["bytes"] for m in h2d) / 1e9, 3),
                         "busy_union_ms": ms(copy_busy)},
            "top_compute_kernels": [{"name": k, "n": famn[k], "sum_ms": ms(v)} for k, v in top],
        }
        results.append(res)
    return {"label": label, "sqlite": str(path), "timings_under_profiler": True,
            "note": "structure only; not speed evidence", "prefills": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = analyze(a.sqlite, a.label)
    text = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    for p in res["prefills"]:
        print(json.dumps({k: p[k] for k in ("index", "chunks", "gpu_window_ms", "compute_busy_ms", "compute_busy_frac",
                                            "idle_host_late_ms", "idle_dependency_ms", "gdn_core_sum_ms",
                                            "fused_calls", "triton_chunk_o_calls")}))


if __name__ == "__main__":
    main()
