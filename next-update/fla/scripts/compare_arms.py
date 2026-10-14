"""Compare fla_nsys.py outputs of the FLA-on and FLA-off captures (structure; under profiler)."""
import json
import sys


def pick(d, chunks):
    for p in d["prefills"]:
        if p["chunks"] == chunks:
            last = p
    return last


def core_calls(p):
    c = p["gdn_core"]
    return {k: (v["n"], v["sum_ms"], v["median_us"]) for k, v in c.items()}


def main(on_path, off_path, out_path):
    on, off = json.load(open(on_path)), json.load(open(off_path))
    rows = {}
    for label, chunks in (("16k", 9), ("120k", 59)):
        a, b = pick(on, chunks), pick(off, chunks)
        win_on, win_off = a["gpu_window_ms"], b["gpu_window_ms"]
        row = {
            "gpu_window_ms": {"on": win_on, "off": win_off, "delta_ms": round(win_on - win_off, 1),
                              "ratio": round(win_on / win_off, 4)},
            "compute_busy_ms": {"on": a["compute_busy_ms"], "off": b["compute_busy_ms"],
                                "delta_ms": round(a["compute_busy_ms"] - b["compute_busy_ms"], 1)},
            "gdn_core_sum_ms": {"on": a["gdn_core_sum_ms"], "off": b["gdn_core_sum_ms"],
                                "delta_ms": round(a["gdn_core_sum_ms"] - b["gdn_core_sum_ms"], 1),
                                "share_of_window_off": round(b["gdn_core_sum_ms"] / win_off, 4),
                                "share_of_window_on": round(a["gdn_core_sum_ms"] / win_on, 4)},
            "non_gdn_core_busy_ms": {"on": round(a["compute_busy_ms"] - a["gdn_core_sum_ms"], 1),
                                     "off": round(b["compute_busy_ms"] - b["gdn_core_sum_ms"], 1)},
            "idle_host_late_ms": {"on": a["idle_host_late_ms"], "off": b["idle_host_late_ms"]},
            "idle_dependency_ms": {"on": a["idle_dependency_ms"], "off": b["idle_dependency_ms"]},
            "fused_calls": {"on": a["fused_calls"], "off": b["fused_calls"]},
            "triton_chunk_o_calls": {"on": a["triton_chunk_o_calls"], "off": b["triton_chunk_o_calls"]},
            "core_kernels_on": core_calls(a), "core_kernels_off": core_calls(b),
            "gdn_tasks_on": a["gdn_layer_tasks"], "gdn_tasks_off": b["gdn_layer_tasks"],
            "copy_h2d_on": a["copy_h2d"], "copy_h2d_off": b["copy_h2d"],
            "host_blocking_on": a["host_blocking_apis"], "host_blocking_off": b["host_blocking_apis"],
            "kkt_grid_hist_on": a["kkt_grid_x_hist (64-row chunks per GDN call)"],
        }
        rows[label] = row
    text = json.dumps({"timings_under_profiler": True, "note": "structure only, not speed evidence",
                       "on": on_path, "off": off_path, "by_length": rows}, indent=1)
    open(out_path, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:4])
