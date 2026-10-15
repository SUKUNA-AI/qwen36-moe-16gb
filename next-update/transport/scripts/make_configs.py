"""Write the S4 arm configs (config/s4-ARM.json) into the experiment overlay.

Every arm is harness/main-release-shortroute.json (T1, the release candidate's transport) plus
only its own switches; ``*-nsys`` arms add the cudaProfilerApi window. Usage:
make_configs.py REPO NATIVE_CONFIG_DIR [--remove]
"""
import json
import os
import pathlib
import sys

ARMS = {
    "t1": {},
    "c0p": {"QWEN36_PREFILL_COPY_PRIORITY": "1"},
    "c0e": {"QWEN36_PREFILL_EPOCH_INVALIDATE": "1"},
    "c0pe": {"QWEN36_PREFILL_COPY_PRIORITY": "1", "QWEN36_PREFILL_EPOCH_INVALIDATE": "1"},
    "t3": {"QWEN36_DECODE_SHARED_OVERLAP": "1"},
    "t3p": {"QWEN36_DECODE_SHARED_OVERLAP": "1", "QWEN36_DECODE_OVERLAP_PRIORITY": "1"},
    "c1": {"QWEN36_PREFILL_SELECTED": "512"},
    "c1pe": {"QWEN36_PREFILL_SELECTED": "512", "QWEN36_PREFILL_COPY_PRIORITY": "1",
             "QWEN36_PREFILL_EPOCH_INVALIDATE": "1"},
    "c1spe": {"QWEN36_PREFILL_SELECTED": "512", "QWEN36_PREFILL_SELECTED_SYNC": "1",
              "QWEN36_PREFILL_COPY_PRIORITY": "1", "QWEN36_PREFILL_EPOCH_INVALIDATE": "1"},
    "c2": {"QWEN36_PREFILL_TMA_GATHER": "1"},
    "c2pe": {"QWEN36_PREFILL_TMA_GATHER": "1", "QWEN36_PREFILL_COPY_PRIORITY": "1",
             "QWEN36_PREFILL_EPOCH_INVALIDATE": "1"},
    "d6pe": {"QWEN36_PREFILL_WARM_DECODE": "8", "QWEN36_PREFILL_COPY_PRIORITY": "1",
             "QWEN36_PREFILL_EPOCH_INVALIDATE": "1"},
    "d3a": {"QWEN36_ROUTE_TRACE": "1", "QWEN36_ROUTE_TRACE_OUT": "~/.cache/qwen36-cloud-review/s4-results/route-trace",
            "_flags": ["--cuda-graph-max-bs", "0"]},
}
NSYS_FORWARDS = "400"


def main():
    repo, native = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    names = [f"s4-{a}" for a in ARMS] + [f"s4-{a}-nsys" for a in ARMS]
    if "--remove" in sys.argv:
        for n in names:
            (native / f"{n}.json").unlink(missing_ok=True)
        return
    base = json.loads((repo / "harness/main-release-shortroute.json").read_text())
    home = os.path.expanduser("~")
    for arm, extra in ARMS.items():
        for suffix, more in (("", {}), ("-nsys", {"QWEN36_NSYS_FORWARDS": NSYS_FORWARDS})):
            cfg = json.loads(json.dumps(base).replace("/home/USER", home))
            cfg["env"].update({k: v.replace("~", home, 1) for k, v in extra.items() if k != "_flags"})
            cfg["flags"] = cfg["flags"] + extra.get("_flags", [])
            cfg["env"].update(more)
            path = native / f"s4-{arm}{suffix}.json"
            path.write_text(json.dumps(cfg, indent=1))
            path.chmod(0o600)


if __name__ == "__main__":
    main()
