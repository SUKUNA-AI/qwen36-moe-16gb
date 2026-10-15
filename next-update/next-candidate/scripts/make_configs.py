"""Write the S8 arm configs (s8-ARM.json) for the coordinator's teacher gate and A/B.

Every arm is harness/main-release-next-candidate.json (main128: T1+T2+T3, host embeddings, S3 kernels,
C0, C1) plus only its own variables. D3b needs T3 (QWEN36_DECODE_SHARED_OVERLAP=1, already in the base).
Usage: make_configs.py REPO NATIVE_CONFIG_DIR [--remove]
"""
import json
import os
import pathlib
import sys

D3B = {"QWEN36_DECODE_PREGATE": "1", "QWEN36_DECODE_PREGATE_ATTN": "2"}
D5 = {"QWEN36_DECODE_SHARED_EPILOGUE": "1"}
D6 = {"QWEN36_PREFILL_WARM_DECODE": "8"}
ARMS = {
    "base": {},
    "d3b": D3B,
    "d3b22": {"QWEN36_DECODE_PREGATE": "2", "QWEN36_DECODE_PREGATE_ATTN": "2"},
    "d5": D5,
    "d6": D6,
    "all": {**D3B, **D5, **D6},
}


def main():
    repo, native = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    if "--remove" in sys.argv:
        for arm in ARMS:
            (native / f"s8-{arm}.json").unlink(missing_ok=True)
        return
    base = json.loads((repo / "harness/main-release-next-candidate.json").read_text())
    assert base["env"].get("QWEN36_DECODE_SHARED_OVERLAP") == "1", "D3b rides on T3"
    home = os.path.expanduser("~")
    for arm, extra in ARMS.items():
        cfg = json.loads(json.dumps(base).replace("/home/USER", home))
        cfg["env"].update(extra)
        path = native / f"s8-{arm}.json"
        path.write_text(json.dumps(cfg, indent=1))
        path.chmod(0o600)


if __name__ == "__main__":
    main()
