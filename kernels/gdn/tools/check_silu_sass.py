#!/usr/bin/env python3
"""SiLU + conv FFMA-chain SASS dataflow: sgl_kernel causal_conv1d_{fwd,update} (BF16, width 4)
vs every SiLU instance in libqwen36_gdn (conv_qkv prefill kernels and the decode kernel).
Reuses the data-flow slicer of work/cuda-gdn-layout/sass_compare.py (copied, unmodified)."""
import collections, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import sass_silu_dataflow as S

orig_files, mine, out = sys.argv[1:-2], sys.argv[-2], sys.argv[-1]
res = {"original": {}, "mine": {}}
orig = collections.Counter()
for p in orig_files:
    ins, silu, chains = S.analyse(Path(p).read_text(errors="replace"))
    orig.update(silu)
    res["original"][Path(p).name] = {"silu": len(silu), "distinct": len(set(silu)),
        "chains": dict(collections.Counter(f"len{l}:{i}" for l, i in chains))}
text = Path(mine).read_text(errors="replace")
minec = collections.Counter()
for chunk in re.split(r"\n\s*Function : ", text)[1:]:
    name = chunk.splitlines()[0].strip()
    if not re.search(r"gdn_conv_qkv_kernel|gdn_decode_fused_kernel|gdn_decode_cluster_kernel|silu_(points|compare|eval)", name):
        continue
    short = re.sub(r"^.*?(gdn_conv_qkv_kernel\w*|gdn_decode_\w+?_kernel|silu_\w+?_kernel).*$", r"\1", name)
    ins, silu, chains = S.analyse(chunk)
    minec.update(silu)
    res["mine"][short] = {"silu": len(silu), "distinct": len(set(silu)),
        "chains": dict(collections.Counter(f"len{l}:{i}" for l, i in chains)),
        "local_memory_insns": sum(op.startswith(("LDL", "STL")) for op, _ in ins)}
res["original_dataflows"] = len(set(orig))
res["mine_dataflows"] = len(set(minec))
res["silu_dataflow_identical"] = set(orig) == set(minec) and len(set(orig)) == 1
res["dataflow"] = [list(s) for s in set(minec)]


def canon(df):
    """Scheduling-independent form: expand every slice line into its full expression tree
    (vN -> expression of line N); two dataflows are identical iff their expression sets are."""
    exprs = []
    for line in df:
        op, _, args = line.partition(" ")
        parts = []
        for a in args.split(","):
            neg = a.startswith("-")
            core = a[1:] if neg else a
            if re.fullmatch(r"v\d+", core):
                core = "(" + exprs[int(core[1:])] + ")"
            parts.append(("-" if neg else "") + core)
        exprs.append(op + " " + ",".join(parts))
    return frozenset(exprs)


orig_c = {canon(d) for d in set(orig)}
mine_c = {canon(d) for d in set(minec)}
res["silu_dataflow_identical_modulo_scheduling"] = orig_c == mine_c and len(orig_c) == 1
Path(out).write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("original_dataflows", "mine_dataflows", "silu_dataflow_identical",
                                      "silu_dataflow_identical_modulo_scheduling")}))
for k, v in res["mine"].items(): print(k, v)
