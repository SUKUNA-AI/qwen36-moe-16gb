#!/usr/bin/env python3
"""CPU-only SASS checks of the fused FLA kernels in libqwen36_gdn (cuobjdump -sass output):
  * every MUFU.EX2 Rd, Rs is the non-ftz ex2.approx expansion that Triton's kernels also get
    (dataflow-matched: FSETP.GEU.AND Pk, PT, Rs, -126 before, @!Pk FMUL Rs, Rs, 0.5 before,
    @!Pk FMUL Rd, Rd, Rd after);
  * no local-memory traffic (LDL/STL) and no FTZ arithmetic (.FTZ) in the kernel;
  * static instruction counts (HMMA.16816.F32.BF16 etc.; all warp branches, loops not unrolled).
Usage: sass_check.py qwen36_gdn.sass out.json"""
import json
import re
import sys

text = open(sys.argv[1], errors="replace").read()
funcs = re.split(r"\n\s*Function : ", text)[1:]
res = {}
for f in funcs:
    name = f.splitlines()[0].strip()
    if "chunk_ho_fused_kernel" not in name and "chunk_wu_ho_fused_kernel" not in name:
        continue
    short = "fused_wu_chunk_ho" if "chunk_wu_ho_fused_kernel" in name else "fused_chunk_ho"
    ins = []
    for line in f.splitlines():
        m = re.match(r"\s+/\*[0-9a-f]+\*/\s+(.*?);", line)
        if m:   # drop scheduling annotations and operand-reuse flags
            ins.append(re.sub(r"\s+(\?\w+|&\S+)", "", m.group(1)).replace(".reuse", "").strip())
    ops = [re.sub(r"^@!?U?P\w+\s+", "", s).split()[0] for s in ins]
    mufu = matched = 0
    for i, s in enumerate(ins):
        m = re.match(r"MUFU\.EX2 (R\d+), (R\d+)$", s)
        if not m:
            continue
        mufu += 1
        d, src = m.group(1), m.group(2)
        guard = half = post = None
        for j in range(i - 1, max(0, i - 80), -1):
            mm = re.match(r"FSETP\.GEU\.AND (P\d), PT, %s, -126, PT$" % src, ins[j])
            if mm:
                guard = mm.group(1)
                break
        if guard:
            half = any(ins[j] == f"@!{guard} FMUL {src}, {src}, 0.5" for j in range(max(0, i - 80), i))
            post = any(ins[j] == f"@!{guard} FMUL {d}, {d}, {d}" for j in range(i + 1, min(len(ins), i + 40)))
        matched += bool(guard and half and post)
    res[short] = {
        "instructions": len(ins),
        "HMMA.16816.F32.BF16": sum(o == "HMMA.16816.F32.BF16" for o in ops),
        "MUFU.EX2": mufu, "MUFU.EX2_with_non_ftz_expansion": matched,
        "ex2_non_ftz_expansion_complete": mufu == matched and mufu > 0,
        "local_memory_insns": sum(o.startswith(("LDL", "STL")) for o in ops),
        "ftz_insns": sum(".FTZ" in o for o in ops),
        "FFMA": sum(o == "FFMA" for o in ops), "FMUL": sum(o == "FMUL" for o in ops),
        "FADD": sum(o == "FADD" for o in ops),
        "LDSM": sum(o.startswith("LDSM") for o in ops), "STSM": sum(o.startswith("STSM") for o in ops),
        "LDGSTS": sum(o.startswith("LDGSTS") for o in ops),
        "BAR.SYNC": sum(o.startswith("BAR.SYNC") for o in ops),
    }
json.dump(res, open(sys.argv[2], "w"), indent=1)
print(json.dumps(res, indent=1))
ok = len(res) == 2 and all(r["ex2_non_ftz_expansion_complete"] and r["local_memory_insns"] == 0
                           and r["ftz_insns"] == 0 for r in res.values())
print("SASS_CHECK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
