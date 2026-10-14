#!/usr/bin/env python3
"""CPU-only SASS comparison: silu + conv-FMA data-flow of the original sgl_kernel
causal_conv1d_fwd (BF16, width 4, sm_120 cubin) vs the fused kernel.

For every silu instance (anchored at FFMA.SAT) a forward data-flow slice is taken from the
conv accumulator x: each instruction that reads a value derived from x is kept, registers are
renamed SSA-style by producing instruction, and registers loaded from immediates (MOV/HFMA2)
are replaced by their constant. The slice ends with the IEEE-division Newton FFMAs. The conv
chain is a backward slice from x through FFMA accumulators (length + zero init operand).
Scheduling and register allocation differences are thereby ignored; arithmetic is not.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

INSN = re.compile(r"/\*[0-9a-f]{4,5}\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_.]+)\s*([^;]*);")
REG = re.compile(r"^-?\|?(U?R\d+|U?P\d|RZ|URZ|PT)(\.reuse)?\|?$")


def instructions(text):
    out = []
    for line in text.splitlines():
        m = INSN.search(line)
        if m:
            body = re.sub(r"&\S+", "", m.group(3))
            body = re.sub(r"\?\S+", "", body)
            ops = [o.strip() for o in body.split(",")] if body.strip() else []
            out.append((m.group(2), [o for o in ops if o]))
    return out


def base_reg(o):
    o = o.replace(".reuse", "").replace("|", "")
    neg = o.startswith("-")
    o = o.lstrip("-")
    return o, neg


def hfma2_const(ops):
    # HFMA2 Rd, -RZ, RZ, a, b  -> packed halves (a high, b low) == 32-bit constant
    import struct
    try:
        hi, lo = float(ops[3]), float(ops[4])
    except (IndexError, ValueError):
        return None
    h = struct.unpack("<H", struct.pack("<e", hi))[0]
    l = struct.unpack("<H", struct.pack("<e", lo))[0]
    return hex((h << 16) | l)


def silu_slice(insns, i):
    consts = {}
    for op, ops in insns[max(0, i - 40):i + 1]:
        if op == "MOV" and len(ops) == 2 and not REG.match(ops[1]):
            consts[ops[0]] = ops[1]
        elif op == "HFMA2" and len(ops) == 5 and ops[1] == "-RZ" and ops[2] == "RZ":
            c = hfma2_const(ops)
            if c:
                consts[ops[0]] = c
    x, _ = base_reg(insns[i][1][1])
    names = {x: "x"}
    out, after_rcp = [], 0
    for j in range(i, min(i + 160, len(insns))):
        op, ops = insns[j]
        if not ops:
            continue
        dst, srcs = ops[0], ops[1:]
        if op in ("MOV", "HFMA2") and srcs and not any(base_reg(s)[0] in names for s in srcs):
            if op == "MOV" and not REG.match(srcs[0]):
                consts[dst] = srcs[0]
            elif op == "HFMA2":
                c = hfma2_const(ops)
                if c:
                    consts[dst] = c
            names.pop(dst, None)
            continue
        if not any(base_reg(s)[0] in names for s in srcs):
            names.pop(base_reg(dst)[0], None)
            continue
        rendered = []
        for s in srcs:
            r, neg = base_reg(s)
            sign = "-" if neg else ""
            if r in names:
                rendered.append(sign + names[r])
            elif r in consts:
                rendered.append(sign + consts[r])
            elif REG.match(s):
                rendered.append(sign + ("RZ" if r in ("RZ", "URZ") else "ext"))
            else:
                rendered.append(s)
        idx = f"v{len(out)}"
        line = f"{op} {','.join(rendered)}"
        # x<<23 is emitted either as IMAD.SHL.U32 by 0x800000 or SHF.L.U32 by 0x17: same bits
        line = re.sub(r"^(IMAD\.SHL\.U32 (\w+),0x800000,RZ|SHF\.L\.U32 (\w+),0x17,RZ)$",
                      lambda m: f"SHL23 {m.group(2) or m.group(3)}", line)
        out.append(line)
        names[base_reg(dst)[0]] = idx
        if op == "MUFU.RCP":
            after_rcp = 1
        elif after_rcp and op.startswith("FFMA"):
            after_rcp += 1
            if after_rcp > 5:
                break
    return tuple(out)


def conv_chain(insns, i):
    """Backward FFMA accumulator chain ending in the silu input register."""
    x, _ = base_reg(insns[i][1][1])
    length, init = 0, None
    target = x
    for j in range(i - 1, max(-1, i - 400), -1):
        op, ops = insns[j]
        if ops and base_reg(ops[0])[0] == target:
            if op != "FFMA":
                init = f"producer:{op}"
                break
            length += 1
            acc, _ = base_reg(ops[3])
            if acc in ("RZ", "URZ"):
                init = "RZ"
                break
            target = acc
            if length >= 4:
                # the accumulator of the 4th FMA is the bias register (0.f when bias_ptr==nullptr)
                init = "bias_register"
                break
    return length, init


def analyse(text):
    ins = instructions(text)
    silu = [silu_slice(ins, i) for i, (op, _) in enumerate(ins) if op == "FFMA.SAT"]
    chains = [conv_chain(ins, i) for i, (op, _) in enumerate(ins) if op == "FFMA.SAT"]
    return ins, silu, chains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", nargs="+", required=True)
    ap.add_argument("--fused", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    res = {"original": {}, "fused": {}}
    orig = collections.Counter()
    for path in args.original:
        ins, silu, chains = analyse(Path(path).read_text(errors="replace"))
        orig.update(silu)
        res["original"][Path(path).name] = {
            "instructions": len(ins), "silu_instances": len(silu),
            "distinct_silu_dataflow": len(set(silu)),
            "conv_chains": dict(collections.Counter(f"len{l}:{i}" for l, i in chains)),
            "local_memory_insns": sum(op.startswith(("LDL", "STL")) for op, _ in ins),
            "ftz_insns": sum(".FTZ" in op and op.startswith(("FFMA", "FADD", "FMUL")) for op, _ in ins),
        }
    text = Path(args.fused).read_text(errors="replace")
    fused = collections.Counter()
    for chunk in re.split(r"\n\s*Function : ", text)[1:]:
        name = chunk.splitlines()[0].strip()
        short = "silu_compare" if "silu_compare" in name else re.sub(r".*kernelILi(\d+)ELi(\d+)E.*", r"prefill_TT\1_BR\2", name)
        ins, silu, chains = analyse(chunk)
        if short.startswith("prefill"):
            fused.update(silu)
        res["fused"][short] = {
            "instructions": len(ins), "silu_instances": len(silu),
            "distinct_silu_dataflow": len(set(silu)),
            "conv_chains": dict(collections.Counter(f"len{l}:{i}" for l, i in chains)),
            "local_memory_insns": sum(op.startswith(("LDL", "STL")) for op, _ in ins),
            "ftz_insns": sum(".FTZ" in op and op.startswith(("FFMA", "FADD", "FMUL")) for op, _ in ins),
        }
        if short == "silu_compare":
            res["fused"][short]["dataflow_both_variants_identical"] = len(set(silu)) == 1
    res["original_silu_dataflow"] = [list(s) for s in set(orig)]
    res["fused_silu_dataflow"] = [list(s) for s in set(fused)]
    res["silu_dataflow_identical"] = set(orig) == set(fused) and len(set(orig)) == 1
    res["note"] = ("conv chain: original accumulates on the bias register (bias_ptr==nullptr -> +0.0f), fused "
                   "uses RZ (+0.0); both FFMA(x, w0, +0.0) then 3 FFMA under IEEE RN, no FTZ.")
    Path(args.output).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
