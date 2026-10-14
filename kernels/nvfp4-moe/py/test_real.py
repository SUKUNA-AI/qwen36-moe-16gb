"""Bit-exactness on REAL checkpoint expert weights (nvidia/Qwen3.6-35B-A3B-NVFP4).

Only the tensors of the selected MoE layers are read (read-only, via safetensors
safe_open); the model is not loaded.  Expert banks are packed exactly like FreeToken's
TritonNvfp4MoEKernel.pack (gate||up rows, fp32 per-row globals from weight_scale_2).

Routing: real router weights of each layer applied to real-token proxies
(GemmaRMSNorm(embed_tokens[token]) * (1 + post_attention_layernorm.weight)), followed by
FreeToken's fused_topk(renormalize=True) -> realistic, skewed expert popularity.  The
same normalised hidden states are the expert inputs.  Token text is read locally and
never written to any report (only counts/hashes).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import fcntl
import subprocess

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_bitexact as tb  # noqa: E402
from bitexact_lib import BitExactPrefill  # noqa: E402
from freetoken.moe.fused import fused_topk  # noqa: E402

MODEL = os.environ.get("QWEN36_MODEL_DIR", "/path/to/model")  # NVFP4 checkpoint directory
PFX = "model.language_model."
H, I, E, TOPK = 2048, 512, 256, 8
GLOBAL_DTYPE = torch.float32  # fp16 only for the fp16-vs-fp32 output-shift measurement


class Ckpt:
    def __init__(self, root):
        self.root = root
        idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))
        self.map = idx["weight_map"]
        self.files = {}

    def get(self, name):
        fn = self.map[name]
        if fn not in self.files:
            self.files[fn] = safe_open(os.path.join(self.root, fn), framework="pt", device="cpu")
        return self.files[fn].get_tensor(name)

    def get_rows(self, name, rows):
        fn = self.map[name]
        if fn not in self.files:
            self.files[fn] = safe_open(os.path.join(self.root, fn), framework="pt", device="cpu")
        sl = self.files[fn].get_slice(name)
        return torch.stack([sl[r:r + 1][0] for r in rows])


def load_layer_banks(ck, layer):
    t0 = time.time()
    gp, gs, gg, up, us, ug, dp, ds, dg = ([] for _ in range(9))
    for e in range(E):
        b = f"{PFX}layers.{layer}.mlp.experts.{e}."
        gp.append(ck.get(b + "gate_proj.weight")); gs.append(ck.get(b + "gate_proj.weight_scale"))
        gg.append(ck.get(b + "gate_proj.weight_scale_2"))
        up.append(ck.get(b + "up_proj.weight")); us.append(ck.get(b + "up_proj.weight_scale"))
        ug.append(ck.get(b + "up_proj.weight_scale_2"))
        dp.append(ck.get(b + "down_proj.weight")); ds.append(ck.get(b + "down_proj.weight_scale"))
        dg.append(ck.get(b + "down_proj.weight_scale_2"))
    assert gp[0].dtype == torch.uint8 and gp[0].shape == (I, H // 2), (gp[0].dtype, gp[0].shape)
    assert gs[0].dtype == torch.float8_e4m3fn and gs[0].shape == (I, H // 16), (gs[0].dtype, gs[0].shape)
    assert dp[0].shape == (H, I // 2) and ds[0].shape == (H, I // 16), (dp[0].shape, ds[0].shape)

    def glob(lst, rows):  # == freetoken global_rows(): reshape(E,-1).to(fp32).expand(-1, rows)
        flat = torch.stack([x.reshape(-1) for x in lst]).reshape(E, -1).to(GLOBAL_DTYPE)
        return flat.expand(-1, rows) if flat.shape[1] == 1 else flat

    gate_up = torch.cat([torch.stack(gp), torch.stack(up)], dim=1).contiguous()
    gate_up_scale = torch.cat([torch.stack(gs), torch.stack(us)], dim=1).contiguous()
    gate_up_global = torch.cat([glob(gg, I), glob(ug, I)], dim=1).contiguous()
    down = torch.stack(dp).contiguous()
    down_scale = torch.stack(ds).contiguous()
    down_global = glob(dg, H).contiguous()
    banks = (gate_up, gate_up_scale, gate_up_global, down, down_scale, down_global)  # CPU only: no CUDA context yet
    stats = {
        "layer": layer, "load_seconds": round(time.time() - t0, 1),
        "scale_codes_used": int(torch.unique(torch.cat([gate_up_scale.view(torch.uint8).reshape(-1),
                                                         down_scale.view(torch.uint8).reshape(-1)])).numel()),
        "global_range": [float(torch.cat([gate_up_global.reshape(-1), down_global.reshape(-1)]).float().min()),
                              float(torch.cat([gate_up_global.reshape(-1), down_global.reshape(-1)]).float().max())],
        "bank_sha256_prefix": hashlib.sha256(gate_up[:2].numpy().tobytes() + down[:2].numpy().tobytes()).hexdigest()[:16],
    }
    return banks, stats


def token_ids(n):
    """Real token ids from local mixed RU/EN/code text (never written out)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
    import freetoken
    ft = os.path.dirname(freetoken.__file__)
    # (the archived run also tokenized a private Russian review note; any mixed text works)
    sources = [
        os.path.join(MODEL, "README.md"),
        os.path.join(ft, "moe", "fused_nvfp4.py"),
        os.path.join(ft, "kernel", "triton", "nvfp4_fused_moe.py"),
        os.path.join(ft, "layers", "quantization", "moe", "nvfp4.py"),
    ]
    ids = []
    for s in sources:
        ids += tok.encode(open(s, encoding="utf-8", errors="replace").read()).ids
    while len(ids) < n:
        ids = ids + ids
    return ids[:n], hashlib.sha256(str(ids[:n]).encode()).hexdigest()[:16]


def embed_rows(ck, ids):
    uniq = sorted(set(ids))
    rows = ck.get_rows(PFX + "embed_tokens.weight", uniq).float()
    pos = {t: i for i, t in enumerate(uniq)}
    return rows[[pos[t] for t in ids]]


def hidden_states(emb, norm_w):
    x = emb * torch.rsqrt(emb.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + norm_w)
    return x.to(torch.bfloat16)


LOCK = os.environ.get("QWEN36_GPU_LOCK", "/tmp/qwen36-gpu-exclusive.lock")  # shared GPU lock file


def acquire_gpu(max_wait_s, retry_s=600, deadline_epoch=None):
    """Non-blocking flock(2) on the shared GPU lock (same lock as flock(1)); retry at most
    every `retry_s` seconds until `deadline_epoch` (or max_wait_s).  Refuse while a model
    service is active.  Exit 75 when giving up."""
    f = open(LOCK, "a+")
    deadline = deadline_epoch if deadline_epoch else time.time() + max_wait_s
    while True:
        busy = False
        for s in ("qwen36-inference", "qwen36-experiment"):
            st = subprocess.run(["systemctl", "--user", "is-active", s], capture_output=True, text=True).stdout.strip()
            if st in ("active", "activating", "reloading", "deactivating"):
                busy = True
        if not busy:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return f
            except BlockingIOError:
                pass
        if time.time() + retry_s > deadline:
            print(f"GPU lock busy / service active at {time.strftime('%H:%M:%S', time.gmtime())} UTC: giving up",
                  flush=True)
            sys.exit(75)
        print(f"GPU busy at {time.strftime('%H:%M:%S', time.gmtime())} UTC; retry in {retry_s} s", flush=True)
        time.sleep(retry_s)


def prepare_cpu(layers, ms):
    """CPU-only phase: every checkpoint read happens here (no CUDA context)."""
    ck = Ckpt(MODEL)
    ids_all, ids_hash = token_ids(max(ms))
    emb = embed_rows(ck, ids_all)
    cpu = []
    for layer in layers:
        banks, stats = load_layer_banks(ck, layer)
        gate_w = ck.get(PFX + f"layers.{layer}.mlp.gate.weight")
        norm_w = ck.get(PFX + f"layers.{layer}.post_attention_layernorm.weight").float()
        print(f"loaded layer {layer} in {stats['load_seconds']} s", flush=True)
        cpu.append((layer, banks, stats, gate_w, norm_w))
    return {"ids_hash": ids_hash, "emb": emb, "layers": cpu, "ms": ms}


def run_gpu(data, output, variants_arg="all"):
    lib = BitExactPrefill(tile=(64, 64, 64))
    variants = list(range(lib.variant_count())) if variants_arg == "all" else [int(v) for v in variants_arg.split(",")]
    ms = data["ms"]
    report = {"library": lib.path, "build_id": lib.build_id(), "device": torch.cuda.get_device_name(),
              "checkpoint": "nvidia/Qwen3.6-35B-A3B-NVFP4 @ 1355db6a052410cfd62085d94b58866fd0f2c3c5",
              "token_ids_sha256_prefix": data["ids_hash"], "variants": variants, "layers": [], "results": []}
    emb = data["emb"].cuda()
    for layer, banks_cpu, stats, gate_w, norm_w in data["layers"]:
        banks = tuple(t.cuda() for t in banks_cpu)
        xall = hidden_states(emb, norm_w.cuda())
        logits = (xall.float() @ gate_w.cuda().float().t())
        for m in ms:
            x = xall[:m].contiguous()
            w, ids = fused_topk(x, logits[:m].contiguous(), TOPK, True)
            counts = torch.bincount(ids.reshape(-1).long(), minlength=E)
            t0 = time.time()
            stages = tb.triton_stages(x, banks, w, ids, False)
            tag = f"real/L{layer}/M{m}"
            res = tb.cuda_checks(lib, x, banks, w, ids, stages, False, variants, True, tag)
            torch.cuda.synchronize()
            bad = [r for r in res if r["mismatch"]]
            print(f"{tag}: {len(res) - len(bad)}/{len(res)} bit-exact; experts used {int((counts > 0).sum())}, "
                  f"max routes/expert {int(counts.max())} ({time.time() - t0:.1f}s)", flush=True)
            for r in bad:
                print("   MISMATCH", json.dumps(r), flush=True)
            report["results"] += res
            stats.setdefault("routing", []).append({"m": m, "experts_used": int((counts > 0).sum()),
                                                    "max_routes_per_expert": int(counts.max()),
                                                    "mean_routes_per_used_expert": float(counts[counts > 0].float().mean())})
        report["layers"].append(stats)
        del banks
        torch.cuda.empty_cache()
    tot = sum(r["elements"] for r in report["results"])
    bad = sum(r["mismatch"] for r in report["results"])
    report["summary"] = {"comparisons": len(report["results"]), "elements": tot, "mismatched_elements": bad,
                         "bit_equal_fraction": (tot - bad) / tot if tot else 1.0, "all_bit_exact": bad == 0}
    print(json.dumps(report["summary"]), flush=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=1)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,13,26,39")
    ap.add_argument("--m", default="1,7,17,64,65,256,2048,4096")
    ap.add_argument("--variants", default="all")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-wait", type=int, default=1800)
    args = ap.parse_args()
    data = prepare_cpu([int(x) for x in args.layers.split(",")], [int(m) for m in args.m.split(",")])
    assert not torch.cuda.is_initialized(), "CUDA touched before the GPU lock"
    lock = acquire_gpu(args.max_wait)  # noqa: F841  (held until exit)
    run_gpu(data, args.output, args.variants)


if __name__ == "__main__":
    main()
