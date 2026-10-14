"""Shared helpers for the GPU tests of the integrated FP8 GEMV (import from scripts/; GPU under the lock).

* the patched FreeToken tree ($FREETOKEN_PYTHON) is the import root for ``freetoken``;
* the PRISTINE fp8_pertensor_linear.py ($FREETOKEN_PRISTINE_PYTHON, tree without the integration) is loaded
  side by side under another module name, as the "current behaviour" baseline;
* real projections are read from the checkpoint with FreeToken's fusion rules for modelopt FP8
  (qwen3_5_moe/weight.py _DenseReader._check/_emit): weights cat(dim=0), each part's scalar
  weight_scale broadcast to its rows (fp32 [N]), input_scale = max over parts (fp32 scalar).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

W = Path(__file__).resolve().parents[1]          # kernels/fp8-gemv
# FreeToken tree WITH the FP8 GEMV integration (kernel/cuda_fp8_gemv.py etc.)
TREE = Path(os.environ.get("FREETOKEN_PYTHON", str(W.parents[1] / "freetoken" / "python")))
# FreeToken tree WITHOUT the integration (pristine fp8_pertensor_linear.py = baseline behaviour)
ORIG = Path(os.environ.get("FREETOKEN_PRISTINE_PYTHON", str(W / "pristine" / "python")))
LIB = Path(os.environ.get("QWEN36_FP8_GEMV_LIBRARY", str(W / "build" / "libqwen_fp8_gemv.so")))
MODEL = Path(os.environ.get("QWEN36_MODEL_DIR", "/path/to/model"))  # NVFP4 checkpoint directory
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

# WSL: the profiler's injection library hangs /usr/bin/file (spawned by platform.architecture);
# drop LD_PRELOAD for that helper only (same workaround as the experiment harness).
if os.environ.get("QWEN36_CUDA_PROFILE") == "1":
    _orig_popen = subprocess.Popen

    class _ProfilePopen(_orig_popen):
        def __init__(self, args, *pos, **kw):
            cmd = args[0] if isinstance(args, (list, tuple)) else ""
            if Path(str(cmd)).name == "file":
                env = dict(kw.get("env") or os.environ)
                env.pop("LD_PRELOAD", None)
                kw["env"] = env
            super().__init__(args, *pos, **kw)

    subprocess.Popen = _ProfilePopen

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

FP8 = torch.float8_e4m3fn


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_orig_fpl():
    """The unmodified fp8_pertensor_linear.py as module 'ft_orig_fp8_pertensor_linear'."""
    path = ORIG / "freetoken/kernel/triton/fp8_pertensor_linear.py"
    spec = importlib.util.spec_from_file_location("ft_orig_fp8_pertensor_linear", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def env_info() -> dict:
    import freetoken.kernel.cuda_fp8_gemv as g

    props = torch.cuda.get_device_properties(0)
    return {
        "device": props.name, "cc": list(torch.cuda.get_device_capability(0)), "sms": props.multi_processor_count,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "library": str(LIB), "library_sha256": sha256(LIB), "pinned_sha256": g.LIBRARY_SHA256,
        "source_sha256": g.SOURCE_SHA256, "variant": g.VARIANT,
        "tree_files_sha256": {rel: sha256(TREE / rel) for rel in (
            "freetoken/kernel/cuda_fp8_gemv.py", "freetoken/kernel/triton/fp8_pertensor_linear.py",
            "freetoken/layers/quantization/linear/fp8_tensor.py")},
        "orig_fpl_sha256": sha256(ORIG / "freetoken/kernel/triton/fp8_pertensor_linear.py"),
        "libtorch_cuda_sha256": sha256(Path(torch.__file__).parent / "lib/libtorch_cuda.so"),
    }


class Checkpoint:
    def __init__(self, root: Path = MODEL):
        self.root = root
        self.map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.layer_types = json.loads((root / "config.json").read_text())["text_config"]["layer_types"]
        self.handles = {}

    def get(self, key: str) -> torch.Tensor:
        f = self.map[key]
        if f not in self.handles:
            self.handles[f] = safe_open(str(self.root / f), framework="pt", device="cpu")
        return self.handles[f].get_tensor(key)

    def fused(self, prefix: str, parts: list[str]):
        ws, ss, xs = [], [], []
        for p in parts:
            w = self.get(f"{prefix}.{p}.weight")
            if w.dtype != FP8:
                raise TypeError(f"{prefix}.{p}.weight is {w.dtype}")
            s = self.get(f"{prefix}.{p}.weight_scale").reshape(-1).to(torch.float32)
            if s.numel() == 1:
                s = s.expand(w.shape[0]).contiguous()
            elif s.numel() != w.shape[0]:
                raise ValueError(f"{prefix}.{p}: weight_scale has {s.numel()} elements for {w.shape[0]} rows")
            ws.append(w)
            ss.append(s.contiguous())
            xs.append(self.get(f"{prefix}.{p}.input_scale").reshape(()).to(torch.float32))
        W_ = ws[0] if len(ws) == 1 else torch.cat(ws, 0)
        S_ = ss[0] if len(ss) == 1 else torch.cat(ss, 0)
        X_ = xs[0] if len(xs) == 1 else torch.stack(xs).max()
        return W_.contiguous().cuda(), S_.contiguous().cuda(), X_.cuda()

    def projection(self, layer: int, kind: str):
        base = f"model.language_model.layers.{layer}"
        lt = self.layer_types[layer]
        if kind == "in":
            if lt == "linear_attention":
                return self.fused(f"{base}.linear_attn", ["in_proj_qkv", "in_proj_z"])
            return self.fused(f"{base}.self_attn", ["q_proj", "k_proj", "v_proj"])
        if lt == "linear_attention":
            return self.fused(f"{base}.linear_attn", ["out_proj"])
        return self.fused(f"{base}.self_attn", ["o_proj"])


def make_layer(Wt, St, Xt):
    return types.SimpleNamespace(weight=Wt, weight_scale=St, input_scale=Xt, bias=None)


def load_projections(ck: Checkpoint, layers=None, kinds=("in", "out")):
    projs = []
    for layer in (range(len(ck.layer_types)) if layers is None else layers):
        for kind in kinds:
            Wt, St, Xt = ck.projection(layer, kind)
            projs.append(dict(layer=layer, kind=kind, type=ck.layer_types[layer], W=Wt, S=St, xs=Xt,
                              N=Wt.shape[0], K=Wt.shape[1], uniform=bool((St == St[0]).all().item())))
    return projs


def activation(kind: str, M: int, K: int, input_scale: float, gen: torch.Generator) -> torch.Tensor:
    """bf16 hidden-state-like activations relative to the layer's calibrated range amax=448*s."""
    amax = float(input_scale) * 448.0
    if kind == "normal":
        x = torch.randn(M, K, generator=gen) * (amax / 6)
    elif kind == "heavy":  # light body, rare outliers up to 2x the calibrated range (saturating)
        x = torch.randn(M, K, generator=gen) * (amax / 20)
        spikes = torch.rand(M, K, generator=gen) < 0.005
        mag = (torch.rand(M, K, generator=gen) * 1.5 + 0.5) * amax * torch.sign(torch.randn(M, K, generator=gen))
        x = torch.where(spikes, mag, x)
    elif kind == "edge":  # +0, -0, saturation, exact 448, e4m3-subnormal and underflowing values, body
        body = torch.randn(M, K, generator=gen) * (amax / 6)
        sel = torch.randint(0, 10, (M, K), generator=gen)
        sgn = torch.sign(torch.randn(M, K, generator=gen))
        choices = [torch.zeros(M, K), torch.full((M, K), -0.0), 2.0 * amax * sgn, amax * sgn,
                   amax * 1e-4 * sgn, amax * 3e-3 * sgn, amax * 1e-9 * sgn, body, body, body]
        x = torch.stack(choices)[sel, torch.arange(M)[:, None], torch.arange(K)[None, :]]
    else:
        raise ValueError(kind)
    return x.to(torch.bfloat16).cuda()


def mismatches(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Device scalar: number of differing bf16 encodings (no sync)."""
    return (a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).sum()


def scaled_mm_reference(fpl, x, W, S, xs, uniform):
    """torch._scaled_mm exactly as the production W8A8 branch issues it (quant included)."""
    qa = fpl._static_quant(x, xs)
    if uniform:
        return torch._scaled_mm(qa, W.t(), scale_a=xs.reshape(()), scale_b=S[0].reshape(()), out_dtype=torch.bfloat16)
    return torch._scaled_mm(qa, W.t(), scale_a=xs.reshape(1, 1).expand(x.shape[0], 1).contiguous(),
                            scale_b=S.reshape(1, -1), out_dtype=torch.bfloat16)


class GemvCounter:
    """Counts GEMV-served calls by (N, M) by wrapping cuda_fp8_gemv.rowwise (looked up per call)."""

    def __init__(self):
        import freetoken.kernel.cuda_fp8_gemv as g

        self.g = g
        self.orig = g.rowwise
        self.served = {}
        self.declined = {}

        def wrapped(qa, weight, input_scale, weight_scale):
            out = self.orig(qa, weight, input_scale, weight_scale)
            key = (int(weight.shape[0]), int(qa.shape[0]))
            book = self.served if out is not None else self.declined
            book[key] = book.get(key, 0) + 1
            return out

        g.rowwise = wrapped

    def reset(self):
        self.served.clear()
        self.declined.clear()

    def total(self):
        return sum(self.served.values())
