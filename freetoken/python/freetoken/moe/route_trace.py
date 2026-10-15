"""D3a route trace (QWEN36_ROUTE_TRACE=1, experimental, default off; eager decode only).

Records what a one-layer-ahead pre-gate would have predicted: at the start of decoder layer
L+1 (before its mixer), layer L+1's router is applied to post_attention_layernorm(L+1) of the
residual stream leaving layer L. Next to it go the real routing of layer L+1 (the ids
``ensure_experts`` receives) and which of them missed the slot cache. Each decode request also
gets a snapshot of the LRU state (slot maps, usage, clock) at its first step, so the trace can be
replayed offline through the exact flashlib LRU, with and without the speculative prefetch.

One ``.npz`` per decode request goes to QWEN36_ROUTE_TRACE_OUT. Needs --cuda-graph-max-bs 0:
the hook is host code and would only run at capture time.
"""
from __future__ import annotations

import os
import pathlib

import torch

ENV = "QWEN36_ROUTE_TRACE"
OUT_ENV = "QWEN36_ROUTE_TRACE_OUT"


def enabled() -> bool:
    raw = os.environ.get(ENV, "0")
    if raw not in ("", "0", "1"):
        raise ValueError(f"{ENV}={raw!r}: expected unset, 0 or 1")
    return raw == "1"


class RouteTrace:
    def __init__(self, cache, out_dir: pathlib.Path):
        self.cache = cache
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.requests = 0
        self._reset()

    def _reset(self):
        self.snapshot = None
        self.steps = []          # per step: {layer: (actual [k], missed [k], predicted [k] | None)}
        self.pred = {}

    def _check_eager(self):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("QWEN36_ROUTE_TRACE needs eager decode: start with --cuda-graph-max-bs 0")

    def on_prefill(self):
        self.flush()

    def predict(self, layer_id: int, gate, norm, hidden: torch.Tensor, residual: torch.Tensor, top_k: int):
        self._check_eager()
        state = norm.forward(hidden + residual)
        self.pred[layer_id] = gate.forward(state).topk(top_k, dim=-1).indices.to(torch.int32).reshape(-1)

    def observe(self, layer_id: int, expert_ids: torch.Tensor):
        self._check_eager()
        c = self.cache
        if self.snapshot is None:
            self.snapshot = {k: t.detach().cpu().clone() for k, t in (
                ("slot_for_id", c.slot_for_id), ("id_of_slot", c.id_of_slot), ("usage", c.usage),
                ("step", c.step))}
        if layer_id == 0 or not self.steps:
            self.steps.append({})
        ids = expert_ids.reshape(-1).to(torch.int32)
        missed = c.slot_for_id[layer_id][ids.long()] < 0
        self.steps[-1][layer_id] = (ids.clone(), missed, self.pred.pop(layer_id, None))

    def flush(self):
        if self.snapshot is None or not self.steps:
            self._reset()
            return
        import numpy as np

        c = self.cache
        n, L = len(self.steps), c.num_layers
        k = next(iter(self.steps[0].values()))[0].numel()
        actual = np.full((n, L, k), -1, np.int32)
        missed = np.zeros((n, L, k), np.bool_)
        pred = np.full((n, L, k), -1, np.int32)
        for s, step in enumerate(self.steps):
            for layer, (ids, miss, p) in step.items():
                actual[s, layer] = ids.cpu().numpy()
                missed[s, layer] = miss.cpu().numpy()
                if p is not None:
                    pred[s, layer, :p.numel()] = p.cpu().numpy()[:k]
        np.savez_compressed(self.out / f"req{self.requests:04d}-pid{os.getpid()}.npz", actual=actual, missed=missed,
                            predicted=pred, num_experts=c.num_experts, cache_size=c.cache_size,
                            **{k: v.numpy() for k, v in self.snapshot.items()})
        self.requests += 1
        self._reset()


def install(cache) -> RouteTrace:
    """Wrap the Qwen3.5-MoE decoder layer with the pre-gate prediction and return the tracer;
    ``ensure_experts`` reports the real routing through ``cache.route_trace``."""
    from freetoken.core import get_global_ctx
    from freetoken.models.qwen3_5_moe import model as qmodel
    from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE

    trace = RouteTrace(cache, pathlib.Path(os.environ.get(OUT_ENV, "route-trace")))
    layer_cls = qmodel.Qwen3_5DecoderLayer
    original = layer_cls.forward
    if getattr(original, "_route_trace", False):
        return trace

    def forward(self, hidden, residual):
        if get_global_ctx().batch.is_prefill:
            if getattr(self.mlp, "experts", None) is not None and self.mlp.experts.layer_id == 0:
                trace.on_prefill()
        elif residual is not None and isinstance(self.mlp, Qwen3_5MoE):
            trace.predict(self.mlp.experts.layer_id, self.mlp.gate, self.post_attention_layernorm,
                          hidden, residual, self.mlp.experts.top_k)
        return original(self, hidden, residual)

    forward._route_trace = True
    layer_cls.forward = forward
    return trace
