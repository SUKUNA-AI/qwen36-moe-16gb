from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    make_moe_layer,
    silu_and_mul,
)

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class _SharedExpert(BaseOP):
    """Always-present shared SwiGLU expert of width ``shared_expert_intermediate_size``."""

    def __init__(
        self, config: ModelConfig, hidden_size: int, intermediate_size: int, *, prefix: str = ""
    ):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = LinearRowParallel(
            intermediate_size, hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class Qwen3_5DenseMLP(_SharedExpert):
    """Dense (non-MoE) SwiGLU MLP for dense Qwen3.x checkpoints (e.g. 27B): ``gate_up_proj``
    (fused gate|up) + ``down_proj`` at full ``intermediate_size``. Same structure as the shared
    expert, so it reuses ``_SharedExpert`` directly and keeps the state-dict keys flat
    (``...layers.N.mlp.{gate_up_proj,down_proj}``)."""

    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        super().__init__(config, config.hidden_size, config.intermediate_size, prefix=prefix)


class Qwen3_5MoE(BaseOP):
    """Routed MoE (256 experts, top-8) plus a gated shared expert:

        out = routed(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x)

    Router softmaxes over all experts, takes top-k, and renormalizes (HF semantics).
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None, *, prefix: str = ""):
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )
        # routers stay bf16 whatever the checkpoint quantizes
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(
            config, config.hidden_size, config.shared_expert_intermediate_size,
            prefix=f"{prefix}.shared_expert",
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)
        from freetoken.moe import shared_epilogue

        # D5 (moe/shared_epilogue.py): decode merges sigmoid * mul + add into one launch
        self._fused_epilogue = shared_epilogue.EPILOGUE if shared_epilogue.enabled() else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        epilogue = getattr(self, "_fused_epilogue", None)
        fused = epilogue is not None and not get_global_ctx().batch.is_prefill
        shared_forward = self._shared_parts if fused else self._shared_forward
        # Compute the router + shared expert BEFORE the routed experts: the fused MoE
        # kernel may write into ``hidden_states`` in place, which would corrupt the
        # shared expert's input (HF also evaluates the shared expert first).
        router_logits = self.gate.forward(hidden_states)
        can_overlap = getattr(self.experts, "can_overlap_shared", None)
        if can_overlap is not None and can_overlap(hidden_states):
            routed, shared = self.experts.forward_with_shared(
                hidden_states, router_logits, shared_forward,
            )
        else:
            shared = shared_forward(hidden_states)
            routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        if fused:
            return epilogue(routed, *shared).view(num_tokens, hidden_dim)
        return (routed + shared).view(num_tokens, hidden_dim)

    def _shared_forward(self, hidden_states):
        shared = self.shared_expert.forward(hidden_states)
        return shared * torch.sigmoid(self.shared_expert_gate.forward(hidden_states))

    def _shared_parts(self, hidden_states):
        # the same two projections as _shared_forward; the gating and the merge run in the fused epilogue
        shared = self.shared_expert.forward(hidden_states)
        return shared, self.shared_expert_gate.forward(hidden_states)


__all__ = ["Qwen3_5MoE", "Qwen3_5DenseMLP"]
