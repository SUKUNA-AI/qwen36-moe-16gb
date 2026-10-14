from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.blocks import embed_input_ids
from freetoken.models.qwen3_vl.vision import Qwen3VLVisionModel, QwenVLVisionMixin
from freetoken.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                quant_config=config.quant,
                prefix=f"{prefix}.linear_attn",
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = (
            Qwen3_5MoE(config, layer_id, prefix=f"{prefix}.mlp")
            if config.moe_enabled
            else Qwen3_5DenseMLP(config, prefix=f"{prefix}.mlp")
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Residual-stream form: fuse each residual-add into the next RMSNorm
        # (GemmaRMSNorm.forward_add_residual) so add + norm are one kernel per sublayer.
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                Qwen3_5DecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = embed_input_ids(self.embed_tokens, input_ids, get_global_ctx().batch)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        x, _ = self.norm.forward_add_residual(x, residual)
        return x

    # --- layer-major prefill hooks (freetoken.engine.layer_major) -----------------------------
    # ``forward`` split into its three parts so one long prompt can run ``for layer: for chunk:``
    # with exactly the same per-(chunk, layer) kernels and inputs as ``for chunk: for layer:``.
    layer_major_supported = True

    def layer_major_embed(self, batch) -> torch.Tensor:
        return embed_input_ids(self.embed_tokens, batch.input_ids, batch)

    def layer_major_num_layers(self) -> int:
        return len(self.layers.op_list)

    def layer_major_layer(self, index: int, hidden: torch.Tensor, residual: torch.Tensor | None):
        return self.layers.op_list[index].forward(hidden, residual)

    def layer_major_uses_attention(self, index: int) -> bool:
        return not self.layers.op_list[index]._is_linear

    def layer_major_final_norm(self, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x, _ = self.norm.forward_add_residual(hidden, residual)
        return x

    # Grouped attention (QWEN36_LAYER_MAJOR_ATTN_BATCH): a full-attention decoder layer split at
    # the attention kernel. pre + one prefill call + post == Qwen3_5DecoderLayer.forward with
    # Qwen3_5Attention.forward (_project -> store_kv + run -> _combine), op for op.
    def layer_major_attn_groupable(self, index: int) -> bool:
        layer = self.layers.op_list[index]
        return not layer._is_linear and not getattr(layer.self_attn, "_use_fused_qk_kv", False)

    def layer_major_attn_dims(self, index: int) -> tuple[int, int]:
        attn = self.layers.op_list[index].self_attn
        return attn.num_q, attn.head_dim

    def layer_major_attn_kv_layer(self, index: int) -> int:
        return self.layers.op_list[index].self_attn.layer_id

    def layer_major_attn_pre(self, index: int, hidden: torch.Tensor, residual: torch.Tensor | None):
        """Input norm, QKV projection, q/k norm, rope and the KV store of one chunk (the active
        batch): returns (q [rows, num_q, head_dim], gate [rows, num_q*head_dim], residual)."""
        layer = self.layers.op_list[index]
        if residual is None:
            residual = hidden
            hidden = layer.input_layernorm.forward(hidden)
        else:
            hidden, residual = layer.input_layernorm.forward_add_residual(hidden, residual)
        attn = layer.self_attn
        ctx = get_global_ctx()
        q, k, v, gate = attn._project(hidden)
        ctx.attn_backend.kvcache.store_kv(k, v, ctx.batch.out_loc, attn.layer_id)
        return q, gate, residual

    def layer_major_attn_post(self, index: int, attn_out: torch.Tensor, gate: torch.Tensor,
                              residual: torch.Tensor):
        """Output gate + o_proj, post-attention norm and MoE of one chunk: (hidden, residual)."""
        layer = self.layers.op_list[index]
        hidden = layer.self_attn._combine(attn_out, gate)
        hidden, residual = layer.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = layer.mlp.forward(hidden)
        return hidden, residual


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        if hasattr(self,'mtp_hidden'):
            self.mtp_hidden[:len(output)].copy_(output)
        return self.lm_head.forward(output)

    # layer-major prefill: valid only while ``forward`` is exactly this one (subclasses that
    # override it fail the identity check in freetoken.engine.layer_major.engine_reason).
    layer_major_plain_forward = forward

    def layer_major_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head.forward(hidden)


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    """The MoE releases share the dense code path: the decoder picks the routed or dense MLP from config.num_experts."""


class Qwen3_5ForConditionalGeneration(QwenVLVisionMixin, Qwen3_5ForCausalLM):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.is_multimodal:
            assert not config.vision_config.deepstack_visual_indexes, "Qwen3.5 consumes no DeepStack features"
            self.visual = Qwen3VLVisionModel(config.vision_config, quant_config=config.quant, prefix="visual")


class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """The MoE releases with the vision tower; see Qwen3_5MoeForCausalLM."""


__all__ = [
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
]
