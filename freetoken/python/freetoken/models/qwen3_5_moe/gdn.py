from __future__ import annotations

import os
import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.layers import BaseOP, GatedRMSNorm, LinearColParallelMerged, LinearReplicated
from freetoken.layers.quantization import QuantConfig

from .gdn_kernels import gdn_decode_fla, gdn_prefill_chunk_fla


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise conv weight ``[conv_dim, 1, K]`` (key ``conv1d.weight``)."""

    def __init__(self, conv_dim: int, kernel: int):
        self.weight = torch.empty(conv_dim, 1, kernel)


class Qwen3_5GatedDeltaNet(BaseOP):
    """GatedDeltaNet op using the vendored flash-linear-attention triton kernels
    (``freetoken.kernel.fla``) for the recurrence and a per-request
    recurrent + conv state held in ``ctx.linear_state_pool`` (keyed by ``Req.table_idx``).

    Parameter names match HF (``in_proj_qkv``/``in_proj_z``/``in_proj_b``/``in_proj_a``/
    ``conv1d``/``A_log``/``dt_bias``/``norm``/``out_proj``). Handles prefill (incl. chunked
    continuation) and single-token decode; state is fresh when ``req.cached_len == 0``.
    """

    def __init__(
        self, hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_kernel_size, rms_norm_eps, layer_id, *, quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        self.layer_id = layer_id
        self._fp32_gdn_snapshot = os.environ.get('QWEN36_FP32_GDN_SNAPSHOT') == '1'
        # Opt-in bit-exact CUDA kernels (libqwen36_gdn): fused conv+L2, gate+cumsum, gated
        # norm+FP8 (prefill) and one fused decode kernel. Unsupported layouts/schemes fall back
        # to the original path below before anything is modified.
        self._cuda_gdn = os.environ.get('QWEN36_CUDA_GDN') == '1'
        # decode kernel variant (identical bytes): 'cluster' = Triton's 128-program layout in 8-CTA
        # clusters (measured faster, default) or 'cta' = one 8-warp CTA per key head
        self._cuda_gdn_decode = 'cta' if os.environ.get('QWEN36_CUDA_GDN_DECODE') == 'cta' else 'cluster'
        # Opt-in on top of QWEN36_CUDA_GDN (libqwen36_gdn ABI 2): QWEN36_CUDA_GDN_FLA=1|wu replaces
        # FLA recompute_w_u + chunk_h + chunk_o by one fused bit-exact CUDA kernel (kkt_solve stays
        # Triton); =ho replaces only chunk_h + chunk_o. Unsupported inputs use the Triton kernels.
        _fla = os.environ.get('QWEN36_CUDA_GDN_FLA', '0')
        self._cuda_gdn_fla = None if not self._cuda_gdn or _fla in ('', '0') else ('ho' if _fla == 'ho' else 'wu')
        # The fla chunk/decode kernels read+write the recurrent state and the per-chunk h as
        # [V, K] while the LinearStatePool declares it [K, V]; these coincide (and the
        # hybrid-radix snapshot scatter h[h_row]->slot is a plain copy) only when the two head
        # dims are equal. Qwen3.5/3.6 satisfy this (128/128); guard any future config.
        assert head_k_dim == head_v_dim, (
            f"GatedDeltaNet requires head_k_dim == head_v_dim, got {head_k_dim} != {head_v_dim}"
        )
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = conv_kernel_size
        # quantized checkpoints quantize qkv|z but not b|a, so the fusion splits into a qkvz GEMM and a ba GEMM with their own schemes (matches sglang / vLLM)
        self._split_in_proj = (
            quant_config is not None and quant_config.scheme_for(f"{prefix}.in_proj_qkvz") is not None
        )

        self._in_proj_split = [self.conv_dim, self.value_dim, num_v_heads, num_v_heads]
        if self._split_in_proj:
            self.in_proj_qkvz = LinearColParallelMerged(
                hidden_size, [self.conv_dim, self.value_dim], has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.in_proj_qkvz",
            )
            self.in_proj_ba = LinearColParallelMerged(
                hidden_size, [num_v_heads, num_v_heads], has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.in_proj_ba",
            )
        else:
            # Fused input projection (one GEMM instead of four): qkv | z | b | a.
            self.in_proj = LinearColParallelMerged(
                hidden_size, self._in_proj_split, has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.in_proj",
            )
        self.conv1d = _DepthwiseConv1d(self.conv_dim, conv_kernel_size)
        # Recurrence-gating params kept in fp32 (exp/softplus is precision-sensitive,
        # and the fla kernel reads them as fp32) -- matches HF/sglang, and avoids a
        # per-call .float() upcast in the decode wrapper. The weight loader exempts
        # *.A_log / *.dt_bias from the model-dtype downcast.
        self.dt_bias = torch.empty(num_v_heads, dtype=torch.float32)
        self.A_log = torch.empty(num_v_heads, dtype=torch.float32)
        self.norm = GatedRMSNorm(head_v_dim, eps=rms_norm_eps)
        self.out_proj = LinearReplicated(
            self.value_dim, hidden_size, has_bias=False,
            quant_config=quant_config, prefix=f"{prefix}.out_proj",
        )

    def _gate_params(self, a: torch.Tensor, b: torch.Tensor):
        beta = b.sigmoid()
        g = -self.A_log.exp() * F.softplus(a.float() + self.dt_bias)
        return g, beta

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)  # [conv_dim, kernel] for the fused kernel

    def _conv_prefill(self, conv_in, pool, cu_seqlens, cache_indices, has_initial_state) -> torch.Tensor:
        """Varlen causal conv (fused sgl_kernel) with silu; reads/updates each request's
        conv state in place by ``cache_indices`` slot. ``conv_in`` [total, conv_dim].
        ``cu_seqlens`` / ``cache_indices`` / ``has_initial_state`` come from FLAMetadata."""
        li = pool.local_index(self.layer_id)
        x = conv_in.transpose(0, 1).contiguous()  # [conv_dim, total]
        out = causal_conv1d_varlen(x, self._conv_weight(), pool.conv_states[li],
                                   cu_seqlens, cache_indices, has_initial_state)
        return out.transpose(0, 1)  # [total, conv_dim]

    def _conv_decode(self, conv_in: torch.Tensor, table_idx: torch.Tensor, pool) -> torch.Tensor:
        """Single-token causal conv update (fused sgl_kernel) by ``table_idx`` slot;
        updates conv state in place, no host loop -> CUDA-graph capturable.
        ``conv_in`` [B, conv_dim] -> silu(conv) [B, conv_dim]."""
        li = pool.local_index(self.layer_id)
        return causal_conv1d_decode(conv_in, pool.conv_states[li], self._conv_weight(), table_idx)

    def _write_track_snapshot(self, pool, li: int, conv_in: torch.Tensor,
                              h: torch.Tensor, fla, *, recurrent_already_stored: bool = False) -> None:
        """Snapshot this layer's recurrent + conv state at the chunk-aligned track boundary
        into a donatable pool slot, on the forward stream (hybrid-radix extra_buffer path).
        SSM normally copies BF16 h into the pool; the opt-in FP32 kernel store skips it.
        FLA prefill and decode both address the pool as [V,K] (K == V in this model).
        Conv: the last (kernel-1) raw conv-input timesteps ending at the boundary."""
        rec = pool.recurrent_states[li]
        if not recurrent_already_stored:
            rec.index_copy_(0, fla.track_dst, h[0, fla.track_h_row].to(rec.dtype))
        cv = pool.conv_states[li]
        # conv_in [total, conv_dim]; gather the (kernel-1) window per tracked req.
        conv_win = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()  # [nt, conv_dim, K-1]
        cv.index_copy_(0, fla.track_dst, conv_win.to(cv.dtype))

    def _forward_cuda(self, qkvz, ba, fla, pool, li, is_decode):
        """Bit-exact CUDA path (libqwen36_gdn) or None when unsupported. Checks everything before
        touching any state, so a None return leaves the original path fully authoritative."""
        from freetoken.kernel.cuda_gdn import fp8_out_proj_operands, get_gdn_lib, has_empty_sequence

        operands = fp8_out_proj_operands(self.out_proj, self.value_dim)
        if operands is None:
            return None
        weight, weight_scale, in_scale = operands
        conv_w = self._conv_weight()
        if (self.num_k_heads, self.num_v_heads, self.head_k_dim, self.head_v_dim, self.conv_kernel_size) \
                != (16, 32, 128, 128, 4):
            return None
        if conv_w.dtype != torch.bfloat16 or not conv_w.is_contiguous() or \
                self.norm.weight.dtype != torch.bfloat16 or self.norm.activation != 'silu':
            return None
        if self.A_log.dtype != torch.float32 or self.dt_bias.dtype != torch.float32:
            return None
        if qkvz.dtype != torch.bfloat16 or ba.dtype != torch.bfloat16 or qkvz.stride(-1) != 1 \
                or ba.stride(-1) != 1:
            return None
        # FLA numbers varlen sequences by counting chunk-0 entries, so with an empty sequence its
        # chunk kernels attribute later chunks to the wrong sequence; the CUDA kernels are per
        # sequence. Keep such batches on the original path (checked before any state change).
        if not is_decode and has_empty_sequence(fla.cu_seqlens):
            return None
        lib = get_gdn_lib()
        rec = pool.recurrent_states[li]
        conv_states = pool.conv_states[li]
        if is_decode:
            res = lib.decode_fused(qkvz, ba, conv_w, conv_states, self.A_log, self.dt_bias, rec,
                                   fla.cache_indices, self.norm.weight, in_scale, self.norm.eps,
                                   self.head_k_dim ** -0.5, variant=self._cuda_gdn_decode)
            qa = res["fp8"]
        else:
            conv_in = qkvz[:, :self.conv_dim]
            z = qkvz[:, self.conv_dim:]
            total = qkvz.shape[0]
            q, k, v = lib.conv_qkv(conv_in, conv_w, conv_states, fla.cu_seqlens, fla.cache_indices,
                                   fla.has_initial_state, key_dim=self.key_dim, value_dim=self.value_dim)
            g, beta = lib.gate_cumsum(ba, self.A_log, self.dt_bias, fla.cu_seqlens,
                                      num_heads=self.num_v_heads)
            if fla.fresh_state_indices is not None:
                rec.index_fill_(0, fla.fresh_state_indices, 0.0)
            track = fla.track_dst is not None
            fp32_snapshot = track and self._fp32_gdn_snapshot
            chunk_args = (
                q.view(1, total, self.num_k_heads, self.head_k_dim),
                k.view(1, total, self.num_k_heads, self.head_k_dim),
                v.view(1, total, self.num_v_heads, self.head_v_dim),
                g.view(1, total, self.num_v_heads), beta.view(1, total, self.num_v_heads),
                rec, fla.cache_indices, fla.cu_seqlens, self.head_k_dim ** -0.5)
            snap_kw = dict(snapshot_h_rows=fla.track_h_row if fp32_snapshot else None,
                           snapshot_state_indices=fla.track_dst if fp32_snapshot else None)
            core_out = h = None
            if self._cuda_gdn_fla is not None and (not track or fp32_snapshot):
                # fused bit-exact FLA kernels (incl. FP32 snapshot); None = unsupported, nothing
                # touched. The bf16 h is never materialised, so the BF16 track snapshot (track
                # without FP32 snapshot) keeps the Triton kernels.
                core_out = lib.chunk_fused(*chunk_args, **snap_kw, variant=self._cuda_gdn_fla)
            if core_out is None:
                # same FLA kernels (kkt_solve, recompute_w_u, chunk_h incl. FP32 snapshot, chunk_o)
                core_out, h = lib.chunk_prenormed(*chunk_args, **snap_kw)
            if track:
                self._write_track_snapshot(pool, li, conv_in, h, fla,
                                           recurrent_already_stored=fp32_snapshot)
            _, qa = lib.gated_norm(core_out, z, self.norm.weight, self.norm.eps, in_scale=in_scale,
                                   heads=self.num_v_heads)
        # identical to the uniform-scale branch of fp8_pertensor_linear's _scaled_mm
        return torch._scaled_mm(qa, weight.t(), scale_a=in_scale.reshape(()),
                                scale_b=weight_scale[0].reshape(()), out_dtype=torch.bfloat16)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        total = hidden_states.shape[0]
        dtype = hidden_states.dtype

        # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
        # built once and shared by all GDN layers. The scheduler/graph set it; build it
        # lazily here (cached on the batch) for direct-op callers (tests).
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla

        if self._split_in_proj:
            qkvz = self.in_proj_qkvz.forward(hidden_states)
            conv_in, z = torch.split(qkvz, [self.conv_dim, self.value_dim], dim=-1)
            ba = self.in_proj_ba.forward(hidden_states)
            b, a = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)
        else:
            proj = self.in_proj.forward(hidden_states)
            conv_in, z, b, a = torch.split(proj, self._in_proj_split, dim=-1)
        li = pool.local_index(self.layer_id)
        if self._cuda_gdn and self._split_in_proj:
            out = self._forward_cuda(qkvz, ba, fla, pool, li, batch.is_decode)
            if out is not None:
                return out
        z = z.reshape(total, self.num_v_heads, self.head_v_dim)

        if batch.is_decode:
            # Fused fla decode kernel: gating + in-kernel l2norm + recurrent update +
            # per-request state read/write-by-index, all in one kernel (no gather/scatter,
            # no clone, no external l2norm). q/k stay at num_k_heads (kernel handles GQA).
            mixed = self._conv_decode(conv_in, fla.cache_indices, pool)  # [B, conv_dim]
            B = mixed.shape[0]
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, B, self.num_v_heads, self.head_v_dim).to(dtype)
            core_out = gdn_decode_fla(
                q, k, v, a, b, A_log=self.A_log, dt_bias=self.dt_bias,
                state_source=pool.recurrent_states[li], indices=fla.cache_indices,
                cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
            )
        else:
            mixed = self._conv_prefill(
                conv_in, pool, fla.cu_seqlens, fla.cache_indices, fla.has_initial_state)
            # fla chunk handles GQA in-kernel: q/k stay at num_k_heads, v at num_v_heads.
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, total, self.num_v_heads, self.head_v_dim).to(dtype)
            g, beta = self._gate_params(a, b)
            g = g.reshape(1, total, self.num_v_heads)
            beta = beta.float().reshape(1, total, self.num_v_heads)
            # The chunk kernel reads + writes back initial_state[cache_indices] in place;
            # fresh sequences (cached_len==0) must start from a zeroed slot.
            if fla.fresh_state_indices is not None:
                pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
            track = fla.track_dst is not None
            fp32_snapshot = track and self._fp32_gdn_snapshot
            result = gdn_prefill_chunk_fla(
                q, k, v, g, beta,
                state_source=pool.recurrent_states[li], indices=fla.cache_indices,
                cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
                return_h=track,
                snapshot_h_rows=fla.track_h_row if fp32_snapshot else None,
                snapshot_state_indices=fla.track_dst if fp32_snapshot else None,
            )
            if track:
                core_out, h = result
                self._write_track_snapshot(pool, li, conv_in, h, fla,
                                           recurrent_already_stored=fp32_snapshot)
            else:
                core_out = result

        core_out = core_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = self.norm.forward(core_out, z).reshape(total, -1)
        return self.out_proj.forward(out)


__all__ = ["Qwen3_5GatedDeltaNet"]
