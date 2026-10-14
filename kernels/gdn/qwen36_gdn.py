"""ctypes wrapper for libqwen36_gdn.so: bit-exact fused GDN prefill/decode kernels (SM120).

Opt-in only. Every function validates layouts/dtypes and raises on anything it does not
support, so callers can keep the original FLA/sgl_kernel/torch path as the authority.
No host synchronisation, no allocation inside CUDA-graph capture except via torch's caching
allocator (same as the original ops).
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

_VP, _I64, _I32, _F32, _U64 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32, ctypes.c_float, ctypes.c_uint64
_ERRORS = {1: "bad sizes", 2: "bad dims", 3: "bad row stride", 5: "bad state stride",
           6: "misaligned tensor", 7: "bad tile/variant", 8: "grid too large", 9: "CUDA launch failed"}
PAD_SLOT_ID = -1
ABI_VERSION = 2         # 2: + qwen_gdn_fla_chunk_ho / qwen_gdn_fla_wu_chunk_ho (fused FLA kernels)
L2_EPS = 1e-6           # l2norm_fwd default eps (fp32 1e-6)
CHUNK = 64


def _ptr(t):
    return None if t is None else ctypes.c_void_p(t.data_ptr())


_EMPTY_SEQ = None


def has_empty_sequence(cu: torch.Tensor) -> bool:
    """True if some varlen sequence is empty. FLA's prepare_chunk_indices numbers sequences by
    counting chunk-0 entries, so every chunk after an empty sequence is attributed to the wrong
    sequence by kkt_solve / recompute_w_u / chunk_o (chunk_h is per sequence and unaffected); the
    fused path must not run then. Cached per cu tensor object (one host sync per batch, like
    prepare_chunk_indices)."""
    global _EMPTY_SEQ
    if _EMPTY_SEQ is None:
        from freetoken.kernel.fla.utils import tensor_cache

        @tensor_cache
        def _empty(c):
            return bool((c[1:] == c[:-1]).any().item())
        _EMPTY_SEQ = _empty
    return _EMPTY_SEQ(cu)


class GdnLib:
    def __init__(self, library: str | None = None):
        path = library or os.environ.get("QWEN36_CUDA_GDN_LIBRARY") or str(
            Path(__file__).with_name("libqwen36_gdn.so"))
        self.path = path
        self.lib = ctypes.CDLL(path)
        L = self.lib
        L.qwen_gdn_abi_version.restype = ctypes.c_int
        if L.qwen_gdn_abi_version() != ABI_VERSION:
            raise RuntimeError("libqwen36_gdn ABI mismatch")
        L.qwen_gdn_fla_chunk_ho.argtypes = [_VP] * 9 + [_I32, _VP, _VP, _I32, _F32, _VP, _VP]
        L.qwen_gdn_fla_chunk_ho.restype = ctypes.c_int
        L.qwen_gdn_fla_wu_chunk_ho.argtypes = [_VP] * 10 + [_I32, _VP, _VP, _I32, _F32, _VP, _VP]
        L.qwen_gdn_fla_wu_chunk_ho.restype = ctypes.c_int
        L.qwen_gdn_conv_qkv.argtypes = [_VP, _I64, _VP, _VP, _I64, _VP, _VP, _VP, _VP, _VP, _VP,
                                        _I32, _I32, _I32, _I32, _I32, _I32, _I32, _F32, _I32, _I32, _VP]
        L.qwen_gdn_conv_qkv.restype = ctypes.c_int
        L.qwen_gdn_gate_cumsum.argtypes = [_VP, _I64, _VP, _VP, _VP, _VP, _VP, _I32, _I32, _I32, _I32, _VP]
        L.qwen_gdn_gate_cumsum.restype = ctypes.c_int
        L.qwen_gdn_gated_norm.argtypes = [_VP, _VP, _I64, _VP, _VP, _VP, _VP, _I32, _I32, _F32, _I32, _VP]
        L.qwen_gdn_gated_norm.restype = ctypes.c_int
        L.qwen_gdn_silu_eval.argtypes = [_U64, _U64, _VP, _VP]
        L.qwen_gdn_silu_eval.restype = ctypes.c_int
        L.qwen_gdn_gate_eval.argtypes = [_VP, _VP, _I32, _VP, _VP, _VP]
        L.qwen_gdn_gate_eval.restype = ctypes.c_int
        L.qwen_gdn_silu_points.argtypes = [_VP, _VP, _I64, _VP]
        L.qwen_gdn_silu_points.restype = ctypes.c_int
        L.qwen_gdn_silu_compare.argtypes = [_U64, _U64, _VP, _VP]
        L.qwen_gdn_silu_compare.restype = ctypes.c_int
        for fn in (L.qwen_gdn_decode_fused, L.qwen_gdn_decode_cluster):
            fn.argtypes = [_VP] * 14 + [_I64, _I64, _I64, _I32, _I32, _F32, _F32, _VP]
            fn.restype = ctypes.c_int

    @staticmethod
    def _check(code: int, what: str):
        if code:
            raise RuntimeError(f"{what}: {_ERRORS.get(code, code)}")

    # ---------------------------------------------------------------------------------------
    def conv_qkv(self, conv_in, weight, conv_states, cu_seqlens, cache_indices, has_initial_state,
                 key_dim=2048, value_dim=4096, norm_qk=True, tile_tokens=16, batch_rows=4,
                 out=None):
        """conv_in [T, C] (row stride >= C, e.g. a column slice of the qkvz GEMM output).
        Returns (q, k, v) contiguous [T, key_dim], [T, key_dim], [T, value_dim]; q, k are
        L2-normalised exactly as FLA l2norm_fwd when norm_qk. Updates conv_states in place."""
        T, C = conv_in.shape
        if conv_in.dtype != torch.bfloat16 or conv_in.stride(1) != 1:
            raise ValueError("conv_in must be BF16 with unit inner stride")
        if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (C, 4) or not weight.is_contiguous():
            raise ValueError("weight must be contiguous BF16 [C, 4]")
        if conv_states.dtype != torch.bfloat16 or conv_states.shape[1:] != (C, 3) or \
                conv_states.stride(2) != 1 or conv_states.stride(1) != 3:
            raise ValueError("conv_states must be BF16 [S, C, 3] with (c,l) strides (3,1)")
        if cu_seqlens.dtype != torch.int64 or cache_indices.dtype != torch.int32 or \
                has_initial_state.dtype != torch.bool:
            raise ValueError("cu_seqlens int64, cache_indices int32, has_initial_state bool required")
        for t in (cu_seqlens, cache_indices, has_initial_state):
            if not t.is_contiguous():
                raise ValueError("metadata must be contiguous")
        B = cache_indices.numel()
        if out is None:
            q = torch.empty(T, key_dim, dtype=torch.bfloat16, device=conv_in.device)
            k = torch.empty(T, key_dim, dtype=torch.bfloat16, device=conv_in.device)
            v = torch.empty(T, value_dim, dtype=torch.bfloat16, device=conv_in.device)
        else:
            q, k, v = out
        err = self.lib.qwen_gdn_conv_qkv(
            _ptr(conv_in), conv_in.stride(0), _ptr(weight), _ptr(conv_states), conv_states.stride(0),
            _ptr(cu_seqlens), _ptr(cache_indices), _ptr(has_initial_state), _ptr(q), _ptr(k), _ptr(v),
            T, B, C, key_dim, value_dim, PAD_SLOT_ID, int(norm_qk), L2_EPS, tile_tokens, batch_rows,
            ctypes.c_void_p(torch.cuda.current_stream(conv_in.device).cuda_stream))
        self._check(err, "qwen_gdn_conv_qkv")
        return q, k, v

    def gate_cumsum(self, ba, A_log, dt_bias, cu_seqlens, num_heads=32, out=None):
        """ba [T, 2H] BF16 (b | a). Returns (g_cumsum [T, H] fp32, beta [T, H] fp32)."""
        T = ba.shape[0]
        H = num_heads
        if ba.dtype != torch.bfloat16 or ba.stride(1) != 1 or ba.shape[1] < 2 * H:
            raise ValueError("ba must be BF16 [T, >=2H] with unit inner stride")
        if A_log.dtype != torch.float32 or dt_bias.dtype != torch.float32 or \
                A_log.numel() != H or dt_bias.numel() != H or not A_log.is_contiguous() or \
                not dt_bias.is_contiguous():
            raise ValueError("A_log / dt_bias must be contiguous fp32 [H]")
        B = cu_seqlens.numel() - 1
        if out is None:
            g = torch.empty(T, H, dtype=torch.float32, device=ba.device)
            beta = torch.empty(T, H, dtype=torch.float32, device=ba.device)
        else:
            g, beta = out
        num_chunks = (T + CHUNK - 1) // CHUNK + B   # host upper bound, no sync
        err = self.lib.qwen_gdn_gate_cumsum(
            _ptr(ba), ba.stride(0), _ptr(A_log), _ptr(dt_bias), _ptr(cu_seqlens), _ptr(g), _ptr(beta),
            T, B, H, num_chunks, ctypes.c_void_p(torch.cuda.current_stream(ba.device).cuda_stream))
        self._check(err, "qwen_gdn_gate_cumsum")
        return g, beta

    @staticmethod
    def norm_lanes(M: int, device) -> int:
        """Triton rms_norm_gated ROWS_PER_BLOCK rule: next_pow2(cdiv(M, 2*SMs)) clipped to 4;
        ROWS_PER_BLOCK == 1 uses the 32-lane reduction layout, >= 2 the 16-lane one."""
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        rpb = 1
        need = (M + 2 * sms - 1) // (2 * sms)
        while rpb < need:
            rpb <<= 1
        return 32 if min(rpb, 4) == 1 else 16

    def gated_norm(self, o, z, norm_weight, eps, in_scale=None, want_bf16=False, heads=32, out=None):
        """o [T, heads, 128] (or [T*heads,128]) BF16 contiguous; z [T, heads*128] BF16 view with unit
        inner stride (row stride may be the qkvz row). Returns (y_bf16 or None, q_fp8 or None)."""
        o2 = o.reshape(-1, 128)
        M = o2.shape[0]
        T = z.shape[0]
        if o2.dtype != torch.bfloat16 or not o2.is_contiguous():
            raise ValueError("o must be contiguous BF16")
        if z.dtype != torch.bfloat16 or z.stride(-1) != 1 or z.shape[-1] != heads * 128 or T * heads != M:
            raise ValueError("z must be BF16 [T, heads*128] with unit inner stride")
        if norm_weight.dtype != torch.bfloat16 or norm_weight.numel() != 128 or not norm_weight.is_contiguous():
            raise ValueError("norm weight must be contiguous BF16 [128]")
        y = q = None
        if out is not None:
            y, q = out
        else:
            if want_bf16:
                y = torch.empty(T, heads * 128, dtype=torch.bfloat16, device=o.device)
            if in_scale is not None:
                q = torch.empty(T, heads * 128, dtype=torch.float8_e4m3fn, device=o.device)
        lanes = self.norm_lanes(M, o.device)
        err = self.lib.qwen_gdn_gated_norm(
            _ptr(o2), _ptr(z), z.stride(0), _ptr(norm_weight), _ptr(in_scale), _ptr(y), _ptr(q),
            M, heads, float(eps), lanes, ctypes.c_void_p(torch.cuda.current_stream(o.device).cuda_stream))
        self._check(err, "qwen_gdn_gated_norm")
        return y, q

    def decode_fused(self, qkvz, ba, conv_w, conv_states, A_log, dt_bias, rec_states, idx,
                     norm_w, in_scale, eps, scale, want_fp8=True, want_bf16=False,
                     want_core=False, want_mixed=False, out=None, variant="cluster"):
        """One decode token per sequence. qkvz [B, >=12288] BF16 (conv_in | z), ba [B, >=64] BF16
        (b | a). Updates conv_states [S, 8192, 3] and rec_states [S, 32, 128, 128] in place.
        Returns dict with fp8 [B, 4096] (out_proj input) and optional bf16/core/mixed."""
        B = qkvz.shape[0]
        if qkvz.dtype != torch.bfloat16 or qkvz.stride(1) != 1 or qkvz.shape[1] < 12288:
            raise ValueError("qkvz must be BF16 [B, >=12288] with unit inner stride")
        if ba.dtype != torch.bfloat16 or ba.stride(1) != 1 or ba.shape[1] < 64:
            raise ValueError("ba must be BF16 [B, >=64] with unit inner stride")
        if conv_states.dtype != torch.bfloat16 or conv_states.shape[1:] != (8192, 3) or \
                conv_states.stride(1) != 3 or conv_states.stride(2) != 1:
            raise ValueError("conv_states must be BF16 [S, 8192, 3]")
        if rec_states.dtype != torch.float32 or rec_states.shape[1:] != (32, 128, 128) or \
                not rec_states.is_contiguous():
            raise ValueError("rec_states must be contiguous fp32 [S, 32, 128, 128]")
        if idx.dtype != torch.int32 or not idx.is_contiguous() or idx.numel() != B:
            raise ValueError("idx must be contiguous int32 [B]")
        dev = qkvz.device
        res = dict(out or {})
        if want_fp8 and "fp8" not in res:
            res["fp8"] = torch.empty(B, 4096, dtype=torch.float8_e4m3fn, device=dev)
        if want_bf16 and "bf16" not in res:
            res["bf16"] = torch.empty(B, 4096, dtype=torch.bfloat16, device=dev)
        if want_core and "core" not in res:
            res["core"] = torch.empty(B, 32, 128, dtype=torch.bfloat16, device=dev)
        if want_mixed and "mixed" not in res:
            res["mixed"] = torch.empty(B, 8192, dtype=torch.bfloat16, device=dev)
        lanes = self.norm_lanes(B * 32, dev)
        fn = self.lib.qwen_gdn_decode_cluster if variant == "cluster" else self.lib.qwen_gdn_decode_fused
        err = fn(
            _ptr(qkvz), _ptr(ba), _ptr(conv_w), _ptr(conv_states), _ptr(A_log), _ptr(dt_bias),
            _ptr(rec_states), _ptr(idx), _ptr(norm_w), _ptr(in_scale), _ptr(res.get("fp8")),
            _ptr(res.get("bf16")), _ptr(res.get("core")), _ptr(res.get("mixed")),
            qkvz.stride(0), ba.stride(0), conv_states.stride(0), B, lanes, float(scale), float(eps),
            ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream))
        self._check(err, "qwen_gdn_decode_fused")
        return res

    # FLA chunk kernels on pre-normalised q/k and pre-cumsum'd g --------------------------------
    @staticmethod
    def chunk_prenormed(q, k, v, g_cumsum, beta, state_source, indices, cu_seqlens, scale,
                        snapshot_h_rows=None, snapshot_state_indices=None):
        """chunk_gated_delta_rule_fwd minus l2norm_fwd(q,k) and chunk_local_cumsum(g), which the
        CUDA kernels already produced bit-exactly. Same FLA Triton kernels and arguments:
        kkt_solve, recompute_w_u, chunk_h (FP32 snapshot), chunk_o. Two byte-neutral differences:
        chunk_o's output is torch.empty instead of torch.zeros (chunk_o writes every element of o:
        all chunks x heads x both V blocks, rows bounded by each sequence), and the chunk index
        table is reused for chunk_o when its BT is 64 (same values; avoids a second lookup).
        Returns (o [total, H, V], h)."""
        import triton
        from freetoken.kernel.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
        from freetoken.kernel.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
        from freetoken.kernel.fla.chunk_o import chunk_fwd_kernel_o
        from freetoken.kernel.fla.index import prepare_chunk_indices

        for t in (q, k, v, g_cumsum, beta, state_source):
            if not t.is_contiguous():
                raise ValueError("prenormed GDN prefill needs contiguous inputs")
        cu = cu_seqlens.to(torch.int64)
        ci = prepare_chunk_indices(cu, 64)
        w, u, _A = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g_cumsum, beta=beta, cu_seqlens=cu,
                                                    chunk_indices=ci)
        h, v_new = chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=g_cumsum, initial_state=state_source,
            initial_state_indices=indices.to(torch.int32), cu_seqlens=cu, chunk_indices=ci,
            snapshot_h_rows=snapshot_h_rows, snapshot_state_indices=snapshot_state_indices)
        # == chunk_fwd_o(q, k, v_new, h, g, scale, cu) with an uninitialised (fully written) o
        B, T, Hg, K, V = *q.shape, v_new.shape[-1]
        H = v_new.shape[-2]
        BT = min(64, max(16, triton.next_power_of_2(T)))
        ci_o = ci if BT == 64 else prepare_chunk_indices(cu, BT)
        o = torch.empty_like(v_new)
        grid = (triton.cdiv(V, 64), len(ci_o), B * H)
        chunk_fwd_kernel_o[grid](q, k, v_new, h, g_cumsum, o, cu, ci_o, scale, T=T, H=H, Hg=Hg, K=K,
                                 V=V, BT=BT, BK=128, BV=64, USE_G=True, IS_VARLEN=True,
                                 num_warps=4, num_stages=2)
        return o[0], h

    # Fused FLA chunk_h + chunk_o (ABI 2) -----------------------------------------------------
    @staticmethod
    def fla_supported(q, k, v, g_cumsum, beta, state_source, indices, cu_seqlens,
                      snapshot_h_rows=None, snapshot_state_indices=None) -> bool:
        """True when the fused kernel reproduces chunk_h + chunk_o bit-exactly for these inputs:
        Qwen3.6 GDN head shapes, varlen B == 1, chunk_o BT == 64 (T > 32), FP32 state pool,
        valid FP32-snapshot metadata. Anything else stays on the Triton kernels (which also
        raise the original errors for invalid snapshot metadata)."""
        if q.dim() != 4 or k.shape != q.shape or v.dim() != 4 or q.shape[0] != 1 or v.shape[0] != 1:
            return False
        _, T, Hg, K = q.shape
        H, V = v.shape[-2], v.shape[-1]
        if (Hg, K, H, V) != (16, 128, 32, 128) or T <= 32 or v.shape[1] != T:
            return False
        if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
            return False
        if g_cumsum.dtype != torch.float32 or beta.dtype != torch.float32 or \
                tuple(g_cumsum.shape) != (1, T, H) or tuple(beta.shape) != (1, T, H):
            return False
        if state_source is None or state_source.dtype != torch.float32 or state_source.dim() != 4 or \
                tuple(state_source.shape[1:]) != (H, V, K) or not state_source.is_contiguous():
            return False
        for t in (q, k, v, g_cumsum, beta):
            if not t.is_contiguous() or t.device != state_source.device:
                return False
        if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2 or cu_seqlens.numel() - 1 > 65535 or \
                indices.dim() != 1 or indices.numel() != cu_seqlens.numel() - 1:
            return False
        if (q.data_ptr() | k.data_ptr() | v.data_ptr() | state_source.data_ptr()) % 16:
            return False
        if indices.dtype not in (torch.int32, torch.int64) or \
                cu_seqlens.dtype not in (torch.int32, torch.int64):
            return False
        if (snapshot_h_rows is None) != (snapshot_state_indices is None):
            return False
        if snapshot_h_rows is not None:
            N = cu_seqlens.numel() - 1
            r, d = snapshot_h_rows, snapshot_state_indices
            if (r.ndim != 1 or d.shape != r.shape or r.device != q.device or d.device != q.device
                    or not r.is_contiguous() or not d.is_contiguous() or r.numel() > N
                    or r.dtype not in (torch.int32, torch.int64)
                    or d.dtype not in (torch.int32, torch.int64)):
                return False
        return True

    def fla_chunk_ho(self, q, k, w, u, g_cumsum, state, indices_i32, cu_i64, scale,
                     chunk_offsets=None, snap_rows_i64=None, snap_dst_i64=None, out=None):
        """Raw fused chunk_h + chunk_o: == chunk_gated_delta_rule_fwd_h (in-place final state,
        FP32 snapshot) + chunk_fwd_kernel_o on its h / v_new. Returns o [1, T, 32, 128] bf16."""
        T = q.shape[1]
        N = cu_i64.numel() - 1
        o = out if out is not None else torch.empty(1, T, 32, 128, dtype=torch.bfloat16, device=q.device)
        ns = 0 if snap_rows_i64 is None else int(snap_rows_i64.numel())
        err = self.lib.qwen_gdn_fla_chunk_ho(
            _ptr(q), _ptr(k), _ptr(w), _ptr(u), _ptr(g_cumsum), _ptr(state), _ptr(indices_i32),
            _ptr(snap_rows_i64), _ptr(snap_dst_i64), ns, _ptr(cu_i64), _ptr(chunk_offsets), N,
            float(scale), _ptr(o), ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream))
        self._check(err, "qwen_gdn_fla_chunk_ho")
        return o

    def fla_wu_chunk_ho(self, q, k, v, a_wy, beta, g_cumsum, state, indices_i32, cu_i64, scale,
                        chunk_offsets=None, snap_rows_i64=None, snap_dst_i64=None, out=None):
        """Raw fused recompute_w_u + chunk_h + chunk_o on the kkt_solve output a_wy."""
        T = q.shape[1]
        N = cu_i64.numel() - 1
        o = out if out is not None else torch.empty(1, T, 32, 128, dtype=torch.bfloat16, device=q.device)
        ns = 0 if snap_rows_i64 is None else int(snap_rows_i64.numel())
        err = self.lib.qwen_gdn_fla_wu_chunk_ho(
            _ptr(q), _ptr(k), _ptr(v), _ptr(a_wy), _ptr(beta), _ptr(g_cumsum), _ptr(state),
            _ptr(indices_i32), _ptr(snap_rows_i64), _ptr(snap_dst_i64), ns, _ptr(cu_i64),
            _ptr(chunk_offsets), N, float(scale), _ptr(o),
            ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream))
        self._check(err, "qwen_gdn_fla_wu_chunk_ho")
        return o

    @staticmethod
    def kkt_solve(k, g_cumsum, beta, cu_i64, chunk_indices):
        """Step 1 of chunk_gated_delta_rule_fwd_intra, verbatim (zero-initialised A: the upper
        blocks kkt_solve never writes are read as zeros by recompute_w_u)."""
        from freetoken.kernel.fla.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
        B, T, Hg, K = k.shape
        H = beta.shape[-1]
        A = torch.zeros(B, T, H, 64, device=k.device, dtype=k.dtype)
        chunk_gated_delta_rule_fwd_kkt_solve_kernel[(len(chunk_indices), B * H)](
            k=k, g=g_cumsum, beta=beta, A=A, cu_seqlens=cu_i64, chunk_indices=chunk_indices, T=T,
            H=H, Hg=Hg, K=K, BT=64, BC=16)
        return A

    def chunk_fused(self, q, k, v, g_cumsum, beta, state_source, indices, cu_seqlens, scale,
                    snapshot_h_rows=None, snapshot_state_indices=None, variant="wu"):
        """chunk_prenormed(...)[0] with FLA kernels replaced by fused CUDA kernels:
          variant "wu": kkt_solve (Triton) -> fused recompute_w_u + chunk_h + chunk_o (CUDA);
          variant "ho": kkt_solve + recompute_w_u (Triton) -> fused chunk_h + chunk_o (CUDA).
        The bf16 h tensor is never materialised, so callers that need it (the BF16 track
        snapshot) must use chunk_prenormed. Returns o [T, H, V] or None when unsupported
        (nothing has been touched then)."""
        if variant not in ("wu", "ho"):
            raise ValueError("variant must be 'wu' or 'ho'")
        if not self.fla_supported(q, k, v, g_cumsum, beta, state_source, indices, cu_seqlens,
                                  snapshot_h_rows, snapshot_state_indices):
            return None
        from freetoken.kernel.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
        from freetoken.kernel.fla.index import prepare_chunk_indices, prepare_chunk_offsets

        cu = cu_seqlens.to(torch.int64)
        if has_empty_sequence(cu):
            return None
        ci = prepare_chunk_indices(cu, 64)
        co = rows = dst = None
        if snapshot_h_rows is not None:
            co = prepare_chunk_offsets(cu, 64)
            rows = snapshot_h_rows.to(torch.int64)
            dst = snapshot_state_indices.to(torch.int64)
        idx = indices.to(torch.int32)
        if variant == "wu":
            A = self.kkt_solve(k, g_cumsum, beta, cu, ci)
            o = self.fla_wu_chunk_ho(q, k, v, A, beta, g_cumsum, state_source, idx, cu, scale,
                                     chunk_offsets=co, snap_rows_i64=rows, snap_dst_i64=dst)
        else:
            w, u, _A = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g_cumsum, beta=beta, cu_seqlens=cu,
                                                        chunk_indices=ci)
            o = self.fla_chunk_ho(q, k, w, u, g_cumsum, state_source, idx, cu, scale,
                                  chunk_offsets=co, snap_rows_i64=rows, snap_dst_i64=dst)
        return o[0]

    # exhaustive-check helpers ----------------------------------------------------------------
    def silu_eval(self, begin: int, count: int, out: torch.Tensor):
        err = self.lib.qwen_gdn_silu_eval(begin, count, _ptr(out),
                                          ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        self._check(err, "qwen_gdn_silu_eval")

    def silu_points(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().contiguous()
        out = torch.empty_like(x)
        err = self.lib.qwen_gdn_silu_points(_ptr(x), _ptr(out), x.numel(),
                                            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        self._check(err, "qwen_gdn_silu_points")
        return out

    def silu_full_range_mismatches(self) -> int:
        cnt = torch.zeros(1, dtype=torch.int64, device="cuda")
        step = 1 << 30
        for begin in range(0, 1 << 32, step):
            err = self.lib.qwen_gdn_silu_compare(begin, step, _ptr(cnt),
                                                 ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
            self._check(err, "qwen_gdn_silu_compare")
        return int(cnt.item())

    def gate_eval(self, A_log, dt_bias):
        H = A_log.numel()
        graw = torch.empty(H, 65536, dtype=torch.float32, device=A_log.device)
        beta = torch.empty(65536, dtype=torch.float32, device=A_log.device)
        err = self.lib.qwen_gdn_gate_eval(_ptr(A_log), _ptr(dt_bias), H, _ptr(graw), _ptr(beta),
                                          ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        self._check(err, "qwen_gdn_gate_eval")
        return graw, beta
