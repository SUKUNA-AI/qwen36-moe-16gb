"""ctypes wrapper for libqwen_nvfp4_bitexact.so (opt-in; never imported by production).

`BitExactPrefill` keeps the calling convention of the previous opt-in wrapper
(freetoken/moe/cuda_nvfp4.py: CudaPrefill.gemm / __call__) so that the same C ABI
(`qwen_nvfp4_prefill`) can be loaded through QWEN36_CUDA_NVFP4_LIBRARY.  Extra entry
points (oracle, explicit variants, SiLU/sum replicas, dequant fixture) are for tests.
"""
import ctypes
import os
import pathlib

import torch

DEFAULT_LIBRARY = str(pathlib.Path(__file__).resolve().parents[1] / "build" / "libqwen_nvfp4_bitexact.so")


class Args(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        "a", "packed", "scales", "globals", "out", "route_weights", "sorted_routes", "expert_slots", "padded_count")]
    _fields_ += [(name, ctypes.c_int64) for name in ("a_row_stride", "packed_slot_stride", "scale_slot_stride", "global_slot_stride")]
    _fields_ += [(name, ctypes.c_int32) for name in ("n", "k", "sorted_capacity", "valid_routes", "input_topk", "multiply_weight")]


class BitExactPrefill:
    def __init__(self, library=None, tile=(64, 64, 64), variant=-1, cuda_sum=False):
        path = library or os.environ.get("QWEN36_CUDA_NVFP4_LIBRARY", DEFAULT_LIBRARY)
        self.path = str(pathlib.Path(path).resolve())
        self.lib = ctypes.CDLL(self.path)
        L = self.lib
        L.qwen_nvfp4_prefill.argtypes = [ctypes.POINTER(Args), *([ctypes.c_int] * 4), ctypes.c_void_p]
        L.qwen_nvfp4_prefill.restype = ctypes.c_int
        L.qwen_nvfp4_cuda_error.argtypes = [ctypes.c_int]
        L.qwen_nvfp4_cuda_error.restype = ctypes.c_char_p
        probe = getattr(L, "qwen_nvfp4_global_bits", None)
        if probe is None or probe() != 32:
            raise RuntimeError(f"{self.path} does not read fp32 NVFP4 globals (pre-v4 build)")
        self.tile = tuple(tile)
        self.variant = variant
        self.cuda_sum = cuda_sum
        try:
            L.qwen_nvfp4_prefill_variant
        except AttributeError:   # an older library: only the original C ABI is available
            self.extended = False
            return
        self.extended = True
        L.qwen_nvfp4_prefill_variant.argtypes = [ctypes.POINTER(Args), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        L.qwen_nvfp4_prefill_variant.restype = ctypes.c_int
        L.qwen_nvfp4_oracle.argtypes = [ctypes.POINTER(Args), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        L.qwen_nvfp4_oracle.restype = ctypes.c_int
        L.qwen_nvfp4_silu_and_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p]
        L.qwen_nvfp4_silu_and_mul.restype = ctypes.c_int
        L.qwen_nvfp4_moe_sum.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        L.qwen_nvfp4_moe_sum.restype = ctypes.c_int
        L.qwen_nvfp4_dequant_fixture.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        L.qwen_nvfp4_dequant_fixture.restype = ctypes.c_int
        L.qwen_nvfp4_variant_count.restype = ctypes.c_int
        L.qwen_nvfp4_variant_bm.argtypes = [ctypes.c_int]
        L.qwen_nvfp4_variant_bm.restype = ctypes.c_int
        L.qwen_nvfp4_build_id.restype = ctypes.c_char_p
        L.qwen_nvfp4_mma_peak.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        L.qwen_nvfp4_mma_peak.restype = ctypes.c_int

    # ------------------------------------------------------------------ helpers
    def _check(self, code):
        if code:
            raise RuntimeError(self.lib.qwen_nvfp4_cuda_error(code).decode())

    @staticmethod
    def _stream(t):
        return torch.cuda.current_stream(t.device).cuda_stream

    def build_id(self):
        return self.lib.qwen_nvfp4_build_id().decode()

    def variant_count(self):
        return self.lib.qwen_nvfp4_variant_count()

    def variant_bm(self, v):
        return self.lib.qwen_nvfp4_variant_bm(v)

    def make_args(self, a, packed, scale, glob, out, weights, sorted_ids, expert_ids, ntpp,
                  valid_routes, input_topk, multiply_weight):
        tensors = (a, packed, scale, glob, out, weights, sorted_ids, expert_ids, ntpp)
        if not all(t.is_cuda and t.device == a.device for t in tensors):
            raise ValueError("All tensors must reside on the same CUDA device")
        if a.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
            raise ValueError("BF16 activations and output are required")
        if packed.dtype != torch.uint8 or scale.dtype != torch.float8_e4m3fn or glob.dtype != torch.float32:
            raise ValueError("Expected unchanged uint8 NVFP4, E4M3 scales, FP32 globals")
        if any(t.dtype != torch.int32 for t in (sorted_ids, expert_ids, ntpp)) or weights.dtype != torch.float32:
            raise ValueError("Expected int32 metadata and float32 router weights")
        if not all(t.is_contiguous() for t in tensors[1:]) or a.stride(1) != 1:
            raise ValueError("Banks and metadata must be contiguous; activation row stride is supported")
        n, k = packed.shape[1], packed.shape[2] * 2
        return Args(*(t.data_ptr() for t in tensors), a.stride(0), packed.stride(0), scale.stride(0), glob.stride(0),
                    n, k, sorted_ids.numel(), valid_routes, input_topk, int(multiply_weight))

    # ------------------------------------------------------------------ old ABI
    def gemm(self, a, packed, scale, glob, out, weights, sorted_ids, expert_ids, ntpp,
             valid_routes, input_topk, multiply_weight=False, fused_silu=False, variant=None, align=None):
        args = self.make_args(a, packed, scale, glob, out, weights, sorted_ids, expert_ids, ntpp,
                              valid_routes, input_topk, multiply_weight)
        variant = self.variant if variant is None else variant
        align = self.tile[0] if align is None else align
        code =self.lib.qwen_nvfp4_prefill_variant(ctypes.byref(args), align, variant, int(fused_silu), self._stream(a))
        self._check(code)

    def gemm_abi(self, a, packed, scale, glob, out, weights, sorted_ids, expert_ids, ntpp,
                 valid_routes, input_topk, multiply_weight=False, fused_silu=False):
        """Exactly the call the previous opt-in wrapper makes."""
        args = self.make_args(a, packed, scale, glob, out, weights, sorted_ids, expert_ids, ntpp,
                              valid_routes, input_topk, multiply_weight)
        self._check(self.lib.qwen_nvfp4_prefill(ctypes.byref(args), *self.tile, int(fused_silu), self._stream(a)))

    def oracle(self, a, packed, scale, glob, out, weights, sorted_ids, expert_ids, ntpp,
               valid_routes, input_topk, multiply_weight=False, align=None, mode=0):
        args = self.make_args(a, packed, scale, glob, out, weights, sorted_ids, expert_ids, ntpp,
                              valid_routes, input_topk, multiply_weight)
        self._check(self.lib.qwen_nvfp4_oracle(ctypes.byref(args), self.tile[0] if align is None else align, mode,
                                               self._stream(a)))

    def silu_and_mul(self, x, out):
        rows, two_d = x.shape
        assert x.is_contiguous() and out.is_contiguous() and x.dtype == torch.bfloat16
        self._check(self.lib.qwen_nvfp4_silu_and_mul(x.data_ptr(), out.data_ptr(), rows, two_d // 2, self._stream(x)))

    def moe_sum(self, routes, out):
        tokens, topk, hidden = routes.shape
        assert routes.is_contiguous() and out.is_contiguous()
        self._check(self.lib.qwen_nvfp4_moe_sum(routes.data_ptr(), out.data_ptr(), tokens, topk, hidden, self._stream(routes)))

    def mma_peak(self, scratch, blocks, iters):
        self._check(self.lib.qwen_nvfp4_mma_peak(scratch.data_ptr(), blocks, iters, self._stream(scratch)))

    def dequant_fixture(self, words, scales):
        out = torch.empty((words.numel(), 8), device=words.device, dtype=torch.int16)
        self._check(self.lib.qwen_nvfp4_dequant_fixture(words.data_ptr(), scales.data_ptr(), out.data_ptr(),
                                                        words.numel(), self._stream(words)))
        return out

    # ------------------------------------------------------------------ full layer
    def __call__(self, x, gate_packed, gate_scale, gate_global, down_packed, down_scale,
                 down_global, weights, ids, num_experts, activation="silu",
                 apply_router_weight_on_input=False, act_alpha=1.702, act_limit=7.0):
        from freetoken.moe.fused import moe_align_block_size
        from freetoken.kernel import moe_sum_reduce_triton
        if activation != "silu" or ids.shape[1] != 8:
            raise ValueError("Only original Qwen SiLU and top-k=8 are supported")
        m, hidden = x.shape
        inter = gate_packed.shape[1] // 2
        sorted_ids, expert_ids, ntpp = moe_align_block_size(ids, self.tile[0], num_experts)
        intermediate = torch.empty((m * 8, inter), device=x.device, dtype=x.dtype)
        routes = torch.empty((m, 8, hidden), device=x.device, dtype=x.dtype)
        out = torch.empty_like(x)
        tw = weights.reshape(-1)
        self.gemm(x, gate_packed, gate_scale, gate_global, intermediate, tw,
                  sorted_ids, expert_ids, ntpp, m * 8, 8, apply_router_weight_on_input, True)
        self.gemm(intermediate, down_packed, down_scale, down_global, routes, tw,
                  sorted_ids, expert_ids, ntpp, m * 8, 1, not apply_router_weight_on_input, False)
        if self.cuda_sum:
            self.moe_sum(routes, out)
        else:
            moe_sum_reduce_triton(routes, out)
        return out
