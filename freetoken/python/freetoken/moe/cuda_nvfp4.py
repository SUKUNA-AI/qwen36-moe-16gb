"""Opt-in local CUDA prefill prototype; never imported by production by default."""
import ctypes
import os
import pathlib

import torch
from freetoken.moe.fused import moe_align_block_size
from freetoken.kernel import moe_sum_reduce_triton


class Args(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        "a", "packed", "scales", "globals", "out", "route_weights", "sorted_routes", "expert_slots", "padded_count")]
    _fields_ += [(name, ctypes.c_int64) for name in ("a_row_stride", "packed_slot_stride", "scale_slot_stride", "global_slot_stride")]
    _fields_ += [(name, ctypes.c_int32) for name in ("n", "k", "sorted_capacity", "valid_routes", "input_topk", "multiply_weight")]


class CudaPrefill:
    def __init__(self, library=None, tile=(32, 64, 64)):
        path = library or os.environ.get("QWEN36_CUDA_NVFP4_LIBRARY", os.path.expanduser("~/qwen36-api/optimization/cuda-nvfp4/libqwen_nvfp4_prefill.so"))
        self.lib = ctypes.CDLL(str(pathlib.Path(path).resolve()))
        self.lib.qwen_nvfp4_prefill.argtypes = [ctypes.POINTER(Args), *([ctypes.c_int] * 4), ctypes.c_void_p]
        self.lib.qwen_nvfp4_prefill.restype = ctypes.c_int
        self.lib.qwen_nvfp4_cuda_error.argtypes = [ctypes.c_int]
        self.lib.qwen_nvfp4_cuda_error.restype = ctypes.c_char_p
        # Args.globals is an untyped pointer: a pre-fp32 library would misread fp32 banks
        probe = getattr(self.lib, "qwen_nvfp4_global_bits", None)
        if probe is None:
            raise RuntimeError(f"{path} predates fp32 NVFP4 globals (no qwen_nvfp4_global_bits); rebuild it")
        probe.restype = ctypes.c_int
        if probe() != 32:
            raise RuntimeError(f"{path} reads {probe()}-bit NVFP4 globals, FreeToken stores fp32")
        self.tile = tile

    def gemm(self, a, packed, scale, glob, out, weights, sorted_ids, expert_ids, ntpp,
             valid_routes, input_topk, multiply_weight=False, fused_silu=False):
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
        if k not in (512, 2048) or n not in (1024, 2048) or valid_routes % 8:
            raise ValueError("Prototype is scoped to Qwen H2048/I512/top8 geometry")
        slots = packed.shape[0]
        output_n = n // 2 if fused_silu else n
        if scale.shape != (slots, n, k // 16) or glob.shape != (slots, n):
            raise ValueError("Scale/global bank shapes must match packed weights")
        if out.numel() != valid_routes * output_n or weights.numel() != valid_routes:
            raise ValueError("Output and router weights must cover every route")
        if a.shape[1] != k or a.shape[0] * input_topk != valid_routes:
            raise ValueError("Input rows and top-k must cover every route")
        if ntpp.numel() != 1 or expert_ids.numel() < (sorted_ids.numel() + self.tile[0] - 1) // self.tile[0]:
            raise ValueError("Padded route metadata capacity is inconsistent")
        args = Args(*(t.data_ptr() for t in tensors), a.stride(0), packed.stride(0), scale.stride(0), glob.stride(0),
                    n, k, sorted_ids.numel(), valid_routes, input_topk, int(multiply_weight))
        code = self.lib.qwen_nvfp4_prefill(ctypes.byref(args), *self.tile, int(fused_silu),
                                          torch.cuda.current_stream(a.device).cuda_stream)
        if code:
            raise RuntimeError(self.lib.qwen_nvfp4_cuda_error(code).decode())

    def __call__(self, x, gate_packed, gate_scale, gate_global, down_packed, down_scale,
                 down_global, weights, ids, num_experts, activation="silu",
                 apply_router_weight_on_input=False, act_alpha=1.702, act_limit=7.0):
        if activation != "silu" or ids.shape[1] != 8:
            raise ValueError("Only original Qwen SiLU and top-k=8 are supported")
        m, hidden = x.shape
        if m < 1 or hidden != 2048 or ids.shape != (m, 8) or weights.shape != (m, 8):
            raise ValueError("Expected non-empty H2048 input and eight router outputs per token")
        sorted_ids, expert_ids, ntpp = moe_align_block_size(ids, self.tile[0], num_experts)
        intermediate = torch.empty((m * 8, 512), device=x.device, dtype=x.dtype)
        routes = torch.empty((m, 8, hidden), device=x.device, dtype=x.dtype)
        out = torch.empty_like(x)
        self.gemm(x, gate_packed, gate_scale, gate_global, intermediate, weights.reshape(-1),
                  sorted_ids, expert_ids, ntpp, m * 8, 8, apply_router_weight_on_input, True)
        self.gemm(intermediate, down_packed, down_scale, down_global, routes, weights.reshape(-1),
                  sorted_ids, expert_ids, ntpp, m * 8, 1, not apply_router_weight_on_input, False)
        moe_sum_reduce_triton(routes, out)
        return out
