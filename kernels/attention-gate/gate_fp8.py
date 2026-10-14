"""Standalone opt-in fixture wrapper; no runtime monkey patch or global changes."""
import ctypes
from pathlib import Path
import torch

class GateFp8:
    def __init__(self, library=None):
        self.lib = ctypes.CDLL(str(library or Path(__file__).with_name('build')/'libqwen_attention_gate.so'))
        self.fn = self.lib.qwen_attention_gate_fp8
        self.fn.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int64]*4 + [ctypes.c_void_p]
        self.fn.restype = ctypes.c_int
        self.lib.qwen_gate_error.argtypes = [ctypes.c_int]
        self.lib.qwen_gate_error.restype = ctypes.c_char_p

    def __call__(self, attention, gate, scale, debug=False):
        if attention.ndim != 2 or attention.shape != gate.shape or attention.shape[1] != 4096:
            raise ValueError('Expected matching [M,4096] full-attention tensors')
        if attention.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
            raise ValueError('BF16 inputs required to preserve original intermediate rounding')
        if scale.dtype != torch.float32 or scale.numel() != 1 or not scale.is_contiguous():
            raise ValueError('One checkpoint FP32 input scale required')
        if not all(t.is_cuda and t.device == attention.device for t in (attention, gate, scale)):
            raise ValueError('All tensors must share one CUDA device')
        if attention.stride(1) != 1 or gate.stride(1) != 1:
            raise ValueError('Innermost stride must be one')
        output = torch.empty(attention.shape, dtype=torch.float8_e4m3fn, device=attention.device)
        sig = torch.empty_like(attention) if debug else None
        product = torch.empty_like(attention) if debug else None
        error = self.fn(attention.data_ptr(), gate.data_ptr(), scale.data_ptr(), output.data_ptr(),
                        sig.data_ptr() if debug else None, product.data_ptr() if debug else None,
                        *attention.shape, attention.stride(0), gate.stride(0),
                        torch.cuda.current_stream(attention.device).cuda_stream)
        if error: raise RuntimeError(self.lib.qwen_gate_error(error).decode())
        return (output, sig, product) if debug else output
