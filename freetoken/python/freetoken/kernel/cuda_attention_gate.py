"""Opt-in full-attention output gate and static FP8 cast; BF16 fallback retained."""
import ctypes
import math
import os
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
        if any(t.stride(0) % 8 or t.data_ptr() % 16 for t in (attention, gate)):
            raise ValueError('BF16 row stride and storage offset must permit aligned vector loads')
        output = torch.empty(attention.shape, dtype=torch.float8_e4m3fn, device=attention.device)
        sig = torch.empty_like(attention) if debug else None
        product = torch.empty_like(attention) if debug else None
        error = self.fn(attention.data_ptr(), gate.data_ptr(), scale.data_ptr(), output.data_ptr(),
                        sig.data_ptr() if debug else None, product.data_ptr() if debug else None,
                        *attention.shape, attention.stride(0), gate.stride(0),
                        torch.cuda.current_stream(attention.device).cuda_stream)
        if error: raise RuntimeError(self.lib.qwen_gate_error(error).decode())
        return (output, sig, product) if debug else output


def fused_attention_output(layer, attention, gate):
    """Return None for unsupported schemes so the original _combine stays authoritative."""
    from freetoken.layers.quantization.linear.fp8_tensor import TorchFp8TensorLinearKernel

    projection = layer.o_proj
    kernel = getattr(getattr(projection, 'quant_method', None), 'kernel', None)
    if type(kernel) is not TorchFp8TensorLinearKernel or getattr(projection, '_fp8_uniform_scale', False) is not True:
        return None
    if getattr(projection, 'bias', None) is not None or layer.qo_attn_dim != 4096:
        return None
    scale = getattr(projection, 'input_scale', None)
    weight = getattr(projection, 'weight', None)
    ws = getattr(projection, 'weight_scale', None)
    if not all(isinstance(t, torch.Tensor) for t in (attention, gate, scale, weight, ws)):
        return None
    if attention.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
        return None
    if not all(t.is_cuda and t.device == attention.device for t in (attention, gate, scale, weight, ws)):
        return None
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2 or weight.shape[1] != 4096 or not weight.is_contiguous():
        return None
    if scale.dtype != torch.float32 or scale.numel() != 1 or not scale.is_contiguous():
        return None
    if ws.dtype != torch.float32 or ws.shape != (weight.shape[0],) or not ws.is_contiguous():
        return None
    try:
        flat = attention.view(-1, 4096)
    except RuntimeError:
        return None
    if gate.shape != flat.shape or not flat.shape[0] or flat.shape[0] > (2**31-1)//4096:
        return None
    if any(t.stride(1) != 1 or t.stride(0) < 4096 or t.stride(0) % 8 or t.data_ptr() % 16 for t in (flat, gate)):
        return None
    try:
        version = scale._version
    except RuntimeError:
        # Checkpoint tensors created under inference_mode have no version counter.
        version = None
    identity = (scale.data_ptr(), version)
    checked = getattr(layer, '_fused_gate_scale_checked', None)
    if checked is None or checked[:2] != identity:
        # The deployment warms each layer before graph capture; never synchronize
        # scale validation from within a CUDA graph or silently change its scheme.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Warm QWEN36_FUSED_ATTN_GATE once before CUDA graph capture')
        value = scale.item()
        supported = math.isfinite(value) and value > 0 and torch.cuda.get_device_capability(flat.device) == (12, 0)
        checked = (*identity, supported)
        layer._fused_gate_scale_checked = checked
    if not checked[2]:
        return None
    if not hasattr(layer, '_fused_attention_gate'):
        library = os.environ.get('QWEN36_FUSED_ATTN_GATE_LIBRARY')
        if not library:
            raise ValueError('QWEN36_FUSED_ATTN_GATE_LIBRARY must name the pinned separate library')
        layer._fused_attention_gate = GateFp8(library)
    quantized = layer._fused_attention_gate(flat, gate, scale)
    # Match the uniform-scale branch of _scaled_mm, without a second quantize.
    from freetoken.kernel.triton.fp8_pertensor_linear import uniform_scaled_mm

    return uniform_scaled_mm(quantized, weight, scale, ws, oproj=getattr(projection, '_oproj_gemv', False))
