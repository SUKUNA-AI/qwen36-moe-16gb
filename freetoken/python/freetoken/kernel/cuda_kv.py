"""Optional SM120 fused Q/K RMSNorm + RoPE + cache store. No model monkeypatch."""
import ctypes as ct
import pathlib
import torch


class Params(ct.Structure):
    _fields_ = [(n, ct.c_void_p) for n in ('q','k','v','gate','qw','kw','rope','positions','locations','qo','go','kc','vc','errors')] + [
        (n, ct.c_int64) for n in ('tokens','qs','qh','ks','kh','vs','vh','gs','gh','capacity','rope_rows')] + [
        (n, ct.c_float) for n in ('epsilon','kscale','vscale')] + [
        (n, ct.c_int32) for n in ('fp8','pos64','loc64','reserved')]


def heads(t, n):
    if t.ndim == 2 and t.shape[1] == n * 256:
        t = t.view(t.shape[0], n, 256)
    if t.ndim != 3 or tuple(t.shape[1:]) != (n,256) or t.stride(-1) != 1:
        raise ValueError(f'Expected [N,{n},256] or [N,{n*256}], contiguous head dimension')
    return t


class FusedKV:
    def __init__(self, library=None):
        self.library = ct.CDLL(str(library or pathlib.Path(__file__).parent / 'build/libqwen36_kv.so'))
        self.library.qwen36_kv_abi_size.restype = ct.c_int
        assert self.library.qwen36_kv_abi_size() == ct.sizeof(Params)
        self.library.qwen36_kv_launch.argtypes = [ct.POINTER(Params), ct.c_void_p, ct.c_int]
        self.library.qwen36_kv_launch.restype = ct.c_int
        self.library.qwen36_kv_offset.argtypes = [ct.c_int64,ct.c_int,ct.c_int]
        self.library.qwen36_kv_offset.restype = ct.c_int64

    def __call__(self, q, k, v, qw, kw, rope, positions, locations, kc, vc,
                 *, gate=None, qo=None, go=None, errors=None, epsilon=1e-6,
                 scales=(1.,1.), warps=4):
        q,k,v = heads(q,16),heads(k,2),heads(v,2)
        gate = heads(gate,16) if gate is not None else None
        n=q.shape[0]
        if k.shape[0] != n or v.shape[0] != n or (gate is not None and gate.shape[0] != n):
            raise ValueError('Mismatched token counts')
        if kc.shape != vc.shape or kc.ndim != 4 or tuple(kc.shape[2:]) != (2,256) or not kc.is_contiguous() or not vc.is_contiguous():
            raise ValueError('Cache requires contiguous [pages,page_size,2,256]')
        if kc.dtype != vc.dtype or kc.dtype not in (torch.bfloat16,torch.float8_e4m3fn):
            raise ValueError('Cache dtype must be BF16 or E4M3FN')
        for t in (q,k,v,qw,kw) + ((gate,) if gate is not None else ()):
            if t.dtype != torch.bfloat16 or t.device != q.device or not t.is_cuda:
                raise ValueError('Inputs must be BF16 on one CUDA device')
        if qw.shape != (256,) or kw.shape != (256,) or not qw.is_contiguous() or not kw.is_contiguous():
            raise ValueError('Norm weights require 256 contiguous folded values')
        if rope.ndim != 2 or rope.shape[1] != 64 or rope.dtype != torch.float32 or not rope.is_contiguous():
            raise ValueError('RoPE requires FP32 [max_position,64] NeoX cos/sin cache')
        for t in (positions,locations):
            if t.shape != (n,) or t.dtype not in (torch.int32,torch.int64) or not t.is_contiguous():
                raise ValueError('Positions and locations require contiguous int32/int64 [N]')
        qo = torch.empty((n,16,256),device=q.device,dtype=q.dtype) if qo is None else qo
        go = torch.empty_like(qo) if go is None and gate is not None else go
        for t in (qo,) + ((go,) if go is not None else ()):
            if t.shape != (n,16,256) or t.dtype != q.dtype or not t.is_contiguous():
                raise ValueError('Outputs require contiguous [N,16,256] BF16')
        for t in (rope,positions,locations,kc,vc,qo) + ((go,) if go is not None else ()) + ((errors,) if errors is not None else ()):
            if t.device != q.device: raise ValueError('All tensors require same CUDA device')
        if errors is not None and (errors.shape != (2,) or errors.dtype != torch.int64 or not errors.is_contiguous()):
            raise ValueError('Errors requires int64 [2] for invalid index and saturation counts')
        p=Params()
        for name,t in dict(q=q,k=k,v=v,gate=gate,qw=qw,kw=kw,rope=rope,positions=positions,
                          locations=locations,qo=qo,go=go,kc=kc,vc=vc,errors=errors).items():
            setattr(p,name,t.data_ptr() if t is not None else 0)
        p.tokens=n
        for name,t in [('q',q),('k',k),('v',v),('g',gate)]:
            if t is not None:
                setattr(p,name+'s',t.stride(0));setattr(p,name+'h',t.stride(1))
        p.capacity=kc.shape[0]*kc.shape[1];p.rope_rows=rope.shape[0]
        # ABI stores reciprocals, matching the existing Triton store's rounding.
        if not all(s>0 for s in scales): raise ValueError('KV scales must be positive')
        p.epsilon=epsilon;p.kscale=1./scales[0];p.vscale=1./scales[1]
        p.fp8=int(kc.dtype==torch.float8_e4m3fn)
        p.pos64=int(positions.dtype==torch.int64);p.loc64=int(locations.dtype==torch.int64)
        rc=self.library.qwen36_kv_launch(ct.byref(p),torch.cuda.current_stream(q.device).cuda_stream,warps)
        if rc: raise RuntimeError(f'CUDA fused KV launch error {rc}')
        return qo,go

    def projection(self, qkv, qw, kw, rope, positions, locations, kc,vc, **kwds):
        if qkv.ndim != 2 or qkv.shape[1] != 9216 or qkv.stride(-1) != 1:
            raise ValueError('Qwen3.6 projection requires [N,9216]')
        qg=qkv[:,:8192].view(-1,16,512)
        return self(qg[:,:,:256],qkv[:,8192:8704],qkv[:,8704:],qw,kw,rope,positions,locations,kc,vc,
                    gate=qg[:,:,256:],**kwds)
