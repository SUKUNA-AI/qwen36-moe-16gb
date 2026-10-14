"""Small C ABI for the CUDA C++ lossless expert transport prototype."""
import ctypes,os,pathlib
import torch

_library=None

def library():
    global _library
    if _library is None:
        path=pathlib.Path(os.environ['QWEN36_COMPLEMENT_LIBRARY'])
        _library=ctypes.CDLL(str(path))
        pointer=ctypes.c_void_p;integer=ctypes.c_int;stream=ctypes.c_size_t
        _library.qwen36_exchange.argtypes=[pointer]*7+[integer]*5+[stream]
        _library.qwen36_gather.argtypes=[pointer]*5+[integer]*5+[stream]
        _library.qwen36_exchange.restype=_library.qwen36_gather.restype=integer
    return _library


def exchange(cache,layer):
    tensors=(cache._c_host,cache._c_gpu,cache._c_feat,cache.host_slot,
             cache.src_indices,cache.evict_slots,cache.num_indices)
    code=library().qwen36_exchange(*[t.data_ptr() for t in tensors],layer*cache.num_experts,
        cache.offset if cache.adaptive else 0,len(cache.bank_schema),cache._plan_capacity,
        int(cache.adaptive),torch.cuda.current_stream().cuda_stream)
    if code:raise RuntimeError(f'CUDA complement transport launch failed: {code}')


def gather(cache,layer,destination):
    tensors=(cache._c_host,cache._c_gpu,cache._c_feat,cache.host_slot,cache.slot_for_id)
    code=library().qwen36_gather(*[t.data_ptr() for t in tensors],layer*cache.num_experts,
        cache.offset,destination,len(cache.bank_schema),cache.num_experts,
        torch.cuda.current_stream().cuda_stream)
    if code:raise RuntimeError(f'CUDA complement gather launch failed: {code}')
