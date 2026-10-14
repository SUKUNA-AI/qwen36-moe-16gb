"""Opt-in BF16-only integration for the pinned Qwen3.5 full-attention class."""
import os
import torch
from freetoken.core import get_global_ctx
from freetoken.attention.fi import FlashInferBackend, FIMetadata
from freetoken.kernel.cuda_kv import FusedKV


def fused_full_attention(layer, x):
    ctx=get_global_ctx()
    backend=ctx.attn_backend
    if not isinstance(backend,FlashInferBackend):
        raise ValueError('QWEN36_FUSED_QK_KV requires the pinned FlashInfer backend')
    if (layer.num_q,layer.num_kv,layer.head_dim)!=(16,2,256):
        raise ValueError('Fused Qwen36 full attention requires Q16/KV2/D256')
    pool=backend.kvcache
    if pool.dtype!=torch.bfloat16:
        raise ValueError('The model integration is qualified only for BF16 KV; FP8 remains component-only')
    if getattr(pool,'calibration_max',None) is not None:
        raise ValueError('Use original QK/store path for FP8 scale calibration')
    positions=ctx.batch.get_attn_positions()
    if positions.ndim!=1 or layer.rotary.rotary_dim!=64 or not layer.rotary.is_neox:
        raise ValueError('Only text-only 1D NeoX partial RoPE64 is supported')
    metadata=ctx.batch.attn_metadata
    if not isinstance(metadata,FIMetadata): raise ValueError('Expected FIMetadata')
    backend._initialize_metadata_once(metadata)
    if not hasattr(layer,'_fused_qk_kv'):
        library=os.environ.get('QWEN36_FUSED_KV_LIBRARY')
        if not library: raise ValueError('QWEN36_FUSED_KV_LIBRARY must name the pinned built library')
        layer._fused_qk_kv=FusedKV(library)
        layer._fused_qk_errors=torch.zeros((2,),device=x.device,dtype=torch.int64)
    projection=layer.qkv_proj.forward(x)
    kc,vc=pool.k_cache(layer.layer_id),pool.v_cache(layer.layer_id)
    q,gate=layer._fused_qk_kv.projection(projection,layer.q_norm.weight,layer.k_norm.weight,
        layer.rotary._cos_sin_cache,positions,ctx.batch.out_loc,kc,vc,
        epsilon=layer.q_norm.eps,errors=layer._fused_qk_errors,warps=4)
    # The fused kernel already wrote these pages; backend.forward would store twice.
    caches=(kc.view(-1,1,kc.shape[2],kc.shape[3]),vc.view(-1,1,vc.shape[2],vc.shape[3]))
    result=metadata.wrapper.run(q=q,paged_kv_cache=caches)
    return layer._combine(result,gate.reshape(-1,4096))
