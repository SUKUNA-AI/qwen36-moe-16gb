"""Real SM120 scatter, mixed Q/KV dtypes, prefill and decode parity fixture."""
import json, pathlib
import torch
import flashinfer
from freetoken.kernel.triton.fp8_kv import store_fp8

torch.manual_seed(91)
device='cuda'
n=257; h=2;d=256
raw=torch.randn(n,h,d*2,device=device,dtype=torch.bfloat16)
k=raw[:,:,:d];v=raw[:,:,d:]
loc=torch.randperm(n,device=device,dtype=torch.int32)
kc=torch.zeros(n,1,h,d,device=device,dtype=torch.float8_e4m3fn);vc=torch.zeros_like(kc)
ks,vs=.015,.02
overflow=torch.zeros((),device=device,dtype=torch.int64)
store_fp8(k,v,loc,kc,vc,(ks,vs),overflow)
expected_k=(k.float()/ks).clamp(-448,448).to(torch.float8_e4m3fn)
expected_v=(v.float()/vs).clamp(-448,448).to(torch.float8_e4m3fn)
for name, actual, expected, values, scale in [('k',kc[:,0][loc.long()],expected_k,k,ks),('v',vc[:,0][loc.long()],expected_v,v,vs)]:
    mismatch=actual.view(torch.uint8)!=expected.contiguous().view(torch.uint8)
    if mismatch.any():
        print(json.dumps({'tensor':name,'mismatches':int(mismatch.sum()),'input':values[mismatch][:10].float().tolist(),
                          'actual':actual.float()[mismatch][:10].tolist(),'expected':expected.float()[mismatch][:10].tolist()}))
    assert not mismatch.any()
assert overflow.item()==0
# Runtime QKV projection supplies flattened, strided [tokens, kv_heads*dim].
# This is distinct from the convenient 3-D fixture and previously exposed a bug.
k2=torch.randn(n,h*d*2,device=device,dtype=torch.bfloat16)
k2v=k2[:,h*d:];k2k=k2[:,:h*d]
store_fp8(k2k,k2v,loc,kc,vc,(ks,vs),overflow)
for actual,source,scale in [(kc,k2k,ks),(vc,k2v,vs)]:
    expected=(source.reshape(n,h,d).float()/scale).clamp(-448,448).to(torch.float8_e4m3fn)
    assert torch.equal(actual[:,0][loc.long()].view(torch.uint8),expected.contiguous().view(torch.uint8))
# Restore the original data for the attention accuracy fixture below.
store_fp8(k,v,loc,kc,vc,(ks,vs),overflow)
reference_k=torch.empty(n,1,h,d,device=device,dtype=torch.bfloat16)
reference_v=torch.empty_like(reference_k)
reference_k[loc.long(),0]=k;reference_v[loc.long(),0]=v
indptr=torch.tensor([0,n],dtype=torch.int32,device=device)
last=torch.tensor([1],dtype=torch.int32,device=device)
workspace=torch.empty(128*1024*1024,dtype=torch.uint8,device=device)
results=[]
for qn in (1,17):
    q=torch.randn(qn,16,d,device=device,dtype=torch.bfloat16)
    outs=[]
    for dtype,caches,scales in [(torch.bfloat16,(reference_k,reference_v),{}),
                              (torch.float8_e4m3fn,(kc,vc),{'k_scale':ks,'v_scale':vs})]:
        if qn==1:
            wrapper=flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace,'NHD')
            wrapper.plan(indptr,loc,last,16,h,d,1,q_data_type=torch.bfloat16,kv_data_type=dtype)
        else:
            wrapper=flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace,'NHD')
            wrapper.plan(torch.tensor([0,qn],device=device,dtype=torch.int32),indptr,loc,last,
                         16,h,d,1,causal=True,q_data_type=torch.bfloat16,kv_data_type=dtype)
        outs.append(wrapper.run(q,caches,**scales))
    relative=(outs[0].float()-outs[1].float()).norm()/outs[0].float().norm()
    assert torch.isfinite(outs[1]).all() and relative.item()<.06
    results.append({'q_tokens':qn,'relative_l2':relative.item(),'max_abs':(outs[0]-outs[1]).abs().max().item()})
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph): store_fp8(k,v,loc,kc,vc,(ks,vs),overflow)
graph.replay();torch.cuda.synchronize()
doc={'result':'PASS','scatter_byte_equal':True,'strided_input':True,'flattened_2d_input':True,'cuda_graph':True,'attention':results}
path=pathlib.Path('/home/USER/qwen36-api/optimization/reports/kv-kernel.json')
path.write_text(json.dumps(doc,indent=2));print(json.dumps(doc))
