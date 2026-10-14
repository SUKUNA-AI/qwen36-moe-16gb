"""Standalone opt-in FP32 GDN snapshot qualification; run only in a reserved GPU slot.

Uses the same vendored recurrence kernel as Qwen. The numerical reference is a
separate invocation ending exactly at each selected boundary, before donation.
"""
import argparse
import json
import pathlib
import statistics
import os

if os.environ.get('QWEN36_PROFILE_HELPERS')=='1':
    # Same scoped workaround as experiment.profile_helpers: Nsight injection's
    # worker in the non-GPU /usr/bin/file helper otherwise keeps its pipe open.
    import subprocess
    original_popen=subprocess.Popen
    class ProfiledPopen(original_popen):
        def __init__(self,args,*pos,**kw):
            command=args[0] if isinstance(args,(list,tuple)) else ''
            if pathlib.Path(command).name=='file':
                env=dict(kw.get('env') or os.environ);env.pop('LD_PRELOAD',None);kw['env']=env
            super().__init__(args,*pos,**kw)
    subprocess.Popen=ProfiledPopen

import torch

from freetoken.kernel.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h


def inputs(lengths,h=32,hg=16,k=128,v=128):
    gen=torch.Generator(device='cuda').manual_seed(31013)
    total=sum(lengths)
    def rand(shape):return torch.randn(shape,generator=gen,device='cuda',dtype=torch.bfloat16)*.03
    keys=rand((1,total,hg,k));values=rand((1,total,h,v));weights=rand((1,total,h,k))
    # Zero log-decay isolates state storage and keeps the prefix arithmetic identical.
    gates=torch.zeros((1,total,h),device='cuda',dtype=torch.float32)
    initial=torch.randn((2*len(lengths)+2,h,v,k),generator=gen,device='cuda')*.1
    live=torch.arange(1,len(lengths)+1,device='cuda',dtype=torch.int32)
    cumulative=[0]
    for n in lengths:cumulative.append(cumulative[-1]+n)
    cu=torch.tensor(cumulative,device='cuda',dtype=torch.int64)
    return keys,values,weights,gates,initial,live,cu


def launch(data,state,rows=None,slots=None):
    k,u,w,g,_initial,live,cu=data
    return chunk_gated_delta_rule_fwd_h(k,w,u,g=g,initial_state=state,
        initial_state_indices=live,cu_seqlens=cu,
        snapshot_h_rows=rows,snapshot_state_indices=slots)


def run_case(lengths,boundaries,h=32,hg=16,k=128,v=128,graph=False):
    data=inputs(lengths,h,hg,k,v);initial=data[4]
    offsets=[];offset=0
    for n in lengths:offsets.append(offset);offset+=(n+63)//64
    rows=torch.tensor([off+b//64 for off,b in zip(offsets,boundaries)],device='cuda',dtype=torch.int64)
    slots=torch.arange(len(lengths)+1,2*len(lengths)+1,device='cuda',dtype=torch.int64)
    base=initial.clone();old_h,old_v=launch(data,base)
    actual=initial.clone();new_h,new_v=launch(data,actual,rows,slots)
    torch.cuda.synchronize()
    assert old_h.dtype==new_h.dtype==torch.bfloat16
    assert torch.equal(new_h,old_h),'ordinary BF16 h changed'
    assert torch.equal(new_v,old_v),'ordinary recurrence output changed'
    assert torch.equal(actual[data[5]],base[data[5]]),'live FP32 state changed'
    assert torch.equal(actual[0],initial[0]) and torch.equal(actual[-1],initial[-1])
    measurements=[];start=0
    for seq,(n,boundary,dst) in enumerate(zip(lengths,boundaries,slots.tolist())):
        reference=initial.clone()
        if boundary:
            prefix=tuple(t[:,start:start+boundary].contiguous() for t in data[:4])
            prefix_data=(*prefix,initial,data[5][seq:seq+1],
                         torch.tensor([0,boundary],device='cuda',dtype=torch.int64))
            launch(prefix_data,reference)
        target=reference[seq+1]
        torch.cuda.synchronize()
        assert torch.equal(actual[dst],target),'direct snapshot differs from same-boundary FP32 recurrence'
        rounded=old_h[0,rows[seq]].float()
        assert torch.equal(actual[dst].to(torch.bfloat16),old_h[0,rows[seq]])
        measurements.append(dict(boundary=boundary,direct_bit_equal=True,
            legacy_relative_l2=float((rounded-target).norm()/target.norm()),
            legacy_max_abs=float((rounded-target).abs().max())))
        start+=n
    if graph:
        graph_obj=torch.cuda.CUDAGraph()
        capture_state=initial.clone()
        # Compile/populate metadata caches before capture.
        launch(data,capture_state,rows,slots)
        with torch.cuda.graph(graph_obj):
            capture_state.copy_(initial)
            graph_h,graph_v=launch(data,capture_state,rows,slots)
        for _ in range(3):graph_obj.replay()
        torch.cuda.synchronize()
        assert torch.equal(capture_state,actual)
        assert torch.equal(graph_h,new_h) and torch.equal(graph_v,new_v)
    return dict(lengths=lengths,boundaries=boundaries,h=h,hg=hg,k=k,v=v,
                ordinary_outputs_bit_equal=True,live_state_bit_equal=True,
                graph_replays=3 if graph else 0,measurements=measurements)


def benchmark(chunk,repeats=5,iterations=20):
    data=inputs([chunk]);initial=data[4]
    rows=torch.tensor([(chunk-8)//64],device='cuda',dtype=torch.int64)
    slots=torch.tensor([2],device='cuda',dtype=torch.int64)
    states=[initial.clone(),initial.clone()]
    def run(mode):
        state=states[mode];state.copy_(initial)
        if mode:
            return launch(data,state,rows,slots)
        h,v=launch(data,state)
        state.index_copy_(0,slots,h[0,rows].to(state.dtype))
        return h,v
    for _ in range(3):run(0);run(1)
    pairs=[]
    for repeat in range(repeats):
        times={}
        for mode in ((0,1) if repeat%2==0 else (1,0)):
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(iterations):run(mode)
            end.record();end.synchronize()
            times['direct_fp32_us' if mode else 'legacy_bf16_us']=begin.elapsed_time(end)*1000/iterations
        pairs.append(times)
    return dict(chunk=chunk,repeats=repeats,iterations=iterations,pairs=pairs,
                median_ratio=statistics.median(p['legacy_bf16_us']/p['direct_fp32_us'] for p in pairs),
                scope='recurrence plus snapshot, includes common pool reset; not model speed')


def wrapper_case():
    """Exercise the actual model wrapper through the vendored autograd API."""
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla
    keys,values,_weights,gates,initial,live,cu=inputs([137])
    beta=torch.full((1,137,32),.3,device='cuda',dtype=torch.float32)
    rows=torch.tensor([1],device='cuda',dtype=torch.int64)
    slots=torch.tensor([2],device='cuda',dtype=torch.int64)
    def run(state,n=137,direct=False):
        return gdn_prefill_chunk_fla(keys[:,:n],keys[:,:n],values[:,:n],gates[:,:n],beta[:,:n],
            state_source=state,indices=live,
            cu_seqlens=cu if n==137 else torch.tensor([0,n],device='cuda',dtype=torch.int64),
            scale=128**-.5,return_h=True,
            snapshot_h_rows=rows if direct else None,snapshot_state_indices=slots if direct else None)
    baseline=initial.clone();base_out,base_h=run(baseline)
    actual=initial.clone();out,h=run(actual,direct=True)
    reference=initial.clone();run(reference,n=64)
    torch.cuda.synchronize()
    assert torch.equal(out,base_out) and torch.equal(h,base_h)
    assert torch.equal(actual[live],baseline[live])
    assert torch.equal(actual[2],reference[1])
    return dict(wrapper_output_bit_equal=True,live_state_bit_equal=True,direct_snapshot_bit_equal=True)


def profile(mode,chunk=4096):
    data=inputs([chunk]);state=data[4].clone()
    rows=torch.tensor([(chunk-8)//64],device='cuda',dtype=torch.int64)
    slots=torch.tensor([2],device='cuda',dtype=torch.int64)
    for _ in range(3):
        state.copy_(data[4]);launch(data,state,rows if mode=='fp32' else None,slots if mode=='fp32' else None)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    with torch.cuda.nvtx.range('gdn_snapshot_'+mode):
        state.copy_(data[4])
        h,_=launch(data,state,rows if mode=='fp32' else None,slots if mode=='fp32' else None)
        if mode=='bf16':state.index_copy_(0,slots,h[0,rows].to(state.dtype))
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--profile',choices=('bf16','fp32'))
    parser.add_argument('--wrapper-only',action='store_true')
    parser.add_argument('--output',type=pathlib.Path)
    args=parser.parse_args()
    if args.wrapper_only:
        report=wrapper_case()
        if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    elif args.profile:profile(args.profile)
    elif args.run:
        report=dict(device=torch.cuda.get_device_name(),cases=[
            run_case([137],[64],graph=True),
            run_case([193],[128]),run_case([137,204],[64,128]),
            run_case([193],[0],h=4,hg=2,k=64,v=80),
            run_case([137],[64],h=4,hg=2,k=256,v=128)],
            wrapper=wrapper_case(),benchmarks=[benchmark(chunk) for chunk in (2048,4096,8192)])
        if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    else:parser.error('Use --run in a reserved GPU slot, or --profile MODE')
