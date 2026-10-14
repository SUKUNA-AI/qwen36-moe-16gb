"""Pinned PyTorch/Triton reference and five alternating pairs; no checkpoint load."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess

# Exclude only the known WSL profiler CPU-helper hang; preserve GPU injection.
if os.environ.get('QWEN36_CUDA_PROFILE') == '1':
    original_popen = subprocess.Popen
    class ProfilePopen(original_popen):
        def __init__(self, args, *pos, **kw):
            command = args[0] if isinstance(args, (list, tuple)) else ''
            if Path(command).name == 'file':
                env = dict(kw.get('env') or os.environ); env.pop('LD_PRELOAD', None); kw['env'] = env
            super().__init__(args, *pos, **kw)
    subprocess.Popen = ProfilePopen

import torch
from freetoken.kernel.triton.fp8_pertensor_linear import _static_quant
from gate_fp8 import GateFp8

HERE = Path(__file__).resolve().parent

def baseline(attention, gate, scale):
    return _static_quant(attention * torch.sigmoid(gate), scale)

def byte_equal(a, b):
    av, bv = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    return dict(equal=bool(torch.equal(av, bv)), different_bytes=int((av != bv).sum()), bytes=av.numel())

def fixture(rows, scale_value, seed, strided=False, amplitude=1):
    torch.manual_seed(seed)
    width = 4112 if strided else 4096
    attention = torch.randn(rows, width, device='cuda', dtype=torch.bfloat16)[:, :4096] * amplitude
    gate = torch.randn(rows, width, device='cuda', dtype=torch.bfloat16)[:, :4096]
    if strided:
        a = torch.empty(rows, width, device='cuda', dtype=torch.bfloat16)
        a[:, :4096].copy_(attention); attention = a[:, :4096]
    scale = torch.tensor(scale_value, device='cuda', dtype=torch.float32)
    return attention, gate, scale

def numerical(candidate, attention, gate, scale, check_linear=False):
    actual, sig, product = candidate(attention, gate, scale, debug=True)
    expected_sig = torch.sigmoid(gate)
    expected_product = attention * expected_sig
    expected = _static_quant(expected_product, scale)
    row = dict(m=attention.shape[0], attention_stride=attention.stride(0), gate_stride=gate.stride(0),
               sigmoid=byte_equal(sig, expected_sig), product=byte_equal(product, expected_product),
               fp8=byte_equal(actual, expected))
    # Same FP8 GEMM on both paths; no independent quantization or weight changes.
    if check_linear:
        weight = (torch.randn(2048, 4096, device='cuda') * .1).to(torch.float8_e4m3fn)
        ws = torch.tensor(.25, device='cuda')
        def mm(a):
            return torch._scaled_mm(a, weight.t(), scale_a=scale, scale_b=ws, out_dtype=torch.bfloat16)
        row['linear_output'] = byte_equal(mm(actual), mm(expected))
    torch.cuda.synchronize()
    return row

def ms(fn, inner):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(inner): fn()
    end.record(); end.synchronize()
    return start.elapsed_time(end)/inner

def benchmark(candidate, rows):
    inputs = fixture(rows, .00731, 198804 + rows)
    fns = {'baseline':lambda:baseline(*inputs), 'cuda':lambda:candidate(*inputs)}
    for _ in range(5):
        for fn in fns.values(): fn()
    torch.cuda.synchronize()
    graphs = {}
    # Amortize Python graph-launch pacing for tiny decode kernels.
    graph_unroll = 32 if rows == 1 else 1
    for name, fn in fns.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(graph_unroll): out = fn()
        graphs[name] = (graph, out)
    records = {}
    for mode in ['eager','graph']:
        pairs=[]
        for repeat in range(5):
            pair={}
            for name in (['baseline','cuda'] if repeat%2==0 else ['cuda','baseline']):
                fn = fns[name] if mode=='eager' else graphs[name][0].replay
                pair[name+'_ms'] = ms(fn, 100 if rows < 65 else 20)/(graph_unroll if mode=='graph' else 1)
            pairs.append(pair)
        records[mode]=dict(pairs=pairs, median_speedup=statistics.median(p['baseline_ms']/p['cuda_ms'] for p in pairs))
    return dict(m=rows, metrics=records, graph_unroll=graph_unroll,
                scope='sigmoid + multiply + static quant only; no GEMM/offload')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--profile', choices=['decode','prefill'])
    p.add_argument('--output', default=str(HERE/'results.json')); p.add_argument('--numeric-only', action='store_true')
    args=p.parse_args(); candidate=GateFp8(); output=Path(args.output)
    report=dict(completed=False, device=torch.cuda.get_device_name(), torch=torch.__version__, numerical=[], benchmark=[])
    def save(): output.write_text(json.dumps(report,indent=2))
    if args.profile:
        inputs=fixture(1 if args.profile=='decode' else 2048, .00731, 189554)
        operations=[('baseline_gate_fp8',lambda:baseline(*inputs)),('cuda_fused_gate_fp8',lambda:candidate(*inputs))]
        for _,fn in operations: fn()
        torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStart()
        for name, fn in operations:
            with torch.cuda.nvtx.range(name): fn(); torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop(); return
    # Every finite BF16 gate, plus infinities, against ordinary PyTorch sigmoid.
    codes=torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(16,4096)
    codes=torch.where(torch.isnan(codes),torch.zeros_like(codes),codes)
    for value in [.00731, .003, .125, 1.13]:
        attention,_,scale=fixture(16,value,48912)
        row=numerical(candidate,attention,codes,scale)
        row.update(fixture='all_non_nan_bf16_gate_encodings',scale=value)
        report['numerical'].append(row); save()
    for rows,value,strided,amp in [(1,.00731,False,1),(33,.003,True,1),(2048,.00731,False,1),(4096,.125,True,100),(8192,1.13,False,1)]:
        inputs=fixture(rows,value,410281+rows,strided,amp)
        row=numerical(candidate,*inputs,check_linear=rows<64)
        row.update(fixture='random_activations',scale=value,amplitude=amp)
        report['numerical'].append(row); print(json.dumps(row),flush=True); save()
        del inputs; gc.collect(); torch.cuda.empty_cache()
    report['numeric_pass']=all(all(r[stage]['equal'] for stage in ['sigmoid','product','fp8']) and
                               r.get('linear_output',{'equal':True})['equal'] for r in report['numerical'])
    save()
    if not report['numeric_pass']: raise AssertionError('FP8 bytes differ: leave candidate disabled and inspect stages')
    if not args.numeric_only:
        for rows in [1,2048,4096,8192]:
            row=benchmark(candidate,rows); report['benchmark'].append(row); print(json.dumps(row),flush=True); save()
            gc.collect(); torch.cuda.empty_cache()
    report['completed']=True; save()

if __name__=='__main__': main()
