"""GPU micro-benchmark for the next candidate's decode levers, CUDA-graph replays, no profiler.

  d5:  40 layers of the MoE-block epilogue -- torch sigmoid/mul/add vs the fused kernel
       (moe/shared_epilogue.py); reports us per decode step saved.
  d3b: L layers of [memory-bound mixer stand-in of ~W us] + k expert misses of the released NVFP4 size:
         mix    -- mixer only (floor)
         serial -- mixer, then ensure + gather of the k misses on the critical path (today)
         spec   -- moe/pregate.py: the k misses are the pre-gate's prediction, copied on the side stream
                   behind the mixer; the real ensure then finds them resident
       hidden fraction = (serial - spec) / (serial - mix).

Usage (owner GPU, under flock gpu-exclusive.lock, nothing else on the GPU):
  python microbench.py OUT_JSON [--layers 8] [--reps 50]
"""
import argparse
import json
import os
import statistics
import time

import torch


def replay_us(graph, reps, inner):
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1e6 / inner)
    return statistics.median(times)


def capture(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def bench_d5(reps):
    from freetoken.moe import shared_epilogue as SE

    E = SE.SharedEpilogue()
    routed = [torch.randn((1, 2048), device='cuda').to(torch.bfloat16) for _ in range(40)]
    shared = [torch.randn((1, 2048), device='cuda').to(torch.bfloat16) for _ in range(40)]
    gate = [torch.randn((1, 1), device='cuda').to(torch.bfloat16) for _ in range(40)]
    assert E.usable(routed[0]), E.reason
    g_ref = capture(lambda: [SE.reference(r, s, g) for r, s, g in zip(routed, shared, gate)])
    g_fused = capture(lambda: [E(r, s, g) for r, s, g in zip(routed, shared, gate)])
    ref, fused = replay_us(g_ref, reps, 1), replay_us(g_fused, reps, 1)
    return {'torch_us_per_step': ref, 'fused_us_per_step': fused, 'saved_us_per_step': ref - fused}


def bench_d3b(layers, reps, k, mixer_rows):
    from freetoken.moe.offload_cache import OffloadMoeCache, _BANK_SCHEMAS

    E, H = 32, 2048
    sizes = (1048576, 131072, 4096, 524288, 65536, 8192)
    sources = {name: [torch.randint(0, 256, (E, size), dtype=torch.uint8).pin_memory() for _ in range(layers)]
               for name, size in zip(_BANK_SCHEMAS['nvfp4'], sizes)}
    os.environ['QWEN36_DECODE_PREGATE'] = str(k)
    os.environ['QWEN36_DECODE_PREGATE_ATTN'] = str(k)
    cache = OffloadMoeCache(layers, E, layers * E, torch.device('cuda'), quant_format='nvfp4')
    cache.set_bank_sources(sources)
    pregate = cache.decode_pregate
    residual = torch.randn((1, H), device='cuda').to(torch.bfloat16)
    norm = type('N', (), {})(); norm.weight = torch.ones(H, device='cuda', dtype=torch.bfloat16); norm.eps = 1e-6
    gates = []
    for _ in range(layers):
        g = type('G', (), {})(); g.weight = torch.randn((E, H), device='cuda').to(torch.bfloat16); gates.append(g)
    x = residual.float()
    y = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16).float()
    ids = [(y @ gates[lid].weight.float().t())[0].topk(k).indices.to(torch.int32) for lid in range(layers)]
    mixer_w = torch.randn((mixer_rows, H), device='cuda').to(torch.bfloat16)
    mixer_x = torch.randn((H,), device='cuda').to(torch.bfloat16)

    def mix():
        for _ in range(layers):
            torch.mv(mixer_w, mixer_x)

    def serial():
        for lid in range(layers):
            torch.mv(mixer_w, mixer_x)
            cache.ensure_experts(lid, ids[lid].clone())
            cache.copy_missing()

    def spec():
        for lid in range(layers):
            pregate.launch(lid, residual, norm, gates[lid], k, 8)
            torch.mv(mixer_w, mixer_x)
            pregate.release_input()
            cache.ensure_experts(lid, ids[lid].clone())
            cache.copy_missing()

    out = {}
    for name, fn in (('mix', mix), ('serial', serial), ('spec', spec)):
        cache.reset()
        graph = capture(fn)
        times = []
        for _ in range(reps):
            cache.reset()  # every replay starts cold: each layer has exactly k misses
            torch.cuda.synchronize()
            t = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1e6 / layers)
        out[name + '_us_per_layer'] = statistics.median(times)
        del graph
    serial_extra = out['serial_us_per_layer'] - out['mix_us_per_layer']
    out['hidden_fraction'] = ((out['serial_us_per_layer'] - out['spec_us_per_layer']) / serial_extra
                              if serial_extra > 0 else None)
    out['k'] = k
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--reps', type=int, default=50)
    a = ap.parse_args()
    torch.manual_seed(0)
    result = {'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'd5': bench_d5(a.reps), 'd3b': []}
    print(json.dumps(result['d5']))
    for mixer_rows in (6400, 12800, 25600):  # ~16 / 33 / 66 MB of bf16: ~25 / 45 / 85 us at ~750 GB/s
        for k in (1, 2):
            row = dict(bench_d3b(a.layers, a.reps, k, mixer_rows), mixer_rows=mixer_rows)
            print(json.dumps(row))
            result['d3b'].append(row)
    with open(a.out, 'w') as f:
        json.dump(result, f, indent=1)


if __name__ == '__main__':
    main()
