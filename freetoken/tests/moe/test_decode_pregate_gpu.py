"""SM120 owner gate for D3b (moe/pregate.py): real cache/LRU/gather, speculative side-stream plans.

Per step and layer the pregate arm forks the speculative plan, runs a stand-in mixer on the compute
stream, then the real routing goes through ensure/copy (serial or T3) into a byte consumer. Checked
against a reference arm without pregate, eager and twice-captured graph replay with changing routes:
  * routed bytes equal the reference arm and the host banks (the arithmetic input is unchanged);
  * every mapped slot of the pregate cache holds its expert's host bytes after every step;
  * the speculative plans really admit experts and remove real misses.
The LRU state is expected to differ from the reference (see moe/pregate.py); the model teacher gate
is additionally required. Run without a profiler under flock.
"""
import pytest
import torch
from types import SimpleNamespace as S

from freetoken.layers.moe import OffloadMoELayer
from freetoken.moe.offload_cache import OffloadMoeCache, _BANK_SCHEMAS
from flashlib.kernels.slot_cache import Stat

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='owner CUDA/SM120 required')

E, H, TOP_K = 32, 256, 8
SIZES = (1048576, 131072, 4096, 524288, 65536, 8192)  # released native NVFP4 rows


def build(layers, sizes=SIZES):
    torch.manual_seed(1)
    return {name: [torch.randint(0, 256, (E, size), dtype=torch.uint8).pin_memory() for _ in range(layers)]
            for name, size in zip(_BANK_SCHEMAS['nvfp4'], sizes)}


def make_models(cache, layers):
    parts = []
    for lid in range(layers):
        model = object.__new__(OffloadMoELayer)
        model.offload_cache = cache; model.layer_id = lid; model.top_k = TOP_K; model.renormalize = True
        model.tp_size = 1; model.num_experts = E

        def consume(cache, hidden, weights, ids, *, views, **kw):
            return tuple(bank.index_select(0, ids.flatten().long()).clone() for bank in views)
        model._expert_gemm = consume
        parts.append(model)
    return parts


@pytest.mark.parametrize('overlap', ['', '1'])
@pytest.mark.parametrize('gather', ['', 'tuned'])
@pytest.mark.parametrize('caps', [('1', '2'), ('3', '3')])
def test_pregate_bytes_residency_and_capture_replay(monkeypatch, overlap, gather, caps):
    layers = 4
    monkeypatch.setenv('QWEN36_DECODE_SHARED_OVERLAP', overlap)
    monkeypatch.setenv('QWEN36_DECODE_GATHER', gather)
    device = torch.device('cuda')
    sources = build(layers)
    caches, models = [], []
    for arm in (0, 1):
        monkeypatch.setenv('QWEN36_DECODE_PREGATE', caps[0] if arm else '')
        monkeypatch.setenv('QWEN36_DECODE_PREGATE_ATTN', caps[1] if arm else '')
        cache = OffloadMoeCache(layers, E, E + 8, device, quant_format='nvfp4')
        cache.set_bank_sources(sources); assert cache._copy_fused_ok
        cache.collect_stats = True
        caches.append(cache); models.append(make_models(cache, layers))
    pregate = caches[1].decode_pregate
    assert caches[0].decode_pregate is None and pregate is not None
    overlap_stream = getattr(caches[1].decode_shared_overlap, 'stream', None)
    assert (pregate.stream is overlap_stream) == bool(overlap)
    truth = [torch.cat([per_layer[lid].to(device) for lid in range(layers)]) for per_layer in sources.values()]

    residual = torch.randn((1, H), device=device).to(torch.bfloat16)
    norm = S(weight=(torch.rand(H, device=device) + 0.5).to(torch.bfloat16), eps=1e-6)
    gates = [S(weight=torch.randn((E, H), device=device).to(torch.bfloat16)) for _ in range(layers)]
    noise = torch.zeros((layers, E), device=device)
    logits = torch.zeros((1, E), device=device)
    hidden = torch.randn((1, 64), device=device); weight = torch.randn((64, 64), device=device)
    mixer_w = torch.randn((2048, 2048), device=device)

    def shared(x):
        return torch.sigmoid(x @ weight) * (x @ weight)

    def execute(arm):
        out = []
        for lid, model in enumerate(models[arm]):
            launched = arm == 1 and lid >= 1
            if launched:
                cap = pregate.cap(attention=lid % 2 == 1)
                assert pregate.predictor_ok(residual, norm, gates[lid], E)
                pregate.launch(lid, residual, norm, gates[lid], cap, TOP_K)
            torch.mm(mixer_w, mixer_w)  # stand-in mixer on the compute stream
            if launched:
                pregate.release_input()
            x = residual.float()
            real = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight.float()) @ gates[lid].weight.float().t()
            logits.copy_(real + noise[lid])
            if overlap:
                routed, s = model.forward_with_shared(hidden, logits, shared)
            else:
                s = shared(hidden); routed = model.decode_forward(hidden, logits)
            out.extend(routed); out.append(s)
        return tuple(out)

    def check_residency(cache):
        ids = cache.id_of_slot.long()
        live = (ids >= 0).nonzero().flatten()
        for (_, bank), t in zip(cache.banks, truth):
            assert torch.equal(bank[live], t[ids[live]]), 'a mapped slot lost its expert bytes'
        flat = cache.slot_for_id.view(-1)
        mapped = (flat >= 0).nonzero().flatten()
        assert torch.equal(ids[flat[mapped].long()], mapped), 'slot maps are not mutually inverse'

    def compare(left, right):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert torch.equal(a, b)
        check_residency(caches[1]); check_residency(caches[0])

    def step(i):
        g = torch.Generator(device='cpu').manual_seed(i)
        residual.copy_(torch.randn((1, H), generator=g).to(torch.bfloat16))
        noise.zero_()
        # mispredictions on some layers: a few experts the pre-gate cannot see coming
        wrong = torch.rand((layers, E), generator=g) < (0.02 * (i % 5))
        noise.masked_fill_(wrong.to(device), 1e3)

    def misses(cache):
        return int(cache.lru_stats[:, Stat.MISS].sum())

    step(0)
    for cache in caches:
        for _, bank in cache.banks:
            bank.fill_(165)
    for i in range(12):  # eager: compile and warm outside capture
        step(i)
        compare(execute(0), execute(1))
    torch.cuda.synchronize()
    assert int(pregate.rows.sum()) > 0, 'speculative plans admitted nothing'
    assert misses(caches[1]) < misses(caches[0]), 'speculation removed no real misses'
    for generation in range(2):
        graphs, outputs = [], []
        for arm in (0, 1):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = execute(arm)
            graphs.append(graph); outputs.append(out)
        assert not pregate._pending, 'every speculative plan joins inside the captured forward'
        for cache in caches:
            cache.reset()
            for _, bank in cache.banks:
                bank.fill_(165)
            cache.reset_stats()
        for i in range(120):
            step(1000 * generation + i)
            for graph in graphs:
                graph.replay()
            compare(*outputs)
        torch.cuda.synchronize()
        assert int(pregate.rows.sum()) > 0
        assert misses(caches[1]) < misses(caches[0])
        del graphs, outputs
    # Cancellation between fork and join: the next cache operation joins before slot reuse.
    cache = caches[1]
    pregate.launch(1, residual, norm, gates[1], 3, TOP_K)
    with pytest.raises(RuntimeError, match='cancellation'):
        raise RuntimeError('synthetic cancellation')
    cache.reset()
    for _, bank in cache.banks:
        bank.fill_(91)
    torch.cuda.synchronize()
    for _, bank in cache.banks:
        assert torch.all(bank == 91)


def test_kernels_match_torch_reference():
    from freetoken.moe import pregate as P

    device = torch.device('cuda')
    torch.manual_seed(3)
    H2, E2 = 2048, 256
    agree = 0
    for trial in range(50):
        res = torch.randn((1, H2), device=device).to(torch.bfloat16)
        w = (torch.rand(H2, device=device) + 0.5).to(torch.bfloat16)
        gate = torch.randn((E2, H2), device=device).to(torch.bfloat16)
        out = torch.zeros(E2, device=device)
        P._pregate_logits_kernel[(E2 // 8,)](res, w, gate, out, 1e-6, gate.stride(0), H=H2, BLOCK_H=H2, BLOCK_E=8,
                                              num_warps=8)
        x = res.float()
        y = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(torch.bfloat16).float()
        ref = (y @ gate.float().t())[0]
        assert float((out - ref).abs().max()) <= 1e-3 * float(ref.abs().max())
        agree += len(set(out.topk(8).indices.tolist()) & set(ref.topk(8).indices.tolist()))
        slots = torch.where(torch.rand(E2, device=device) < 0.5, -1, 7).to(torch.int32)
        query = torch.full((3,), -9, dtype=torch.int32, device=device)
        P._pregate_select_kernel[(1,)](out, slots, query, 2, E2, TOP_K=8, Q=3, BLOCK=E2, num_warps=1)
        values = out.tolist(); order = sorted(range(E2), key=lambda e: (-values[e], e))[:8]
        miss = [e for e in order if int(slots[e]) < 0][:2]
        pad = miss[0] if miss else order[0]
        assert query.tolist() == (miss + [pad] * 3)[:3]
    assert agree >= 0.99 * 50 * 8
