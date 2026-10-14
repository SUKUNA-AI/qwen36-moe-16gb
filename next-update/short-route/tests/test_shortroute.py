"""CPU tests of the short-route patch (QWEN36_LAYER_MAJOR_MIN_TOKENS) on the layer-major toy stack
(work/layer-major/tests/layer_major/toy.py): routing by uncached tokens, prefix hits, bitwise equality
with the chunked reference, hit-D2D coexistence gate and the full-copy layer-major epoch."""
import importlib.util
import pathlib
import sys
from unittest.mock import patch

import torch

LM_TESTS = pathlib.Path('/path/to/qwen36-workspace/work/layer-major/tests/layer_major')
_spec = importlib.util.spec_from_file_location("lm_eq", LM_TESTS / "test_toy_equivalence.py")
eq = sys.modules.get("lm_eq") or importlib.util.module_from_spec(_spec)
if "lm_eq" not in sys.modules:
    sys.modules["lm_eq"] = eq
    _spec.loader.exec_module(eq)
toy = eq.toy

from freetoken.moe.offload_cache import OffloadMoeCache  # noqa: E402


def lm_prefills(eng):
    return (getattr(eng, "layer_major_stats", None) or {}).get("prefills", 0)


def test_min_tokens_parsing():
    from freetoken.engine.layer_major import min_tokens

    for value, want in ((None, 0), ("", 0), ("0", 0), ("2049", 2049), ("-5", 0), ("x", 0)):
        env = {} if value is None else {"QWEN36_LAYER_MAJOR_MIN_TOKENS": value}
        with patch.dict("os.environ", env, clear=False):
            if value is None:
                import os
                os.environ.pop("QWEN36_LAYER_MAJOR_MIN_TOKENS", None)
            assert min_tokens() == want, value


def test_hit_d2d_refused_by_default_allowed_with_min_tokens():
    from freetoken.engine.layer_major import engine_reason

    with patch.dict("os.environ", dict(eq.BASE_ENV, QWEN36_LAYER_MAJOR_PREFILL="1", QWEN36_LAYER_MAJOR_MIN_TOKENS="0")):
        eng = toy.ToyEngine()
        eng.moe_offload_cache.prefill_hit_d2d = True
        assert "hit-D2D" in engine_reason(eng)
    with patch.dict("os.environ", dict(eq.BASE_ENV, QWEN36_LAYER_MAJOR_PREFILL="1", QWEN36_LAYER_MAJOR_MIN_TOKENS="300")):
        eng = toy.ToyEngine()
        eng.moe_offload_cache.prefill_hit_d2d = True
        assert engine_reason(eng) is None
        eng.native_mtp = object()  # every other gate still applies
        assert "MTP" in engine_reason(eng)


def test_layer_major_epoch_never_uses_hit_d2d():
    cache = OffloadMoeCache(num_layers=2, num_experts=4, cache_size=12, device=torch.device("cpu"),
                            prefill_overlap=True, quant_format="bf16", prefill_hit_d2d=True)
    seen = []
    orig = OffloadMoeCache.begin_prefill

    def spy(self):
        seen.append(self.prefill_hit_d2d)
        return orig(self)

    with patch.object(OffloadMoeCache, "begin_prefill", spy):
        cache._prefill_hit_d2d_active = True  # left over from a previous chunked prefill
        cache.begin_layer_major_prefill()
    assert seen == [False] and cache.prefill_hit_d2d is True
    assert cache._prefill_hit_d2d_active is False and cache.layer_major_active


def test_routing_by_uncached_tokens_bitwise_equal():
    """budget 128: 1000 tokens (8 chunks) >= T -> layer-major; 400 tokens (4 chunks) < T -> chunked."""
    env = {"QWEN36_LAYER_MAJOR_MIN_TOKENS": "600"}
    for n, want in ((1000, 1), (599, 0), (600, 1), (400, 0), (100, 0)):
        reqs = [[toy.user(1, eq.prompt(n, n), max_tokens=4)]]
        ref = eq.state(*eq.serve("chunk", reqs))
        eng, sched = eq.serve("lm", reqs, extra_env=env)
        assert lm_prefills(eng) == want, (n, eng.__dict__.get("layer_major_stats"))
        eq.assert_same(ref, eq.state(eng, sched))
        sched.cache_manager.check_integrity()


def test_prefix_hit_counts_only_uncached_tokens():
    """uid 2 hits a retained snapshot (~704 cached) with 296 new tokens (>= 2 chunks): layer-major by
    default, chunked with T=600; uid 1 (900 new) stays layer-major. Bits equal the chunked run."""
    base = eq.prompt(900, 11)
    follow = torch.cat([base[:800], eq.prompt(200, 12)])
    reqs = [[toy.user(1, base, 4)], [toy.user(2, follow, 4)]]
    ref = eq.state(*eq.serve("chunk", reqs))
    default_eng, default_sched = eq.serve("lm", reqs)
    assert lm_prefills(default_eng) == 2
    eq.assert_same(ref, eq.state(default_eng, default_sched))
    eng, sched = eq.serve("lm", reqs, extra_env={"QWEN36_LAYER_MAJOR_MIN_TOKENS": "600"})
    assert lm_prefills(eng) == 1
    eq.assert_same(ref, eq.state(eng, sched))
    sched.cache_manager.check_integrity()


def test_unset_threshold_is_the_current_behaviour():
    reqs = [[toy.user(1, eq.prompt(700, 41), 6), toy.user(2, eq.prompt(450, 42), 3)]]
    a_eng, a_sched = eq.serve("lm", reqs)
    b_eng, b_sched = eq.serve("lm", reqs, extra_env={"QWEN36_LAYER_MAJOR_MIN_TOKENS": "0"})
    assert lm_prefills(a_eng) == lm_prefills(b_eng) == 2
    eq.assert_same(eq.state(a_eng, a_sched), eq.state(b_eng, b_sched))


def test_hit_d2d_with_threshold_serves_both_paths():
    """hit-D2D on (CPU: falls back to full copies) + T: short prompt chunked, long one layer-major."""
    reqs = [[toy.user(1, eq.prompt(1000, 71), 4)], [toy.user(2, eq.prompt(300, 72), 4)]]
    ref = eq.state(*eq.serve("chunk", reqs))
    env = dict(eq.BASE_ENV, QWEN36_LAYER_MAJOR_PREFILL="1", QWEN36_LAYER_MAJOR_MIN_TOKENS="600")
    with patch.dict("os.environ", env):
        eng = toy.ToyEngine(lend_slots=64)
        eng.moe_offload_cache.prefill_hit_d2d = True
        sched = toy.CpuScheduler(eng, budget=128)
        for msgs in reqs:
            toy.run(sched, msgs, overlap=True)
    assert lm_prefills(eng) == 1
    eq.assert_same(ref, eq.state(eng, sched))
