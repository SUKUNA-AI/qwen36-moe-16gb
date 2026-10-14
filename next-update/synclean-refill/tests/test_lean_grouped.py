"""QWEN36_SYNC_LEAN rebased onto the release build's grouped attention (QWEN36_LAYER_MAJOR_ATTN_BATCH=k).

CPU toy stack of work/host-sync/tests/sync_lean/fi_toy.py (REAL FlashInferBackend over a fake flashinfer whose
kernel reads its plan back from the wrapper's int workspace), with three test-side changes so that the
executor really groups:
  * plan info like FlashInfer 0.6 (15 fields, [3] CTA_TILE_Q and [14] split-KV from toy.toy_fi_plan);
  * a row-invariant fake kernel (every query row attends exactly its own key prefix), so one call over a
    group is bitwise the per-chunk calls, as on the GPU under the executor's conditions;
  * ``run(out=...)`` honoured (grouped attention writes into the scratch buffer).
Claims: chunked == layer-major k=2/3 legacy == k=2/3 lean, bitwise, incl. windows and k=1 fallbacks;
lean plans once per (chunk|group, window) with exactly the legacy plan arguments; mutations of the
grouped pool logic are caught. Plus the CUDA GDN empty-sequence seed.
"""
import contextlib
import importlib.util
import pathlib
import sys
from unittest.mock import patch

import torch

_HS = pathlib.Path("/path/to/qwen36-workspace/work/host-sync/tests/sync_lean/fi_toy.py")
_spec = importlib.util.spec_from_file_location("hs_fi_toy", _HS)
ft = sys.modules.get("hs_fi_toy") or importlib.util.module_from_spec(_spec)
if "hs_fi_toy" not in sys.modules:
    sys.modules["hs_fi_toy"] = ft
    _spec.loader.exec_module(ft)
toy = ft.toy

from freetoken.engine import layer_major as elm  # noqa: E402

N_ATTN = sum(p == "attn" for p in toy.PATTERN)
_real_stage = ft._FakeWrapperBase._stage
SPLIT_KV_ABOVE = [None]   # test knob: member/group plans with kv > this report split-KV
GROUP_CTA = [None]        # test knob: group plans (rows > one chunk) report this CTA_TILE_Q


def _stage(self, kind, qo, kv, kwargs):
    _real_stage(self, kind, qo, kv, kwargs)
    rows, kv_len = int(qo[-1] - qo[0]), int(kv[-1] - kv[0])
    cta, split, _ = toy.toy_fi_plan(max(1, rows), kv_len, toy.NQ, 1, toy.HD, 1)
    if SPLIT_KV_ABOVE[0] is not None and kind == "prefill" and kv_len > SPLIT_KV_ABOVE[0]:
        split = True
    if GROUP_CTA[0] is not None and kind == "prefill" and rows > 128:
        cta = GROUP_CTA[0]
    self._plan_info = [self._plan_info[0], 0, 0, cta, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, int(split)]


def _attend(self, q, paged_kv_cache, causal):
    qo, kv = self._workspace_plan()  # the kernel reads its plan from the int workspace
    k_cache, v_cache = paged_kv_cache
    nq, hd = toy.NQ, toy.HD
    qq_all = q.reshape(q.shape[0], nq, hd)
    outs = []
    for i in range(len(qo) - 1):
        lo, hi = qo[i], qo[i + 1]
        idx = self._paged_kv_indices_buf[kv[i]:kv[i + 1]].long()
        kv_len, q_len = kv[i + 1] - kv[i], hi - lo
        for r in range(q_len):
            p = kv_len - q_len + r  # causal: own key prefix (decode: q_len 1 -> all keys)
            kk = k_cache[idx[:p + 1], 0, 0].float()
            vv = v_cache[idx[:p + 1], 0, 0].float()
            s = torch.einsum("hd,kd->hk", qq_all[lo + r].float(), kk) / hd ** 0.5
            outs.append(torch.einsum("hk,kd->hd", s.softmax(-1), vv).reshape(1, nq * hd))
    return torch.cat(outs).to(ft.BF)


def _run(self, q, paged_kv_cache, out=None, **kw):
    res = self._attend(q, paged_kv_cache, causal=True)
    if out is not None:
        out.copy_(res.view(out.shape))
        return out
    return res


PATCHES = (lambda: patch.object(ft._FakeWrapperBase, "_stage", _stage),
           lambda: patch.object(ft._FakeWrapperBase, "_attend", _attend),
           lambda: patch.object(ft.FakePrefillWrapper, "run", _run))


def serve(mode, reqs, *, k=1, lean=False, extra=None, patches=()):
    env = dict(QWEN36_LAYER_MAJOR_ATTN_BATCH=str(k), **(extra or {}))
    return ft.serve(mode, reqs, lean=lean, extra_env=env, patches=[p() for p in PATCHES] + list(patches))


def _one(n, seed, max_tokens=3):
    return [[toy.user(1, ft.prompt(n, seed), max_tokens)]]


@contextlib.contextmanager
def knobs(split=None, group_cta=None):
    SPLIT_KV_ABOVE[0], GROUP_CTA[0] = split, group_cta
    try:
        yield
    finally:
        SPLIT_KV_ABOVE[0] = GROUP_CTA[0] = None


def test_grouped_lean_is_bitwise_chunked_and_legacy():
    for n in (517, 1000, 1031):
        reqs = _one(n, n)
        ref = ft.state(*serve("chunk", reqs)[:2])
        for k in (2, 3):
            el, sl, _ = serve("lm", reqs, k=k)
            en, sn, _ = serve("lm", reqs, k=k, lean=True)
            ft.assert_same(ref, ft.state(el, sl))
            ft.assert_same(ref, ft.state(en, sn))
            for eng in (el, en):
                st = eng.layer_major_stats
                assert st["last_attn_batch"] == k and st["attn_groups"] > 0 and st["attn_fallback_groups"] == 0, st


def test_grouped_lean_plan_counts():
    # 1000 tokens / budget 128: chunks 0-6 full, chunk 7 = 104 rows; k=2 per attention layer:
    # groups (0,1) (2,3) (4,5), singles 6 and 7. Probes: chunks 0-5 once per window.
    reqs = _one(1000, 3, max_tokens=4)
    _, _, cl = serve("lm", reqs, k=2)
    _, _, cn = serve("lm", reqs, k=2, lean=True)
    decode = cl["decode_plans"]
    assert decode == cn["decode_plans"] == 3
    # legacy: probes 6 + every grouped task re-plans (3 x N_ATTN) + every single task re-plans (2 x N_ATTN)
    assert cl["prefill_plans"] == 6 + 3 * N_ATTN + 2 * N_ATTN, cl["prefill_plans"]
    # lean: probes 6 + one plan per group (3) + one per single chunk (2), per window
    assert cn["prefill_plans"] == 6 + 3 + 2, cn["prefill_plans"]
    assert cl["shared_syncs"] == cl["prefill_plans"] + decode
    assert cn["shared_syncs"] == decode            # lean prefill never waits on the shared event
    assert cn["pool_syncs"] == cn["prefill_plans"]  # own staging events only


def test_grouped_lean_plans_get_exactly_the_legacy_arguments():
    reqs = [[toy.user(1, ft.prompt(1000, 9), 3)], [toy.user(2, ft.prompt(530, 10), 3)]]
    _, _, cl = serve("lm", reqs, k=2)
    _, _, cn = serve("lm", reqs, k=2, lean=True)

    def plans(counts):
        return [tuple(sorted(kw.items(), key=lambda x: x[0])) for kind, _key, kw in counts["kwargs"] if kind == "prefill"]

    legacy, lean = plans(cl), plans(cn)
    assert set(lean) == set(legacy)
    assert len(lean) == len(set(lean))  # each distinct plan (member / group / single) exactly once


def test_grouped_lean_windows_prefix_hits_back_to_back():
    base = ft.prompt(900, 11)
    reqs = [[toy.user(1, base, 4)], [toy.user(2, torch.cat([base[:800], ft.prompt(300, 12)]), 4)],
            [toy.user(3, ft.prompt(1031, 13), 3)]]
    for extra in ({}, {"QWEN36_LAYER_MAJOR_WINDOW_CHUNKS": "3"}, {"QWEN36_LAYER_MAJOR_WINDOW_CHUNKS": "4"},
                  {"QWEN36_LAYER_MAJOR_STEP_TASKS": "1", "QWEN36_LAYER_MAJOR_WINDOW_CHUNKS": "2"}):
        ref = ft.state(*serve("chunk", reqs, extra=extra)[:2])
        el, sl, cl = serve("lm", reqs, k=2, extra=extra)
        en, sn, cn = serve("lm", reqs, k=2, lean=True, extra=extra)
        ft.assert_same(ref, ft.state(el, sl))
        ft.assert_same(ref, ft.state(en, sn))
        assert cn["prefill_plans"] <= cl["prefill_plans"], (extra, cn["prefill_plans"], cl["prefill_plans"])
        pool = en.attn_backend.__dict__.get("_qwen36_chunk_wrappers", [])
        assert all(w is None or w._paged_kv_indices_buf is None for w in pool)  # released after each job


def test_grouped_lean_member_split_fallback_is_bitwise():
    """Members whose own plan splits KV (kv-length dependent) are run one by one on their own pooled
    wrappers (planned at the probe, no re-plan)."""
    reqs = _one(1000, 21)
    with knobs(split=500):
        ref = ft.state(*serve("chunk", reqs)[:2])
        el, sl, cl = serve("lm", reqs, k=2)
        en, sn, cn = serve("lm", reqs, k=2, lean=True)
    ft.assert_same(ref, ft.state(el, sl))
    ft.assert_same(ref, ft.state(en, sn))
    for eng in (el, en):
        st = eng.layer_major_stats
        assert st["attn_fallback_groups"] > 0 and st["attn_groups"] > 0, st
    assert cn["prefill_plans"] < cl["prefill_plans"]


def test_grouped_lean_group_plan_mismatch_fallback_is_bitwise():
    """A group plan whose CTA_TILE_Q differs from the members' disables grouping for the prompt; the
    remaining tasks run per chunk on the members' pooled wrappers."""
    reqs = _one(1000, 22)
    with knobs(group_cta=64):
        ref = ft.state(*serve("chunk", reqs)[:2])
        el, sl, _ = serve("lm", reqs, k=2)
        en, sn, _ = serve("lm", reqs, k=2, lean=True)
    ft.assert_same(ref, ft.state(el, sl))
    ft.assert_same(ref, ft.state(en, sn))
    assert en.layer_major_stats["attn_groups"] == 0 and en.layer_major_stats["attn_fallback_groups"] > 0


def test_flag_off_creates_no_pool_and_matches_release_counts():
    reqs = _one(1000, 31)
    eng, _, off = serve("lm", reqs, k=2)
    assert "_qwen36_chunk_wrappers" not in eng.attn_backend.__dict__
    assert off["pool_syncs"] == 0


# --------------------------------------------------------------------------------- mutations
def _lean_differs(patches, reqs=None, extra=None, kn=None):
    reqs = reqs or _one(1000, 41)
    with knobs(**(kn or {})):
        ref = ft.state(*serve("chunk", reqs, extra=extra)[:2])
        try:
            mut = ft.state(*serve("lm", reqs, k=2, lean=True, extra=extra, patches=patches)[:2])
        except Exception:  # noqa: BLE001 -- a crash also counts as caught
            return True
    return not ft.same(ref, mut)


def test_mutation_group_plans_on_a_member_wrapper_is_caught():
    """The group plan on the first member's pooled wrapper clobbers that member's probe plan; the k=1
    fallback after a group-plan mismatch then runs the member with the group's plan."""
    real = ft.fi.FlashInferBackend.chunk_plan_wrapper

    def attend(self, layer, batches, rows, q, out, key, _real=elm.LayerMajorExecutor._attend):
        W = self.plan.window_chunks
        with patch.object(ft.fi.FlashInferBackend, "chunk_plan_wrapper",
                          lambda be, index, r: real(be, index - W if index >= W else index, r)):
            return _real(self, layer, batches, rows, q, out, key)

    assert _lean_differs([patch.object(elm.LayerMajorExecutor, "_attend", attend)], kn={"group_cta": 64})


def test_mutation_stale_pooled_plan_across_windows_is_caught():
    def plan_now(self, meta, _real=elm.LayerMajorExecutor._plan_now):
        if elm._pooled(meta) and meta.wrapper._plan_info is not None:
            meta.initialized = True  # reuse whatever the pool slot holds (a previous window's plan)
        return _real(self, meta)

    assert _lean_differs([patch.object(elm.LayerMajorExecutor, "_plan_now", plan_now),
                          patch.object(ft.fi.FlashInferBackend, "release_chunk_plans", lambda self: None)],
                         extra={"QWEN36_LAYER_MAJOR_WINDOW_CHUNKS": "4"})


def test_mutation_release_inside_a_grouped_window_is_caught():
    real = elm.LayerMajorExecutor.run_attn_group

    def release_every_group(self, layer, locals_, batches, key):
        self.engine.attn_backend.release_chunk_plans()
        return real(self, layer, locals_, batches, key)

    assert _lean_differs([patch.object(elm.LayerMajorExecutor, "run_attn_group", release_every_group)])


# ------------------------------------------------------------------- CUDA GDN empty-sequence seed
def test_empty_sequence_seed_matches_and_avoids_item():
    from freetoken.kernel import cuda_gdn

    for host in ([0, 5, 5, 9], [0, 3, 7], [0, 0], [0, 4]):
        cu_host = torch.tensor(host, dtype=torch.int64)
        dev = cu_host.clone()  # stand-in for the device copy (a distinct tensor object)
        expect = bool((dev[1:] == dev[:-1]).any().item())
        cuda_gdn.seed_empty_sequence(dev, cu_host)
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("drained")):
            assert cuda_gdn.has_empty_sequence(dev) is expect
    # mismatched copies are never seeded (the lookup then computes)
    dev = torch.tensor([0, 2, 2], dtype=torch.int64)
    cuda_gdn.seed_empty_sequence(dev, torch.tensor([0, 2], dtype=torch.int64))
    assert cuda_gdn.has_empty_sequence(dev) is True


def test_build_fla_metadata_seeds_empty_sequence_only_when_lean():
    from types import SimpleNamespace

    from freetoken.attention import linear
    from freetoken.kernel import cuda_gdn

    reqs = [SimpleNamespace(extend_len=n, cached_len=0, linear_slot_idx=i, table_idx=i, device_len=n)
            for i, n in enumerate((5, 7))]
    batch = SimpleNamespace(padded_reqs=reqs, is_decode=False, linear_table_idx=None)
    with patch.object(linear, "_build_track_metadata", lambda *a, **k: {}):
        for lean in ("0", "1"):
            with patch.dict("os.environ", {"QWEN36_SYNC_LEAN": lean}):
                fla = linear.build_fla_metadata(batch, torch.device("cpu"))
            calls = []
            real_item = torch.Tensor.item

            def item(t, _real=real_item):
                calls.append(1)
                return _real(t)

            with patch.object(torch.Tensor, "item", item):
                assert cuda_gdn.has_empty_sequence(fla.cu_seqlens) is False
            assert (len(calls) == 0) == (lean == "1"), (lean, calls)
