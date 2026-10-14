"""Exact CPU ownership/position checks for the opt-in two-slot retention policy.

These use the real request/prefill/cache/pool implementations. Synthetic states
encode their complete token prefix, so a stale slot or wrong boundary fails
exact equality independently of BF16 model arithmetic.
"""
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import patch

import torch

from freetoken.core import Batch, SamplingParams
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.prefill import ChunkedReq
from freetoken.scheduler.utils import PendingReq

_identity = runpy.run_path(str(Path(__file__).with_name('test_prefix_snapshot_identity.py')))
History = _identity['History']
fake_forward = _identity['fake_forward']
assert_stamp = _identity['assert_stamp']


def history(*, enabled=True, slots=9, page_size=1):
    with patch.dict('os.environ', {'QWEN36_RETAIN_CHUNK_SNAPSHOT': str(int(enabled))}):
        h = History(slots=slots)
        if page_size != 1:
            h.cm = CacheManager(40000, page_size, h.pages, 'hybrid_radix', linear_state_pool=h.pool)
            h.pm.cache_manager = h.cm
    return h


def audit(h, active=()):
    """Exact partition, not just an upper bound: no leaks, aliases or double-free."""
    free = list(h.pool._free_slots)
    tree = [n.mamba_value for n in h.cm.prefix_cache._snapshot_nodes()]
    own = []
    for r in active:
        if r.linear_slot_idx is not None:
            own.append(r.linear_slot_idx)
        if r.mamba_ping_pong is not None:
            own.extend(r.mamba_ping_pong)
        for mark, slot in zip(r.mamba_track_seqlens, r.mamba_ping_pong or ()):
            if mark is not None:
                assert_stamp(h.pool.recurrent_states[0, slot], r.input_ids[:mark])
    all_slots = free + tree + own
    assert len(all_slots) == len(set(all_slots)), 'multiple owners of a state slot'
    assert set(all_slots) == set(range(1, h.pool.num_slots)), 'lost state slot'
    h.cm.prefix_cache.check_integrity()
    # Tree nodes can be tombstoned in GDN while retaining full-attention KV.
    stack = [h.cm.prefix_cache.root]
    kv = []
    while stack:
        node = stack.pop()
        stack.extend(node.children.values())
        if not node.is_root():
            kv.extend(node._value[::h.cm.page_size].tolist())
    pages = h.cm.free_slots.tolist() + kv
    for r in active:
        end = ((r.cached_len + h.cm.page_size - 1)//h.cm.page_size)*h.cm.page_size
        pages.extend(h.pages[r.table_idx, r.cache_handle.cached_len:end:h.cm.page_size].tolist())
    assert len(pages) == len(set(pages)), 'duplicate/free tree or request KV page'
    assert set(pages) == set(range(0, h.cm.num_pages*h.cm.page_size, h.cm.page_size)), 'lost KV page'
    if not active:
        h.audit_tree()


def prefill(h, ids, *, budget=4096, margin=8, commit=True, stop_after=None):
    h.uid += 1
    pending = PendingReq(h.uid, ids, SamplingParams(max_tokens=4))
    h.pm.pending_list.append(pending)
    records = []
    with patch.dict('os.environ', {'QWEN36_PREFIX_MARGIN': str(margin)}):
        while h.pm.runnable:
            batch = h.pm.schedule_next_batch(budget)
            assert batch is not None
            r = batch.reqs[0]
            start = r.cached_len
            inherited = r.mamba_track_seqlens
            h.cm.allocate_paged([r])
            fla = fake_forward(batch, h.pool)
            record = dict(start=start, end=r.device_len, inherited=inherited,
                          marks=r.mamba_track_seqlens, slots=r.mamba_ping_pong,
                          last=r.mamba_last_track_seqlen,
                          has_snapshot=fla.track_dst is not None)
            r.complete_one()
            audit(h, [r])
            records.append(record)
            if stop_after is not None and len(records) == stop_after:
                return r, records
            if not isinstance(r, ChunkedReq):
                r.append_host(torch.tensor([90001], dtype=torch.int32))
                if commit:
                    h.cm.cache_req(r, finished=False)
                    audit(h, [r])
    return r, records


def finish(h, r):
    if isinstance(r, ChunkedReq):
        assert h.pm.abort_req(r.uid) is r
    h.finish(r)
    audit(h)


def test_edit_inside_final_chunk_restores_previous_boundary_without_extra_pool():
    for enabled in (False, True):
        h = history(enabled=enabled)
        allocated = h.pool.recurrent_states.numel() + h.pool.conv_states.numel()
        ids = torch.arange(3*4096+300, dtype=torch.int32)+51
        req, records = prefill(h, ids)
        assert req.cache_handle.cached_len == 12544
        finish(h, req)
        changed = ids.clone()
        changed[-100:] += 123456
        match = h.cm.match_req(SimpleNamespace(input_ids=changed, input_len=len(changed)))
        assert match.cuda_handle.cached_len == (12224 if enabled else 0)
        if enabled:
            assert_stamp(h.pool.recurrent_states[0, match.mamba_value], ids[:12224])
            assert h.cm.snapshot_stats['retained_donations'] == 1
            assert h.cm.snapshot_stats['latest_donations'] == 1
        assert h.pool.num_slots == 9
        assert allocated == h.pool.recurrent_states.numel() + h.pool.conv_states.numel()
        audit(h)


def test_short_final_chunks_carry_marks_by_physical_slot_for_both_margins():
    for margin in (1, 8):
        for budget in (128, 2048, 4096):
            for tail in (1, 7, 8, 63, 64, 65, 71, 72):
                h = history()
                ids = torch.arange(2*budget+tail, dtype=torch.int32)+21
                r, records = prefill(h, ids, budget=budget, margin=margin, commit=False)
                assert records[1]['inherited'] == records[0]['marks']
                assert records[2]['inherited'] == records[1]['marks']
                if tail < 64+margin:
                    assert not records[-1]['has_snapshot']
                    assert records[-1]['marks'] == records[-2]['marks']
                marks = sorted(x for x in records[-1]['marks'] if x is not None)
                assert len(marks) == 2 and marks[0] < marks[1] <= r.cached_len
                assert records[-1]['last'] == marks[-1]
                h.cm.cache_req(r, finished=False)
                assert r.mamba_track_seqlens == (None, None)
                assert r.mamba_last_track_seqlen is None
                audit(h, [r])
                finish(h, r)


def test_restore_is_copy_on_write_and_duplicate_donation_keeps_canonical_pages():
    h = history(slots=12)
    ids = torch.arange(12588, dtype=torch.int32)+13
    first, _ = prefill(h, ids)
    finish(h, first)
    # End earlier than the existing latest snapshot: restore the older state.
    # Inserting a new boundary inside existing KV must keep canonical pages.
    short = ids[:12540].clone()
    r, records = prefill(h, short, budget=4096, margin=1, commit=False)
    assert records[0]['start'] == 12224
    old = r.cache_handle
    original = h.pool.recurrent_states[0, old.node.mamba_value].clone()
    assert_stamp(original, ids[:12224])
    assert not torch.equal(h.pool.recurrent_states[0, r.linear_slot_idx], original)
    canonical = h.cm.prefix_cache.match_prefix(ids[:12544]).kv_indices[:12480].clone()
    h.cm.cache_req(r, finished=False)
    assert r.cache_handle.cached_len == 12480
    assert torch.equal(h.pages[r.table_idx, :12480], canonical)
    assert torch.equal(h.pool.recurrent_states[0, old.node.mamba_value], original)
    audit(h, [r])
    finish(h, r)


def test_duplicate_snapshot_donations_from_concurrent_stale_handles_free_once():
    for terminal in (False, True):
        h = history(slots=16)
        ids = torch.arange(1300, dtype=torch.int32)+501
        a, _ = prefill(h, ids, budget=512, commit=False)
        # Admit before a is committed, like two requests sharing the same
        # uncached prompt in a batch: both legitimately own duplicate KV.
        h.pm.pending_list.append(PendingReq(999, ids.clone(), SamplingParams(max_tokens=4)))
        b = None
        with patch.dict('os.environ', {'QWEN36_PREFIX_MARGIN': '8'}):
            while h.pm.runnable:
                batch = h.pm.schedule_next_batch(512)
                b = batch.reqs[0]
                h.cm.allocate_paged([b])
                fake_forward(batch, h.pool)
                b.complete_one()
        b.append_host(torch.tensor([90001], dtype=torch.int32))
        audit(h, [a, b])
        h.cm.cache_req(a, finished=False)
        audit(h, [a, b])
        canonical = h.pages[a.table_idx, :a.cache_handle.cached_len].clone()
        h.cm.cache_req(b, finished=terminal)
        assert h.cm.snapshot_stats['deduplicated_donations'] == 2
        if terminal:
            h.tm.free(b.table_idx)
            b.table_idx = -1
            audit(h, [a])
        else:
            assert torch.equal(h.pages[b.table_idx, :b.cache_handle.cached_len], canonical)
            audit(h, [a, b])
            h.cm.cache_req(b, finished=True)
            h.tm.free(b.table_idx)
            b.table_idx = -1
        finish(h, a)


def test_pool_pressure_recycles_only_unlocked_states_and_preserves_conservation():
    for slots in (5, 6, 9):
        h = history(slots=slots)
        for i in range(20):
            ids = torch.arange(1400+i*3, dtype=torch.int32)+100000*(i+1)
            r, records = prefill(h, ids, budget=512)
            assert records[0]['start'] == 0 and records[0]['inherited'] == (None, None)
            finish(h, r)
        assert h.pool.num_slots == slots
        assert h.cm.snapshot_stats['retained_donations'] == 20


def test_abort_and_terminal_drain_donate_owned_snapshots_without_replacement():
    for partial in (False, True):
        for page_size in (1, 16, 64):
            h = history(page_size=page_size)
            ids = torch.arange(12589, dtype=torch.int32)+103
            r, records = prefill(h, ids, commit=False, stop_after=3 if partial else None)
            assert len([x for x in r.mamba_track_seqlens if x is not None]) == 2
            r.aborted = partial
            with patch.object(h.pool, 'alloc', side_effect=AssertionError('finish allocated replacement')):
                finish(h, r)
            assert r.mamba_ping_pong is None and r.linear_slot_idx is None
            assert r.mamba_track_seqlens == (None, None)
            # Defense in depth for a repeated _free_req_slots call.
            h.cm._free_req_slots(r)
            audit(h)


def test_anchor_after_prefill_uses_idle_slot_with_no_stale_chunk_marks():
    h = history()
    ids = torch.arange(12588, dtype=torch.int32)+89
    r, _ = prefill(h, ids)
    assert r.mamba_track_seqlens == (None, None)
    r.toolcall_anchor_len = r.input_ids.numel()
    anchor = r.toolcall_anchor_len
    h.decode(r, 90002)
    assert r.mamba_last_track_seqlen is None
    h.decode(r, 90003)
    marks = [x for x in r.mamba_track_seqlens if x is not None]
    assert marks == [anchor]
    audit(h, [r])
    anchor_ids = r.input_ids[:anchor].clone()
    finish(h, r)
    match = h.cm.match_req(SimpleNamespace(
        input_ids=torch.cat((anchor_ids, torch.tensor([99], dtype=torch.int32))), input_len=anchor+1))
    assert match.cuda_handle.cached_len == anchor
    assert_stamp(h.pool.recurrent_states[0, match.mamba_value], anchor_ids)


def test_abort_before_any_forward_has_no_valid_snapshot_or_tree_donation():
    h = history()
    h.pm.pending_list.append(PendingReq(777, torch.arange(1000, dtype=torch.int32), SamplingParams()))
    batch = h.pm.schedule_next_batch(512)
    r = batch.reqs[0]
    assert h.cm._pending_hybrid_snapshots(r) == []
    assert r.cached_len == 0
    assert h.pm.abort_req(r.uid) is r
    h.cm.cache_req(r, finished=True)
    h.tm.free(r.table_idx)
    r.table_idx = -1
    audit(h)
    assert h.cm.snapshot_stats['retained_donations'] == 0
    assert h.pool.num_free_slots == h.pool.num_slots-1


def test_batch_tracks_nonzero_prefixes_and_never_cross_assigns_slot_marks():
    h = history(slots=9)
    a = torch.arange(1600, dtype=torch.int32)+71
    b = torch.arange(1900, dtype=torch.int32)+9991
    h.pm.pending_list.extend([PendingReq(1, a, SamplingParams()), PendingReq(2, b, SamplingParams())])
    batch = h.pm.schedule_next_batch(3500)
    assert len(batch.reqs) == 2
    h.cm.allocate_paged(batch.reqs)
    with patch.dict('os.environ', {'QWEN36_PREFIX_MARGIN': '8'}):
        fake_forward(batch, h.pool)
    for r in batch.reqs:
        r.complete_one()
        assert len([x for x in r.mamba_track_seqlens if x is not None]) == 1
    audit(h, batch.reqs)
    for r in batch.reqs:
        h.cm.cache_req(r, finished=True)
        h.tm.free(r.table_idx)
        r.table_idx = -1
    audit(h)


def scheduler_stub(h):
    from freetoken.scheduler.scheduler import Scheduler
    fixture = runpy.run_path(str(Path(__file__).with_name('test_abort_inflight_prefill.py')))
    stub = fixture['_setup']()[-1]
    stub.cache_manager, stub.table_manager = h.cm, h.tm
    stub.decode_manager, stub.prefill_manager = h.dm, h.pm
    stub.engine = None
    stub._kv_usage_pages = h.cm.page_usage
    stub._free_req_resources = lambda r: Scheduler._free_req_resources(stub, r)
    return stub, fixture['_as_last_data']


def test_overlap_drains_no_intermediate_snapshot_and_commits_both_only_at_final():
    from freetoken.scheduler.scheduler import Scheduler
    h = history()
    stub, as_data = scheduler_stub(h)
    h.pm.pending_list.append(PendingReq(123, torch.arange(1600, dtype=torch.int32)+19,
                                        SamplingParams(max_tokens=4)))
    last = None
    final = None
    with patch.dict('os.environ', {'QWEN36_PREFIX_MARGIN': '8'}), \
         patch.object(h.cm, 'cache_req', wraps=h.cm.cache_req) as commits:
        while h.pm.runnable or last is not None:
            batch = h.pm.schedule_next_batch(512)
            if batch is not None:
                h.cm.allocate_paged(batch.reqs)
                fake_forward(batch, h.pool)
                for r in batch.reqs:
                    r.complete_one()
            # Next chunk has already inherited metadata and overwritten its
            # physical ping-pong slot when the previous drain runs.
            if last is not None:
                Scheduler._process_last_data(stub, as_data(last))
                if isinstance(last.reqs[0], ChunkedReq):
                    assert commits.call_count == 0
                else:
                    final = last.reqs[0]
            last = batch
        assert commits.call_count == 1
    assert final.mamba_track_seqlens == (None, None)
    assert h.cm.snapshot_stats['retained_donations'] == 1
    assert h.cm.snapshot_stats['latest_donations'] == 1
    audit(h, [final])
    finish(h, final)


def test_abort_inflight_with_two_carried_snapshots_defers_release_until_drain():
    from freetoken.message import AbortBackendMsg
    from freetoken.scheduler.scheduler import Scheduler
    h = history()
    stub, as_data = scheduler_stub(h)
    r, _ = prefill(h, torch.arange(1700, dtype=torch.int32)+29,
                   budget=512, commit=False, stop_after=3)
    assert isinstance(r, ChunkedReq)
    stub._last_data = as_data(Batch([r], 'prefill'))
    before = list(h.pool._free_slots)
    Scheduler._process_one_msg(stub, AbortBackendMsg(uid=r.uid))
    assert r.aborted and r.table_idx != -1
    assert h.pool._free_slots == before
    assert not h.pm.runnable
    Scheduler._process_last_data(stub, stub._last_data)
    assert r.table_idx == -1 and r.mamba_ping_pong is None
    assert h.cm.snapshot_stats['retained_donations'] == 1
    assert h.cm.snapshot_stats['latest_donations'] == 1
    audit(h)
