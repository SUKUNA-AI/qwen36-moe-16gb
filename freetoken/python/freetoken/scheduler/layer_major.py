"""Scheduler side of layer-major prefill (opt-in: QWEN36_LAYER_MAJOR_PREFILL=1).

One long prompt is prefilled ``for layer: for chunk:`` instead of ``for chunk: for layer:``
(see freetoken.engine.layer_major for why the result is bitwise the chunked one). The scheduler
owns the request for the whole prefill through a ``LayerMajorJob``:

  * Start: the first chunk was scheduled and prepared by the normal path (admission, prefix
    match, KV pages, GDN restore). The job takes the request out of ``pending_list`` and becomes
    ``PrefillManager.layer_major_job`` (``runnable`` stays True, so the loop never blocks, idle
    rebuilds are refused/deferred and no other prefill or decode batch is scheduled: other
    requests wait in the queue, exactly as behind a chunked prefill).
  * Steps: each scheduler iteration enqueues up to ``QWEN36_LAYER_MAJOR_STEP_TASKS`` (layer,
    chunk) tasks (default num_layers // 2, i.e. half the GPU work of one chunked iteration), with
    at most two steps in flight, so incoming messages (abort, new requests) are handled at the
    same or finer granularity than chunk by chunk.
  * Chunks k >= 1 are prepared lazily, right before their layer-0 task: the host bookkeeping a
    chunk's forward would do (``complete_one``: cached_len advance) runs without a forward, then
    the continuation is built by the same PrefillAdder rule and ``_prepare_batch`` (pages,
    positions, FLA track snapshots, attention metadata) -- identical chunk boundaries/metadata.
  * Finish: the last chunk's final norm + LM head + sampling; the result is handed to the loop
    as the usual (ForwardInput, ForwardOutput) pair, so ``_process_last_data`` commits the
    prefix (KV + both retained/latest GDN ping-pong snapshots, whose bytes are final because every
    layer processed its chunks in order) exactly as after the final chunk of a chunked prefill.
  * Abort: the job drains the enqueued tasks and frees everything the request owns WITHOUT
    inserting anything into the prefix cache (layers past the abort point never ran, so neither
    the KV of later layers nor the GDN states are valid).
"""
from __future__ import annotations

import collections
import os
from typing import TYPE_CHECKING

import torch

from freetoken.utils import div_ceil, init_logger

if TYPE_CHECKING:
    from .scheduler import ForwardData, ForwardInput, Scheduler

logger = init_logger(__name__)


_QUIET = {"disabled", "not a single-request prefill", "prompt fits one chunk"}
_LOGGED: set[str] = set()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def plan_chunk_extends(first_end: int, input_len: int, budget: int, align: int) -> list[int]:
    """Extend lengths of the continuation chunks after a first chunk ending at ``first_end``:
    the PrefillAdder._add_one_req rule for a lone continuation (full budget, no SWA, no images)."""
    out, cached = [], first_end
    while cached < input_len:
        remain = input_len - cached
        size = min(budget, remain)
        if align > 1 and 0 < size < remain:
            aligned = (cached + size) // align * align - cached
            size = aligned if aligned > 0 else size
        out.append(size)
        cached += size
    return out


def task_order(windows: list[tuple[int, int]], num_layers: int, group_layers=(), k: int = 1,
               groupable=None):
    """(window, layer, chunks) in execution order: windows in order, inside a window L-major.
    ``chunks`` is a tuple of consecutive chunk indices processed by one task: on the layers in
    ``group_layers`` up to ``k`` consecutive ``groupable`` chunks of the window (never across a
    window boundary), else one chunk."""
    group_layers = set(group_layers) if k > 1 else set()
    for w, (a, b) in enumerate(windows):
        for layer in range(num_layers):
            if layer not in group_layers:
                for c in range(a, b):
                    yield w, layer, (c,)
                continue
            run: list[int] = []
            for c in range(a, b):
                if groupable is not None and not groupable(c):
                    if run:
                        yield w, layer, tuple(run)
                        run = []
                    yield w, layer, (c,)
                    continue
                run.append(c)
                if len(run) == k:
                    yield w, layer, tuple(run)
                    run = []
            if run:
                yield w, layer, tuple(run)


def scheduler_reason(sched: "Scheduler", fi: "ForwardInput") -> str | None:
    """None if this freshly prepared first chunk may start a layer-major job, else why not."""
    from freetoken.engine.layer_major import enabled, engine_reason

    from .prefill import ChunkedReq

    if not enabled():
        return "disabled"
    batch = fi.batch
    if not batch.is_prefill or len(batch.reqs) != 1:
        return "not a single-request prefill"
    req = batch.reqs[0]
    if not isinstance(req, ChunkedReq):
        return "prompt fits one chunk"
    pm = sched.prefill_manager
    if pm.layer_major_job is not None:
        return "a layer-major job is already running"
    if not pm.pending_list or pm.pending_list[0].uid != req.uid or pm.pending_list[0].chunked_req is not req:
        return "continuation is not at the head of the queue"
    if sched.decode_manager.running_reqs:
        return "decode requests are running"
    if req.mm_items or batch.mm_gather_plan:
        return "multimodal prompt"
    cm = sched.cache_manager
    if cm.swa_paged or cm.is_swa or cm.prefill_chunk_budget:
        return "sliding-window cache"
    if os.environ.get("QWEN36_SPEC"):
        return "speculative decoding"
    return engine_reason(sched.engine)


class LayerMajorJob:
    def __init__(self, sched: "Scheduler", first: "ForwardInput", extends: list[int], plan,
                 bank_ids=None, refill: bool = False, attn_k: int = 1, attn_layers=()):
        from freetoken.engine.layer_major import LayerMajorExecutor, attn_batch

        self.sched = sched
        self.engine = sched.engine
        self.attn_requested = attn_batch()
        req = first.batch.reqs[0]
        self.uid = req.uid
        pm = sched.prefill_manager
        self.pending = pm.pending_list.pop(0)
        self.extends = extends
        self.plan = plan
        self.windows = plan.windows()
        self.inputs: list = [first]
        self.abort_requested = False
        num_layers = self.engine.model.model.layer_major_num_layers()
        self.num_layers = num_layers
        # Grouped attention (QWEN36_LAYER_MAJOR_ATTN_BATCH): only full-size chunks are grouped (the
        # final, shorter chunk keeps its own call); the executor still checks bit-exactness.
        full = max(extends)
        self.order = list(task_order(self.windows, num_layers, attn_layers, attn_k,
                                     groupable=lambda c: extends[c] == full))
        self.cursor = 0
        # cursor value at which window w is complete; its chunks' metadata (attention indices
        # grow with the prefix: ~K^2/2 x chunk x 4 B per window) is dropped then.
        self._window_done = []
        for w in range(len(self.windows)):
            self._window_done.append(max(i for i, t in enumerate(self.order) if t[0] == w) + 1)
        self._released = 0
        self.step_tasks = max(1, _int_env("QWEN36_LAYER_MAJOR_STEP_TASKS", max(1, num_layers // 2)))
        self.max_inflight = max(1, _int_env("QWEN36_LAYER_MAJOR_INFLIGHT_STEPS", 2))
        self._events: collections.deque = collections.deque()
        first.batch.input_ids = sched.token_pool[first.input_tuple]
        from freetoken.engine.telemetry import ENABLED, batch_started

        if ENABLED:
            batch_started(first.batch, self.engine)
        self.executor = LayerMajorExecutor(self.engine, plan, max(extends), bank_ids=bank_ids, refill=refill,
                                           attn_k=attn_k, attn_layers=list(attn_layers))
        if self.executor.attn_k == 1 and attn_k > 1:  # no scratch / no grouped layers: k=1 order
            self.order = list(task_order(self.windows, num_layers))
            self._window_done = [max(i for i, t in enumerate(self.order) if t[0] == w) + 1
                                 for w in range(len(self.windows))]
        pm.layer_major_job = self
        # Refill (QWEN36_LAYER_MAJOR_REFILL=1): cursor at which each borrowed slot group is no
        # longer read by the store (None = only after the final norm).
        self.refill = None
        store = self.executor.store
        cache = self.engine.moe_offload_cache
        if store.piece_group is not None and cache is not None and cache.refill is not None:
            self.refill = cache.refill
            self._refill_free_at = self._group_free_points(store.piece_group, len(self.refill.groups), num_layers)
            for g in store.scratch_groups:  # the grouped-attention scratch lives until the end
                self._refill_free_at[g] = None
        logger.info_rank0(
            f"layer-major prefill uid={self.uid}: {len(extends)} chunks, {plan.num_windows} window(s) "
            f"of <= {plan.window_chunks}, borrowed {plan.borrow_slots} MoE slots"
            + (f", refill in {len(self.refill.groups)} groups of {self.refill.G}" if self.refill else "")
            + (f", attention in groups of {self.executor.attn_k}" if self.executor.attn_k > 1 else "")
        )

    def _group_free_points(self, piece_group, n_groups: int, num_layers: int) -> list:
        """Per slot group: the task cursor after which no task reads its pieces (the last-layer
        task of the last chunk using them, in the last window holding that local index), or
        None if the final chunk uses it (freed after the final norm). Unused groups: 0."""
        last_task = {k: idx for idx, (_, layer, ks) in enumerate(self.order) if layer == num_layers - 1 for k in ks}
        final = len(self.extends) - 1
        free_at = [0] * n_groups
        for p, g in enumerate(piece_group):
            c = p // 2  # pieces 2c / 2c+1 = hidden / residual of window-local chunk c
            k = next((a + c for a, b in reversed(self.windows) if a + c < b), None)
            if k is None or free_at[g] is None:
                continue
            free_at[g] = None if k == final else max(free_at[g], last_task[k] + 1)
        return free_at

    def _refill_tick(self) -> None:
        """After each last-layer task: free the groups no task reads any more and issue a few
        (QWEN36_LAYER_MAJOR_REFILL_TASK_SLOTS, default 16 slots ~ 0.7 ms of host enqueue work, below
        one task's GPU time, so the host never starves the GPU; ~11 slots free up per task). The
        copies run on the otherwise idle copy engine, behind the last layer's compute."""
        if self.refill is None:
            return
        self.refill.mark_free([g for g, t in enumerate(self._refill_free_at) if t is not None and t <= self.cursor])
        self.refill.tick(budget=max(1, _int_env("QWEN36_LAYER_MAJOR_REFILL_TASK_SLOTS", 16)))

    # -- start ------------------------------------------------------------------------------
    @classmethod
    def maybe_start(cls, sched: "Scheduler", fi: "ForwardInput") -> "LayerMajorJob | None":
        reason = scheduler_reason(sched, fi)
        if reason is not None:
            if reason not in _QUIET and reason not in _LOGGED:
                _LOGGED.add(reason)  # once per reason: why a long prompt stayed chunked
                logger.info_rank0(f"layer-major prefill not used: {reason}")
            return None
        from freetoken.engine.layer_major import attn_batch, attn_group_layers, attn_scratch_bytes, store_banks
        from freetoken.moe.layer_major import bank_row_bytes, lendable_slots, plan_store

        req = fi.batch.reqs[0]
        extends = [req.extend_len] + plan_chunk_extends(
            req.device_len, sched.prefill_manager.pending_list[0].input_len,
            sched.prefill_budget, sched.cache_manager.prefill_chunk_align)
        if len(extends) < max(2, _int_env("QWEN36_LAYER_MAJOR_MIN_CHUNKS", 2)):
            return None
        engine = sched.engine
        cache = engine.moe_offload_cache
        inner = engine.model.model
        weight = inner.embed_tokens.weight
        bank_ids, refill = store_banks(cache)
        rows = None
        if cache is not None:
            rows = bank_row_bytes(cache)
            rows = rows if bank_ids is None else [rows[b] for b in bank_ids]
        args = (lendable_slots(cache) if cache is not None else 0, len(extends), max(extends),
                weight.shape[-1], weight.element_size(), _int_env("QWEN36_LAYER_MAJOR_WINDOW_CHUNKS", 0))
        plan = plan_store(rows, *args, grouped=refill)
        if plan is None:
            logger.warning_rank0("layer-major prefill: no lendable slot room for one chunk; using chunked prefill")
            return None
        k, layers = attn_batch(), []
        if k > 1:
            layers = attn_group_layers(inner, engine)
            # Grouped attention (QWEN36_LAYER_MAJOR_ATTN_BATCH): the largest k <= the requested one
            # whose Q/O/gate scratch fits in the borrowed slots WITHOUT more windows (an extra
            # window re-copies every expert layer, far more than grouping saves); else k=1.
            for kk in range(k, 1, -1):
                scratch = attn_scratch_bytes(inner, layers, kk, max(extends), weight.element_size())
                grouped_plan = plan_store(rows, *args, scratch_bytes=scratch, grouped=refill) if scratch else None
                if grouped_plan is not None and grouped_plan.num_windows == plan.num_windows:
                    plan, k = grouped_plan, kk
                    break
            else:
                why = ("grouped attention not used: no groupable full-attention layer / backend" if not layers else
                       "grouped attention not used: its Q/O/gate scratch would cost an extra window")
                if why not in _LOGGED:
                    _LOGGED.add(why)
                    logger.info_rank0(f"layer-major prefill: {why}")
                k, layers = 1, []
        return cls(sched, fi, extends, plan, bank_ids=bank_ids, refill=refill, attn_k=k, attn_layers=layers)

    # -- steps ------------------------------------------------------------------------------
    def step(self) -> "ForwardData | None":
        """Enqueue the next tasks; returns the final (ForwardInput, ForwardOutput) when done."""
        if self.abort_requested:
            self._cancel()
            return None
        while len(self._events) >= self.max_inflight:
            self._events.popleft().synchronize()
        n = 0
        final = len(self.extends) - 1
        while n < self.step_tasks and self.cursor < len(self.order):
            w, layer, ks = self.order[self.cursor]
            for k in ks:
                if k == len(self.inputs):
                    self._prepare_next(k)
            a = self.windows[w][0]
            if len(ks) == 1:
                self.executor.run_task(layer, ks[0] - a, self.inputs[ks[0]].batch)
            else:
                self.executor.run_attn_group(layer, [k - a for k in ks], [self.inputs[k].batch for k in ks], ks)
            self.cursor += 1
            n += len(ks)
            if self.refill is not None:
                if layer == self.num_layers - 2 and final in ks:
                    self.refill.capture_routes()  # final chunk routed through layers 0..L-2
                elif layer == self.num_layers - 1 and self.cursor < len(self.order):
                    self._refill_tick()
        while self._released < len(self.windows) - 1 and self.cursor >= self._window_done[self._released]:
            a, b = self.windows[self._released]
            self.executor.drop_group_metadata(a, b)
            for fi in self.inputs[a:b]:  # stream-ordered frees: later allocations reuse safely
                batch = fi.batch
                batch.attn_metadata = batch.fla_metadata = None
                batch.input_ids = batch.positions = batch.out_loc = None
            self._released += 1
        if self.executor.cuda:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.executor.device))
            self._events.append(event)
        if self.cursor == len(self.order):
            return self._finish()
        return None

    def _prepare_next(self, k: int) -> None:
        sched = self.sched
        prev = self.inputs[-1].batch.reqs[0]
        prev.complete_one()  # what Engine.forward_batch does once chunk k-1 is enqueued
        batch = sched.prefill_manager.schedule_continuation(self.pending, sched.prefill_budget)
        assert batch is not None and batch.reqs[0].extend_len == self.extends[k], (
            "layer-major continuation diverged from the planned chunk boundaries")
        fi = sched._prepare_batch(batch)
        sched._report_prompt_admissions(batch)  # continuations carry none; kept for symmetry
        batch.input_ids = sched.token_pool[fi.input_tuple]
        self.inputs.append(fi)
        if self.refill is not None and k == len(self.extends) - 1:
            self.engine.moe_offload_cache.refill_route_batch = batch  # count the final chunk's routing

    def _finish(self) -> "ForwardData":
        sched = self.sched
        batches = [fi.batch for fi in self.inputs]
        out = self.executor.finish(batches, [fi.sample_args for fi in self.inputs])
        # close first: its compute-stream wait on the copy stream must cover prefill copies only,
        # not the refill copies below (those target RESERVED slots nothing reads until publish)
        self.executor.close()
        if self.refill is not None:
            # The final norm is enqueued: the store is dead. Issue the remaining groups now (a few
            # slots: everything else was refilled behind the last layer); decode publishes them.
            self.refill.mark_all_free()
            self.refill.tick(budget=None)
        last = self.inputs[-1]
        sched.token_pool[last.write_tuple] = out.next_tokens_gpu
        sched.decode_manager.filter_reqs(last.batch.reqs)
        last.batch.log_new_tokens = sum(b.log_new_tokens for b in batches)
        last.batch.log_cached_tokens = batches[0].log_cached_tokens
        if self.executor.begin is not None and self.executor.end is not None:
            last.batch.qwen36_timing = (self.executor.begin, self.executor.end)
        self._detach(completed=True)
        return (last, out)

    def _cancel(self) -> None:
        """Abort: drain, free pages + GDN slots + table row, insert nothing into the prefix cache."""
        sched = self.sched
        self.executor.close(drain=True)
        if self.refill is not None:
            # Drained: no task reads the store any more. Restore the borrowed slots paced, like
            # the tail of a finished prefill (later forwards keep ticking it).
            from freetoken.moe.slot_refill import slots_per_step

            self.refill.mark_all_free()
            self.refill.tick(budget=slots_per_step(), max_inflight=2)
        req = self.inputs[-1].batch.reqs[0]
        req.aborted = True
        from freetoken.engine.telemetry import complete

        complete(req, self.engine, cache_manager=sched.cache_manager)
        if req.table_idx != -1:
            cm = sched.cache_manager
            handle = req.cache_handle
            end = div_ceil(req.device_len, cm.page_size) * cm.page_size
            own = cm.page_table[req.table_idx, handle.cached_len:end]
            cm.unlock(handle)
            cm._free(own)
            if cm.is_hybrid:
                cm._free_req_slots(req)
            sched.table_manager.free(req.table_idx)
            req.table_idx = -1
        self._detach(completed=False)
        logger.info_rank0(f"layer-major prefill uid={self.uid} aborted at task {self.cursor}/{len(self.order)}")

    def _detach(self, *, completed: bool) -> None:
        self.sched.prefill_manager.layer_major_job = None
        if self.engine.moe_offload_cache is not None:
            self.engine.moe_offload_cache.refill_route_batch = None
        stats = getattr(self.engine, "layer_major_stats", None)
        if stats is None:
            stats = self.engine.layer_major_stats = {"prefills": 0, "aborted": 0}
        stats["prefills" if completed else "aborted"] += 1
        stats.update(last_chunks=len(self.extends), last_windows=self.plan.num_windows,
                     last_window_chunks=self.plan.window_chunks, last_borrowed_slots=self.plan.borrow_slots)
        if self.attn_requested > 1:  # QWEN36_LAYER_MAJOR_ATTN_BATCH set: k used (1 = no room)
            stats["last_attn_batch"] = self.executor.attn_k
            for key, value in self.executor.attn_stats.items():
                stats[key] = stats.get(key, 0) + value


__all__ = ["LayerMajorJob", "plan_chunk_extends", "scheduler_reason", "task_order"]
