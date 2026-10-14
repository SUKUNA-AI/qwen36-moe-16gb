"""Background refill of the MoE slots a layer-major prefill borrowed (QWEN36_LAYER_MAJOR_REFILL=1).

Layer-major prefill lends the tail of the slot cache (slots >= 2E, the decode residency region)
to its hidden/residual store. Without a refill those slots come back empty and the first decode
tokens re-miss. In refill mode the store borrows only the big banks (rows >= 256 KiB: gate_up
1 MiB, down 512 KiB) and the slots are refilled group by group as the store frees them (layer 39
of the last window), i.e. behind the prefill's own compute, on the copy engine.

Policies (QWEN36_LAYER_MAJOR_REFILL_POLICY):
  * ``last_chunk`` (default): the most frequently routed (layer, expert) of the prompt's final
    chunk in layers 0..L-2 that are not resident, with usage = the LRU clock ("hot"). Copies every
    bank of the new expert. Chosen by a CPU simulation (exact flashlib LRU, synthetic routing
    calibrated to the measured decode miss rates): fewest decode misses at every length.
  * ``restore``: exactly the experts, LRU usage and slot positions held before the borrow (only
    big rows are copied: the small rows were never lent). The decode then sees bit for bit the
    slot cache a chunked prefill would have left.
  ``last_chunk`` falls back to ``restore`` when the routes were not captured (abort before the
  final chunk reached layer L-2).

Ownership (as in the prefill double buffer):
  * borrow -> publish: the slots are unmapped (no slot_for_id points at them, id_of_slot = -1) and
    carry usage = RESERVED, which the device LRU (flashlib lru_ensure: victims in ascending
    (usage, slot)) never selects while any ordinary slot exists: a decode miss can never be
    installed into a slot a refill copy targets, and nothing reads such a slot.
  * group free: the store's last read of every piece in a slot group is enqueued; an event there
    gates the group's copies on the copy stream (cudaMemcpyAsync rows, or one
    cudaMemcpyBatchAsync of big rows when that entry point is already loaded), so a copy never
    overwrites live store bytes. Slots of the group without an assignment are released (usage 0).
  * publish: only after the group's copy event has completed (host event.query(), never a host
    sync) the compute stream waits on it (complete: no stall) and maps the group in one vectorized
    step; an expert that a decode miss already brought back elsewhere is NOT mapped again (its
    slot is released), so slot_for_id / id_of_slot stay a bijection.
  * stop (a new layer-major borrow): in-flight copies are waited for on the compute stream
    (stream order, bounded by the in-flight groups) and published; unissued groups are released.
    Rebuild / reset drop the refill after the device is synchronized.
Plans are only built once their inputs have landed (event.query()), so nothing here syncs the host.
Everything runs eagerly between forwards, never inside CUDA-graph capture.
"""
from __future__ import annotations

import collections
import os

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

# Far above any real usage (the LRU clock advances once per ensure call) and small enough that
# flashlib's packed (usage << SLOT_BITS | slot) keys cannot overflow for caches < 2**20 slots.
RESERVED_USAGE = 1 << 42
MIN_REFILL_ROW_BYTES = 256 * 1024
_ON = {"1", "true", "yes", "on"}


def refill_enabled() -> bool:
    return os.environ.get("QWEN36_LAYER_MAJOR_REFILL", "0").strip().lower() in _ON


def refill_policy() -> str:
    p = os.environ.get("QWEN36_LAYER_MAJOR_REFILL_POLICY", "last_chunk").strip().lower()
    return p if p in ("last_chunk", "restore") else "last_chunk"


def min_row_bytes() -> int:
    try:
        return int(os.environ.get("QWEN36_LAYER_MAJOR_REFILL_MIN_ROW", MIN_REFILL_ROW_BYTES))
    except ValueError:
        return MIN_REFILL_ROW_BYTES


def slots_per_step() -> int:
    try:
        return max(1, int(os.environ.get("QWEN36_LAYER_MAJOR_REFILL_SLOTS_PER_STEP", "32")))
    except ValueError:
        return 32


class _Ready:
    """A host copy of device data, valid once its event completed (query only, never synchronize)."""

    def __init__(self, cpu_tensors, event):
        self.tensors, self.event = cpu_tensors, event

    def ready(self) -> bool:
        return self.event is None or self.event.query()


def _snap(cache, tensors):
    """Non-blocking D2H of ``tensors`` into pinned host buffers + an event on the compute stream."""
    pin = cache.device.type == "cuda"
    out = []
    for t in tensors:
        h = torch.empty(t.shape, dtype=t.dtype, pin_memory=pin)
        h.copy_(t, non_blocking=pin)
        out.append(h)
    event = None
    if pin:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(cache.device))
    return _Ready(out, event)


class SyncCopier:
    """CPU / no-stream fallback: copies happen at issue time."""

    def issue(self, copies, after=None):
        for dst, src in copies:
            dst.copy_(src)
        return None

    def done(self, handle) -> bool:
        return True

    def wait(self, handle) -> None:
        return None

    def mark(self):
        return None


class CudaCopier:
    """Copy engine on the cache's prefill copy stream."""

    def __init__(self, cache):
        self.cache = cache
        self.stream = cache.prefill_copy_stream

    def mark(self):
        """Event on the compute stream: everything enqueued so far (the store's last reads)."""
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.cache.device))
        return event

    def issue(self, copies, after=None):
        with torch.cuda.stream(self.stream):
            if after is not None:
                self.stream.wait_event(after)
            batch = getattr(self.cache, "_batch_memcpy", None)
            if copies and callable(batch) and isinstance(copies[0][0], int):
                # one cudaMemcpyBatchAsync; only big rows (>= 256 KiB) get here, the size above
                # which the driver keeps a batch asynchronous
                batch(torch.tensor([c[0] for c in copies], dtype=torch.int64),
                      torch.tensor([c[1] for c in copies], dtype=torch.int64),
                      torch.tensor([c[2] for c in copies], dtype=torch.int64), self.stream.cuda_stream)
            else:
                for dst, src in copies:
                    dst.copy_(src, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        return done

    def done(self, handle) -> bool:
        return handle is None or handle.query()

    def wait(self, handle) -> None:
        if handle is not None:
            torch.cuda.current_stream(self.cache.device).wait_event(handle)


class SlotRefill:
    """Refill the borrowed slots [lo, lo + n) of ``cache`` group by group."""

    def __init__(self, cache, lo: int, n: int, group_slots: int, bank_ids, old_ids, old_usage,
                 snapshot_ready=None, copier=None, policy: str | None = None):
        self.cache = cache
        self.lo, self.hi = lo, lo + n
        self.G = max(1, group_slots)
        self.lent_banks = list(bank_ids)
        self.groups = [(s, min(s + self.G, self.hi)) for s in range(lo, self.hi, self.G)]
        self.free = [False] * len(self.groups)    # the store no longer reads the group
        self.issued = [False] * len(self.groups)
        self.closed = [False] * len(self.groups)  # published or released
        self.policy = policy or refill_policy()
        self._snapshot = _Ready([old_ids, old_usage], snapshot_ready)
        self._routes = None   # _Ready([counts, slot_for_id, step]) for last_chunk
        self.route_counts = None
        if self.policy == "last_chunk":
            # filled by OffloadMoELayer._prefill_routed while cache.refill_route_batch (the final
            # chunk, set by the layer-major job) is the active batch
            self.route_counts = torch.zeros((cache.num_layers, cache.num_experts), dtype=torch.int32,
                                            device=cache.device)
        cache.refill_route_counts = self.route_counts
        cache.refill_route_batch = None
        self._assign = None   # per borrowed slot: (flat expert id, usage) or None
        self._copy_banks = None
        self.inflight: collections.deque = collections.deque()
        if copier is None:
            copier = CudaCopier(cache) if (cache.device.type == "cuda" and cache.prefill_copy_stream is not None) else SyncCopier()
        self.copier = copier
        stats = getattr(cache, "layer_major_refill_stats", None)
        if stats is None:
            stats = cache.layer_major_refill_stats = dict(
                refills=0, slots_borrowed=0, slots_copied=0, bytes_copied=0, slots_released_empty=0,
                stopped_groups=0, policy_last_chunk=0, policy_restore=0)
            cache.layer_major_refill_published = torch.zeros((), dtype=torch.int64, device=cache.device)
        stats["refills"] += 1
        stats["slots_borrowed"] += n

    # -- inputs ------------------------------------------------------------------------------
    def capture_routes(self) -> None:
        """Called once the final chunk's routing of layers 0..L-2 is enqueued: snapshot the counts,
        the residency map and the LRU clock (non-blocking; used when it has landed)."""
        if self.route_counts is not None and self._routes is None:
            c = self.cache
            self._routes = _snap(c, [self.route_counts[: c.num_layers - 1].reshape(-1),
                                     c.slot_for_id.view(-1), c.step.reshape(1)])

    def _plan(self, fallback: bool) -> bool:
        """Build the per-slot assignment once its inputs have landed. False = not yet."""
        if self._assign is not None:
            return True
        n = self.hi - self.lo
        if self.policy == "last_chunk" and self._routes is not None:
            if not self._routes.ready():
                return False
            counts, sf, step = (t.tolist() for t in self._routes.tensors)
            hot = int(step[0])
            order = sorted((i for i, c in enumerate(counts) if c > 0 and sf[i] < 0), key=lambda i: (-counts[i], i))
            self._assign = [(e, hot) for e in order[:n]] + [None] * max(0, n - len(order))
            self._copy_banks = list(range(len(self.cache.banks)))  # a new expert: every bank
            self.cache.layer_major_refill_stats["policy_last_chunk"] += 1
        elif self.policy == "restore" or fallback:
            if not self._snapshot.ready():
                return False
            ids, usage = (t.tolist() for t in self._snapshot.tensors)
            self._assign = [(e, u) if e >= 0 else None for e, u in zip(ids, usage)]
            self._copy_banks = self.lent_banks  # small rows were never lent: still the old expert's
            self.cache.layer_major_refill_stats["policy_restore"] += 1
        else:
            return False
        self._snapshot = self._routes = None
        self.route_counts = None
        if getattr(self.cache, "refill_route_counts", None) is not None:
            self.cache.refill_route_counts = None
        return True

    # -- driven by the layer-major job / Engine.forward_batch ---------------------------------
    def mark_free(self, groups) -> None:
        for g in groups:
            self.free[g] = True

    def mark_all_free(self) -> None:
        self.free = [True] * len(self.groups)

    @property
    def finished(self) -> bool:
        return all(self.closed)

    def tick(self, budget: int | None = None, max_inflight: int | None = None) -> None:
        """Publish landed groups, then issue free groups (``budget`` slots per call, None = all)."""
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        self._publish_completed()
        if any(self.free[g] and not self.issued[g] for g in range(len(self.groups))):
            # an abort frees every group before the final chunk's routes exist: restore instead
            if self._plan(fallback=all(self.free) and self._routes is None):
                issued = 0
                for g, (s0, s1) in enumerate(self.groups):
                    if not self.free[g] or self.issued[g]:
                        continue
                    if budget is not None and issued and issued + (s1 - s0) > budget:
                        break
                    if max_inflight is not None and len(self.inflight) >= max_inflight:
                        break
                    self._issue(g)
                    issued += s1 - s0
                self._publish_completed()
        if self.finished and getattr(self.cache, "refill", None) is self:
            self.cache.refill = None

    def stop(self) -> None:
        """Drain: wait (stream order) for in-flight copies and publish them; release the rest."""
        while self.inflight:
            g, handle, plan = self.inflight.popleft()
            self.copier.wait(handle)
            self._publish(plan)
            self.closed[g] = True
        for g, (s0, s1) in enumerate(self.groups):
            if not self.closed[g]:
                self._release(list(range(s0, s1)))
                self.closed[g] = True
                self.cache.layer_major_refill_stats["stopped_groups"] += 1
        self.route_counts = None
        if getattr(self.cache, "refill_route_counts", None) is not None:
            self.cache.refill_route_counts = None
        if getattr(self.cache, "refill", None) is self:
            self.cache.refill = None

    # -- internals ------------------------------------------------------------------------
    def _issue(self, g: int) -> None:
        cache = self.cache
        s0, s1 = self.groups[g]
        E = cache.num_experts
        keep, empty = [], []
        for s in range(s0, s1):
            a = self._assign[s - self.lo]
            if a is None:
                empty.append(s)
            else:
                keep.append((s, a[0], a[1]))
        after = self.copier.mark()
        copies, nbytes = [], 0
        # One cudaMemcpyBatchAsync only for big rows and only if the entry point is already
        # loaded (never JIT-built here, on the prefill/decode path); else a cudaMemcpyAsync per
        # row, still on the copy engine.
        rows = [cache._copy_feat_bytes_host[b] for b in self._copy_banks] if getattr(cache, "_copy_fused_ok", False) else []
        fused = (isinstance(self.copier, CudaCopier) and callable(getattr(cache, "_batch_memcpy", None))
                 and rows and min(rows) >= MIN_REFILL_ROW_BYTES)
        for b in self._copy_banks:
            per_layer, bank = cache.banks[b]
            for s, flat, _ in keep:
                layer, expert = divmod(flat, E)
                if fused:
                    feat = cache._copy_feat_bytes_host[b]
                    copies.append((cache._copy_dst_ptrs_host[b] + s * feat,
                                   cache._copy_src_ptrs_host[layer][b] + expert * feat, feat))
                    nbytes += feat
                else:
                    copies.append((bank[s], per_layer[layer][expert]))
                    nbytes += bank[s].numel() * bank.element_size()
        handle = self.copier.issue(copies, after) if copies else None
        if empty:
            self._release(empty)
            cache.layer_major_refill_stats["slots_released_empty"] += len(empty)
        self.issued[g] = True
        stats = cache.layer_major_refill_stats
        stats["slots_copied"] += len(keep)
        stats["bytes_copied"] += nbytes
        if keep:
            self.inflight.append((g, handle, keep))
        else:
            self.closed[g] = True

    def _publish_completed(self) -> None:
        while self.inflight and self.copier.done(self.inflight[0][1]):
            g, handle, plan = self.inflight.popleft()
            self.copier.wait(handle)  # complete already: formal stream order, no stall
            self._publish(plan)
            self.closed[g] = True

    def _to_device(self, values) -> torch.Tensor:
        # pinned staging: a pageable H2D may synchronize the host with the stream
        pin = self.cache.device.type == "cuda"
        return torch.tensor(values, dtype=torch.int64, pin_memory=pin).to(self.cache.device, non_blocking=pin)

    def _publish(self, plan) -> None:
        """Map refilled slots unless a decode miss already brought the expert back elsewhere."""
        cache = self.cache
        slots = self._to_device([p[0] for p in plan])
        ids = self._to_device([p[1] for p in plan])
        use = self._to_device([p[2] for p in plan])
        sf = cache.slot_for_id.view(-1)
        cur = sf.index_select(0, ids)
        ok = cur < 0
        sf.index_copy_(0, ids, torch.where(ok, slots.to(sf.dtype), cur))
        cache.id_of_slot.index_copy_(0, slots, torch.where(ok, ids.to(cache.id_of_slot.dtype),
                                                           torch.full_like(ids, -1, dtype=cache.id_of_slot.dtype)))
        cache.usage.index_copy_(0, slots, torch.where(ok, use, torch.zeros_like(use)))
        cache.layer_major_refill_published += ok.sum()

    def _release(self, slots) -> None:
        cache = self.cache
        idx = self._to_device(slots)
        cache.id_of_slot.index_fill_(0, idx, -1)
        cache.usage.index_fill_(0, idx, 0)


__all__ = ["RESERVED_USAGE", "SlotRefill", "min_row_bytes", "refill_enabled", "refill_policy", "slots_per_step"]
