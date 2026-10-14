"""Hidden/residual store for layer-major prefill, in VRAM borrowed from the MoE slot cache.

Layer-major prefill keeps hidden + residual of every chunk of the prompt between layers:
2 x tokens x hidden x 2 B (bf16) = 8 KiB/token for Qwen3.6 (0.98 GB at 120K, 2.05 GB at 250K).
During a single-request prefill the decode residency region of the slot cache (slots >= 2E;
the prefill double buffers own slots < 2E, hit-D2D is off) is idle, so its tail is lent to the
store: the borrowed slots are unmapped first (slot_for_id -1, id_of_slot -1, usage 0) on the
compute stream, decode refetches those experts on demand afterwards. Every registered bank of
the slot cache contributes its bytes (gate_up 1 MiB/slot, down 512 KiB, scales 192 KiB, globals
12 KiB), carved into fixed pieces of ``chunk_rows x hidden`` elements (one chunk tensor each).

When the whole prompt does not fit, the prompt is processed in windows of chunks (layer-major
inside a window, windows one after another); chunk boundaries and kernels are unchanged, so the
result stays bitwise the chunked one. This module is pure host planning + tensor views; it never
synchronizes the device.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

ALIGN = 256  # every piece starts 256-byte aligned (vectorized kernels need >= 16 B)


def bank_row_bytes(cache) -> list[int]:
    """Bytes per slot of every registered bank, in registration order."""
    return [math.prod(c.shape[1:]) * c.element_size() for _, c in cache.banks]


def lendable_slots(cache) -> int:
    """Slots that may be borrowed: the decode residency region (>= 2E) of an overlap cache."""
    return max(0, cache.cache_size - 2 * cache.num_experts)


def piece_capacity(row_bytes: list[int], n_slots: int, piece_bytes: int) -> int:
    """Number of whole ``piece_bytes`` pieces the tail ``n_slots`` of every bank can hold."""
    if n_slots <= 0:
        return 0
    total = 0
    for rb in row_bytes:
        if rb % ALIGN:  # region starts would not stay aligned; such banks are not lent
            continue
        total += (n_slots * rb) // piece_bytes
    return total


def slots_for_pieces(row_bytes: list[int], lendable: int, pieces: int, piece_bytes: int) -> int | None:
    """Smallest tail slot count holding ``pieces`` pieces, or None if the lendable region can't."""
    if piece_capacity(row_bytes, lendable, piece_bytes) < pieces:
        return None
    lo, hi = 1, lendable
    while lo < hi:
        mid = (lo + hi) // 2
        if piece_capacity(row_bytes, mid, piece_bytes) >= pieces:
            hi = mid
        else:
            lo = mid + 1
    return lo


@dataclass(frozen=True)
class StorePlan:
    num_chunks: int
    window_chunks: int     # chunks per window (== num_chunks when the prompt fits at once)
    num_windows: int
    borrow_slots: int      # 0 for a plain (non-borrowed) allocation
    piece_bytes: int
    scratch_bytes: int = 0  # one contiguous block next to the pieces (grouped attention buffers)
    grouped: bool = False   # refill mode: pieces carved slot group by slot group

    def windows(self) -> list[tuple[int, int]]:
        """[start, end) chunk index ranges, balanced, in order."""
        out, start = [], 0
        for w in range(self.num_windows):
            size = self.window_chunks if w < self.num_windows - 1 else self.num_chunks - start
            out.append((start, start + size))
            start += size
        assert start == self.num_chunks
        return out


def group_slots_for(row_bytes: list[int], piece_bytes: int) -> int:
    """Smallest slot group in which every lent bank holds a whole number of pieces (refill mode
    carves the store group by group, so each group frees as soon as its chunks are done)."""
    g = 1
    for rb in row_bytes:
        g = math.lcm(g, piece_bytes // math.gcd(piece_bytes, rb))
    return g


def store_layout(row_bytes: list[int], n: int, piece: int, group: int | None, scratch_bytes: int = 0,
                 limit: int | None = None):
    """Carve order of the borrowed tail of ``n`` slots: ``pieces`` = [(bank, byte offset, group)]
    and ``scratch`` = (bank, byte offset) of one contiguous ``scratch_bytes`` block, or None.

    Banks whose rows are not 256-byte multiples are not lent. The scratch block sits at the END
    of the bank with the largest rows (fewest slots; in refill mode those slots belong to the
    last groups, which free at the end anyway) and pieces never overlap it. ``group`` None = one
    group (bank after bank, the layout without refill); otherwise slot groups of ``group`` slots
    with the banks inside each group. Returns ([], None) when the scratch block does not fit."""
    if n <= 0:
        return [], None
    lent = [b for b, rb in enumerate(row_bytes) if rb % ALIGN == 0]
    cut = {b: n * row_bytes[b] for b in lent}
    scratch = None
    if scratch_bytes:
        if not lent:
            return [], None
        sb = max(lent, key=lambda b: (row_bytes[b], -b))
        if cut[sb] < scratch_bytes:
            return [], None
        off = (cut[sb] - scratch_bytes) // ALIGN * ALIGN
        scratch, cut[sb] = (sb, off), off
    G = group or n
    pieces = []
    for g, s0 in enumerate(range(0, n, G)):
        s1 = min(s0 + G, n)
        for b in lent:
            rb = row_bytes[b]
            off, hi = s0 * rb, min(s1 * rb, cut[b])
            while off + piece <= hi:
                if limit is not None and len(pieces) == limit:
                    return pieces, scratch
                pieces.append((b, off, g))
                off += piece
    return pieces, scratch


def plan_store(
    row_bytes: list[int] | None,
    lendable: int,
    num_chunks: int,
    chunk_rows: int,
    hidden: int,
    elem_bytes: int,
    max_window_chunks: int = 0,
    scratch_bytes: int = 0,
    grouped: bool = False,
) -> StorePlan | None:
    """Window/borrow plan, or None if not even one chunk fits. ``row_bytes`` None = plain alloc.
    ``scratch_bytes``: also hold one contiguous scratch block (rounded up to 256 B);
    ``grouped``: refill-mode layout (slot groups)."""
    piece = chunk_rows * hidden * elem_bytes
    piece = -(-piece // ALIGN) * ALIGN
    scratch_bytes = -(-scratch_bytes // ALIGN) * ALIGN
    if row_bytes is None:
        w = num_chunks if max_window_chunks <= 0 else min(num_chunks, max_window_chunks)
        nw = -(-num_chunks // w)
        w = -(-num_chunks // nw)
        return StorePlan(num_chunks, w, nw, 0, piece, scratch_bytes, False)
    G = group_slots_for(row_bytes, piece) if grouped else None

    def capacity(n: int) -> int:  # non-decreasing in n
        return len(store_layout(row_bytes, n, piece, G, scratch_bytes)[0])

    w_max = capacity(lendable) // 2
    if max_window_chunks > 0:
        w_max = min(w_max, max_window_chunks)
    if w_max < 1:
        return None
    nw = -(-num_chunks // w_max)
    w = -(-num_chunks // nw)  # balanced windows borrow fewer slots than one full + one small
    lo, hi = 1, lendable
    while lo < hi:
        mid = (lo + hi) // 2
        if capacity(mid) >= 2 * w:
            hi = mid
        else:
            lo = mid + 1
    assert capacity(lo) >= 2 * w
    return StorePlan(num_chunks, w, nw, lo, piece, scratch_bytes, grouped)



class HiddenStore:
    """``2 x window_chunks`` chunk tensors ``[chunk_rows, hidden]``: hidden(i) and residual(i)
    for window-local chunk i, plus an optional contiguous ``scratch`` byte block (grouped
    attention buffers). Views into borrowed slot-cache bytes (or plain allocations when there is
    no offload cache, e.g. CPU tests).

    ``bank_ids``: the banks to borrow from (default all). ``refill``: carve slot group by slot
    group (``piece_group[p]`` = group of piece p, ``group_slots`` slots each, in chunk order;
    ``scratch_groups`` = the groups whose slots hold the scratch block) and let the cache restore
    the borrowed slots as groups free up (freetoken.moe.slot_refill)."""

    def __init__(self, cache, plan: StorePlan, chunk_rows: int, hidden: int, dtype: torch.dtype,
                 device: torch.device, bank_ids=None, refill: bool = False):
        self.rows, self.hidden, self.dtype = chunk_rows, hidden, dtype
        n_pieces = 2 * plan.window_chunks
        elem = torch.empty((), dtype=dtype).element_size()
        self.piece_group = None
        self.group_slots = 0
        self.scratch_groups: set[int] = set()
        self.scratch = None
        if plan.borrow_slots == 0:
            self._backing = torch.empty((n_pieces, plan.piece_bytes // elem), dtype=dtype, device=device)
            pieces = [self._backing[i] for i in range(n_pieces)]
            if plan.scratch_bytes:
                self.scratch = torch.empty(plan.scratch_bytes, dtype=torch.uint8, device=device)
        else:
            sel = list(range(len(cache.banks))) if bank_ids is None else list(bank_ids)
            rows = [bank_row_bytes(cache)[b] for b in sel]
            n = plan.borrow_slots
            G = group_slots_for(rows, plan.piece_bytes) if refill else None
            regions = cache.borrow_tail_slots(n, bank_ids=sel, refill_group=G)
            for region, rb in zip(regions, rows):
                # CUDA allocations are >= 512 B aligned and lent rows are 256 B multiples, so every
                # lent region starts 256 B aligned (CPU tensors: 64 B; nothing there needs more).
                assert not (rb % ALIGN == 0 and region.is_cuda and region.data_ptr() % ALIGN)
            layout, scratch = store_layout(rows, n, plan.piece_bytes, G, plan.scratch_bytes, limit=n_pieces)
            assert len(layout) == n_pieces, (len(layout), n_pieces)
            pieces = [regions[b][off: off + plan.piece_bytes].view(dtype) for b, off, _ in layout]
            if scratch is not None:
                sb, soff = scratch
                self.scratch = regions[sb][soff: soff + plan.scratch_bytes]
            if refill:
                self.piece_group, self.group_slots = [g for _, _, g in layout], G
                if scratch is not None:
                    sb, soff = scratch
                    first = soff // rows[sb] // G
                    last = (soff + plan.scratch_bytes - 1) // rows[sb] // G
                    self.scratch_groups = set(range(first, last + 1))
            self._backing = None
        need = chunk_rows * hidden
        self._hid = [p[:need].view(chunk_rows, hidden) for p in pieces[0::2]]
        self._res = [p[:need].view(chunk_rows, hidden) for p in pieces[1::2]]
        self.cache = cache
        self.borrowed = plan.borrow_slots

    def hidden_view(self, i: int, rows: int) -> torch.Tensor:
        return self._hid[i][:rows]

    def residual_view(self, i: int, rows: int) -> torch.Tensor:
        return self._res[i][:rows]

    def release(self) -> None:
        # Borrowed slots stay unmapped (refilled, or refetched by decode); nothing to copy back.
        self._hid = self._res = []
        self._backing = None
        self.scratch = None


__all__ = [
    "HiddenStore",
    "StorePlan",
    "bank_row_bytes",
    "group_slots_for",
    "lendable_slots",
    "piece_capacity",
    "plan_store",
    "slots_for_pieces",
    "store_layout",
]
