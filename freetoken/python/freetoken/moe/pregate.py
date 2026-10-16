"""D3b: one-layer-ahead speculative expert prefetch for single-token GPU decode (default off).

``QWEN36_DECODE_PREGATE=<n>`` caps the speculative misses per linear-attention (GDN) layer,
``QWEN36_DECODE_PREGATE_ATTN=<n>`` per full-attention layer (unset: the same n); 0/unset = off.

At the start of decoder layer L+1 (after its input norm, before its mixer) layer L+1's router is
applied to post_attention_layernorm(L+1) of the residual stream that leaves layer L -- the D3a
predictor (coverage 0.75 under exact LRU replay). Its top-k, in score order, is checked against the
slot map and the first n predicted misses go through the unchanged flashlib ``lru_ensure`` into
the shared copy plan; the unchanged gather then copies them from the host banks. All of it runs on
a side stream (T3's when present; its own at the highest priority with
``QWEN36_DECODE_PREGATE_PRIORITY=1``) while the compute stream runs the mixer:

  compute: input_norm(L+1) -> fork -> mixer(L+1) -> wait(read) -> post_norm(L+1) -> router ->
           top-k -> wait(done) -> ensure_experts(L+1) -> copy_missing / T3 -> routed GEMM
  side:    wait(fork) -> logits/select -> read -> lru_ensure(predicted misses) -> gather -> done

The predictor reads the residual tensor, which post_norm(L+1) rewrites in place: ``read`` orders
that. The speculative plan lives in the cache's shared plan buffers (evict_slots / src_indices /
num_indices) and writes slot maps and slot bytes: ``done`` is joined before the real
``ensure_experts`` (and before any other cache operation that touches them). Between fork and that
join no compute-stream work reads or writes slot maps, plan buffers or slot bytes: layer L's GEMM
is ordered before the fork, layer L+1's after the join.

The real ``ensure_experts`` still runs on the real routing, so every routed expert is resident with
host-bank bytes before the routed GEMM; the decode GEMMs index slots per (token, k) and sum top-k
in k order, so which slot an expert occupies does not reach the arithmetic. A wrong prediction
costs PCIe traffic and residency only. Speculative admissions are not counted in ``lru_stats``
(those count the real misses that remain on the critical path); they are counted separately in
``DecodePregate.rows`` when the cache collects stats.

LRU state does differ from the reference: each speculative call advances the LRU clock once,
installs up to n predicted experts (victims: least recent slots, as for a real miss) and, only when
no predicted expert misses, bumps the top-1 predicted expert's usage (the fixed-shape query is
padded with it). Hence the teacher gate.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

ENV = "QWEN36_DECODE_PREGATE"
ATTN_ENV = "QWEN36_DECODE_PREGATE_ATTN"
# Own side stream at the highest priority instead of T3's: CTAs of equal-priority grids are dispatched
# in launch order, so without it the speculative chain waits behind the mixer kernel's last wave.
PRIORITY_ENV = "QWEN36_DECODE_PREGATE_PRIORITY"
MAX_CAP = 8


def _cap(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    raw = raw.strip()
    if not raw.isdigit() or int(raw) > MAX_CAP:
        raise ValueError(f"{name}={raw!r}: expected unset or an integer 0..{MAX_CAP}")
    return int(raw)


def priority() -> bool:
    raw = os.environ.get(PRIORITY_ENV, "0")
    if raw not in ("", "0", "1"):
        raise ValueError(f"{PRIORITY_ENV}={raw!r}: expected unset, 0 or 1")
    return raw == "1"


def caps() -> tuple[int, int]:
    """(linear-attention cap, full-attention cap); (0, 0) = off."""
    linear = _cap(ENV, 0)
    return linear, _cap(ATTN_ENV, linear)


@triton.jit
def _pregate_logits_kernel(res_ptr, norm_w_ptr, gate_w_ptr, logits_ptr, eps, stride_e,
                           H: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_E: tl.constexpr):
    # Speculative only: any rounding difference from the real norm/router moves traffic, not bytes.
    h = tl.arange(0, BLOCK_H)
    hm = h < H
    x = tl.load(res_ptr + h, mask=hm, other=0.0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / H + eps)
    w = tl.load(norm_w_ptr + h, mask=hm, other=0.0).to(tl.float32)
    y = (x * inv * w).to(tl.bfloat16).to(tl.float32)
    e = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    g = tl.load(gate_w_ptr + e[:, None].to(tl.int64) * stride_e + h[None, :], mask=hm[None, :], other=0.0)
    tl.store(logits_ptr + e, tl.sum(g.to(tl.float32) * y[None, :], axis=1))


@triton.jit
def _pregate_select_kernel(logits_ptr, slot_row_ptr, query_ptr, cap, E,
                           TOP_K: tl.constexpr, Q: tl.constexpr, BLOCK: tl.constexpr):
    """query[:Q] <- the first ``cap`` predicted misses in score order (ties: lower id), padded with
    the first of them, or with the top-1 prediction when none misses. Ids stay in [0, E)."""
    n = tl.arange(0, BLOCK)
    nm = n < E
    v = tl.load(logits_ptr + n, mask=nm, other=0.0)
    v = tl.where(nm & (v == v), v, -float("inf"))
    slot = tl.load(slot_row_ptr + n, mask=nm, other=0)
    count = tl.full((), 0, tl.int32)
    first = tl.full((), 0, tl.int32)
    top1 = tl.full((), 0, tl.int32)
    for k in tl.static_range(TOP_K):
        best = tl.max(v, axis=0)
        win = tl.min(tl.where(nm & (v == best), n, BLOCK), axis=0).to(tl.int32)
        win = tl.minimum(win, E - 1)
        if k == 0:
            top1 = win
        miss = (tl.sum(tl.where(n == win, slot, 0), axis=0) < 0) & (count < cap)
        tl.store(query_ptr + count, win, mask=miss)
        first = tl.where(miss & (count == 0), win, first)
        count += miss.to(tl.int32)
        v = tl.where(n == win, -float("inf"), v)
    pad = tl.where(count > 0, first, top1)
    for p in tl.static_range(Q):
        tl.store(query_ptr + p, pad, mask=p >= count)


class DecodePregate:
    """Owns the side-stream fork/join of the speculative plan; one instance per cache."""

    def __init__(self, cache, cuda, device, linear_cap: int, attn_cap: int, stream=None, high_priority=False):
        self.cache = cache
        self.cuda = cuda
        self.device = device
        self.linear_cap = linear_cap
        self.attn_cap = attn_cap
        # fixed query width: duplicates collapse in lru_ensure, so padding never adds a copy
        self.width = max(2, linear_cap, attn_cap)
        if high_priority:
            self.stream = cuda.Stream(device=device, priority=-1)
        else:
            self.stream = stream if stream is not None else cuda.Stream(device=device)
        self.fork = cuda.Event()
        self.read = cuda.Event()
        self.done = cuda.Event()
        E = cache.num_experts
        self.logits = torch.zeros((E,), dtype=torch.float32, device=device)
        self.query = torch.zeros((self.width,), dtype=torch.int32, device=device)
        self.slots = torch.zeros((self.width,), dtype=torch.int32, device=device)
        self.rows = torch.zeros((cache.num_layers,), dtype=torch.int64, device=device)
        self._pending = False
        self._reading = False
        # Materialize CUDA handles before capture; construction follows cache init.
        producer = cuda.current_stream(device)
        for event in (self.fork, self.read, self.done):
            event.record(producer)
            self.stream.wait_event(event)
        self.done.record(self.stream)
        producer.wait_event(self.done)

    def cap(self, attention: bool) -> int:
        return self.attn_cap if attention else self.linear_cap

    @staticmethod
    def predictor_ok(residual, norm, gate, num_experts: int) -> bool:
        weight = getattr(gate, "weight", None)
        nw = getattr(norm, "weight", None)
        H = residual.shape[-1]
        return (weight is not None and nw is not None and residual.dim() == 2 and residual.stride(-1) == 1
                and weight.dim() == 2 and tuple(weight.shape) == (num_experts, H) and weight.stride(-1) == 1
                and weight.dtype in (torch.bfloat16, torch.float16, torch.float32)
                and nw.numel() == H and nw.is_contiguous() and num_experts % 8 == 0)

    def launch(self, layer_id: int, residual: torch.Tensor, norm, gate, cap: int, top_k: int) -> None:
        """Fork the speculative plan of ``layer_id`` off the compute stream (see module doc)."""
        producer = self.cuda.current_stream(self.device)
        self.join()
        self.fork.record(producer)
        self._pending = self._reading = True
        try:
            with self.cuda.stream(self.stream):
                self.stream.wait_event(self.fork)
                try:
                    try:
                        self._predict(layer_id, residual, norm, gate, cap, top_k)
                    finally:
                        self.read.record(self.stream)
                    self._admit_and_copy(layer_id)
                finally:
                    self.done.record(self.stream)
        except BaseException:
            self.join()
            raise

    def release_input(self) -> None:
        """Before the compute stream rewrites the residual the predictor read."""
        if self._reading:
            self.cuda.current_stream(self.device).wait_event(self.read)
            self._reading = False

    def join(self) -> None:
        """Before anything reuses the shared plan buffers, slot maps or slot bytes."""
        if self._pending:
            self.cuda.current_stream(self.device).wait_event(self.done)
            self._pending = self._reading = False

    def _predict(self, layer_id, residual, norm, gate, cap, top_k):
        cache = self.cache
        E = cache.num_experts
        H = residual.shape[-1]
        weight = gate.weight
        _pregate_logits_kernel[(E // 8,)](
            residual, norm.weight, weight, self.logits, float(norm.eps), weight.stride(0),
            H=H, BLOCK_H=triton.next_power_of_2(H), BLOCK_E=8, num_warps=8,
        )
        _pregate_select_kernel[(1,)](
            self.logits, cache.slot_for_id[layer_id], self.query, cap, E,
            TOP_K=top_k, Q=self.width, BLOCK=triton.next_power_of_2(E), num_warps=1,
        )

    def _admit_and_copy(self, layer_id):
        from flashlib.kernels.slot_cache import lru_ensure

        cache = self.cache
        lru_ensure(
            self.query, cache.slot_for_id.view(-1), cache.id_of_slot, cache.usage, cache.step,
            self.slots, cache.src_indices, cache.evict_slots, cache.num_indices,
            stats=None, id_base=layer_id * cache.num_experts,
        )
        cache.copy_planned(layer_id)
        if cache.collect_stats:
            self.rows[layer_id] += cache.num_indices[0]

    def reset_stats(self) -> None:
        self.rows.zero_()
