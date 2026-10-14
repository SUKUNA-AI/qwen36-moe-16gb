"""Engine side of layer-major prefill: one long prompt, ``for layer: for chunk:``.

Chunked prefill runs the task graph (chunk k, layer L) in k-major order; layer-major runs the
same graph in L-major order (inside windows of chunks when the hidden store does not fit at once).
Task (k, L) depends on (k, L-1) (hidden/residual of the chunk) and on (k-1, L) (KV pages and GDN
conv/recurrent state of layer L written by earlier chunks). Both orders are topological sorts of
this graph and every task runs the same kernels on the same shapes (scheduler chunk boundaries)
and bitwise the same inputs, so logits are bitwise the chunked ones as long as each kernel is
deterministic on its inputs. What changes is data movement: the MoE double buffer receives each
layer's experts once per window instead of once per chunk (copy L+1 hides behind all chunks of
layer L), hidden/residual of the window live in VRAM borrowed from the slot cache.

Specifics handled here:
  * FlashInfer: the prefill wrapper is shared and planned lazily per metadata object
    (``initialized``); when a full-attention task switches chunk the metadata of that chunk is
    re-planned (same plan inputs -> same plan as the chunked run, which plans once per chunk).
  * Only the last chunk computes the final norm + LM head (rows are independent; the chunked run
    computes and discards logits of intermediate chunks). For unseeded non-greedy sampling the
    RNG offset consumed by the discarded intermediate samples is replayed with dummy logits of
    the same shape, so the final token's random stream is the chunked one too.
  * Eager only: decode keeps its CUDA graphs, prefill never used them.
"""
from __future__ import annotations

import os

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_ON = {"1", "true", "yes", "on"}


def enabled() -> bool:
    return os.environ.get("QWEN36_LAYER_MAJOR_PREFILL", "0").strip().lower() in _ON


def min_tokens() -> int:
    """QWEN36_LAYER_MAJOR_MIN_TOKENS=<T> (unset/0 = current behaviour): start layer-major only when
    the request's uncached prompt tokens (prompt minus prefix-cache hit) are >= T, on top of the
    >= QWEN36_LAYER_MAJOR_MIN_CHUNKS rule. When set, layer-major may also coexist with
    --moe-prefill-hit-d2d: prefills below T stay chunked and take the hit/miss split (decode-resident
    experts gathered D2D, only misses over PCIe); a layer-major epoch always copies full layers."""
    try:
        return max(0, int(os.environ.get("QWEN36_LAYER_MAJOR_MIN_TOKENS", "0")))
    except ValueError:
        return 0


def engine_reason(engine) -> str | None:
    """None when this engine can run layer-major prefill, else the reason it cannot."""
    model = engine.model
    inner = getattr(model, "model", None)
    if not getattr(inner, "layer_major_supported", False):
        return "model has no layer-major hooks"
    if getattr(type(model), "layer_major_plain_forward", None) is not type(model).forward:
        return "model overrides the plain embed/layers/norm/lm_head forward"
    if getattr(engine, "native_mtp", None) is not None or hasattr(model, "mtp_hidden"):
        return "native MTP observes every prefill hidden"
    tp = getattr(getattr(engine, "config", None), "tp_info", None)
    if tp is not None and getattr(tp, "size", 1) > 1:
        return "tensor parallel"
    backend = engine.attn_backend
    if not (type(backend).__name__ == "FlashInferBackend" or getattr(backend, "layer_major_safe", False)):
        return f"attention backend {type(backend).__name__} is not qualified for layer-major"
    cache = engine.moe_offload_cache
    if cache is None:
        return "no MoE offload cache (nothing to save: experts are resident)"
    from freetoken.moe.offload_cache import OffloadMoeCache

    if type(cache) is not OffloadMoeCache:
        return f"{type(cache).__name__} slot cache"
    if not cache.prefill_overlap or not cache.prefill_bank_buffers:
        return "MoE prefill overlap (double buffer) is off"
    if cache.prefill_hit_d2d and not min_tokens():
        # Epoch state suppresses hit-D2D for its entire lifetime, including delayed copies.
        return "hit-D2D prefill reads the decode residency slots the store borrows"
    return None


def store_banks(cache):
    """(bank ids the hidden store borrows -- None = all --, refill?). Refill mode
    (QWEN36_LAYER_MAJOR_REFILL=1, GPU decode only) lends only the big banks so the borrowed
    slots keep their small-bank rows and the refill copies only big rows."""
    if cache is None:
        return None, False
    from freetoken.moe.layer_major import ALIGN, bank_row_bytes
    from freetoken.moe.slot_refill import min_row_bytes, refill_enabled

    if refill_enabled() and cache.decode_target == "gpu" and not cache.cpu_layer_ids:
        big = [b for b, rb in enumerate(bank_row_bytes(cache)) if rb >= min_row_bytes() and rb % ALIGN == 0]
        if big:
            return big, True
    return None, False


def attn_batch() -> int:
    """QWEN36_LAYER_MAJOR_ATTN_BATCH: consecutive chunks per full-attention prefill call (1-4)."""
    try:
        k = int(os.environ.get("QWEN36_LAYER_MAJOR_ATTN_BATCH", "1"))
    except ValueError:
        return 1
    return k if k in (1, 2, 3, 4) else 1


def attn_group_layers(inner, engine) -> list[int]:
    """Decoder layers whose attention may be grouped (FlashInfer-style backend with grouped
    prefill support, plain projection path)."""
    backend = engine.attn_backend
    if not hasattr(backend, "group_metadata") or not hasattr(backend, "run_prefill"):
        return []
    hook = getattr(inner, "layer_major_attn_groupable", None)
    if hook is None:
        return []
    return [i for i in range(inner.layer_major_num_layers()) if hook(i)]


def attn_scratch_bytes(inner, layers: list[int], k: int, chunk_rows: int, elem: int) -> int:
    """Q, O and gate buffers of one group of ``k`` chunks: 3 x k x rows x num_q x head_dim x elem."""
    if k <= 1 or not layers:
        return 0
    num_q, head_dim = inner.layer_major_attn_dims(layers[0])
    return 3 * k * chunk_rows * num_q * head_dim * elem


def _set_fla_tensor_cache(size: int) -> int | None:
    try:
        from freetoken.kernel.fla import utils as fla_utils
    except Exception:  # noqa: BLE001 -- CPU tests without triton
        return None
    box = getattr(fla_utils, "TENSOR_CACHE_SIZE", None)
    if box is None:
        return None
    old = box[0]
    box[0] = size
    return old


class LayerMajorExecutor:
    """Runs (layer, chunk) tasks of one request into a borrowed hidden store."""

    def __init__(self, engine, plan, chunk_rows: int, bank_ids=None, refill: bool = False,
                 attn_k: int = 1, attn_layers=()):
        from freetoken.moe.layer_major import HiddenStore

        self.engine = engine
        self.model = engine.model
        self.inner = engine.model.model
        self.cache = engine.moe_offload_cache
        self.ctx = engine.ctx
        self.plan = plan
        self.num_layers = self.inner.layer_major_num_layers()
        weight = self.inner.embed_tokens.weight
        self.device = weight.device
        self.cuda = self.device.type == "cuda"
        if self.cache is not None and getattr(self.cache, "refill", None) is not None:
            # A previous request's refill still owns tail slots this store may borrow: drain it
            # (in-flight copies published in stream order, unissued groups released).
            self.cache.refill.stop()
        self.store = HiddenStore(self.cache, plan, chunk_rows, weight.shape[-1], weight.dtype, self.device,
                                 bank_ids=bank_ids, refill=refill)
        self._planned = None
        self._closed = False
        # Grouped attention (QWEN36_LAYER_MAJOR_ATTN_BATCH=k): Q / O / gate buffers of one group
        # carved from the store's scratch block (borrowed slot bytes, or a plain allocation).
        self.attn_k = attn_k if (attn_k > 1 and plan.scratch_bytes and attn_layers) else 1
        self.attn_stats = dict(attn_groups=0, attn_grouped_chunks=0, attn_fallback_groups=0,
                               attn_member_probes=0)
        self._group_meta: dict = {}
        self._member_info: dict = {}  # chunk index -> (cta_tile_q, split_kv) of its own k=1 plan
        self._member_cta = None
        self._grouping_ok = self.attn_k > 1
        if self.attn_k > 1:
            num_q, head_dim = self.inner.layer_major_attn_dims(attn_layers[0])
            n = self.attn_k * chunk_rows
            seg = n * num_q * head_dim * weight.element_size()
            raw = self.store.scratch
            assert raw is not None and raw.numel() >= 3 * seg
            self._qbuf = raw[:seg].view(weight.dtype).view(n, num_q, head_dim)
            self._obuf = raw[seg:2 * seg].view(weight.dtype).view(n, num_q, head_dim)
            self._gbuf = raw[2 * seg:3 * seg].view(weight.dtype).view(n, num_q * head_dim)
        if self.cache is not None and not self.cache.prefill_hit_d2d:
            self.cache.begin_layer_major_prefill()
        # FLA caches chunk indices per cu_seqlens object (4 entries, host sync on a miss); keep
        # every chunk of the window cached instead of thrashing (values are identical).
        self._fla_prev = _set_fla_tensor_cache(max(4, 4 * plan.window_chunks + 8))
        self.begin = self.end = None
        self._nsys_begin()
        if self.cuda:
            self.begin = torch.cuda.Event(enable_timing=True)
            self.begin.record(torch.cuda.current_stream(self.device))
        if self.cache is not None and self.cache.prefill_hit_d2d:
            try:
                self.cache.begin_layer_major_prefill()
            except BaseException:
                self.store.release()
                if self._fla_prev is not None:
                    _set_fla_tensor_cache(self._fla_prev)
                self._nsys_end()
                raise

    # -- tasks ------------------------------------------------------------------------------
    def _select_plan(self, batch) -> None:
        self._select_meta(getattr(batch, "attn_metadata", None))

    def _select_meta(self, meta) -> None:
        if meta is not None and meta is not self._planned and hasattr(meta, "initialized"):
            meta.initialized = False  # FlashInfer: re-plan the shared prefill wrapper for this chunk
        self._planned = meta

    def _plan_now(self, meta):
        """Plan ``meta`` on the shared wrapper (if it is not the current plan) and return the
        backend's (cta_tile_q, split_kv) for it."""
        backend = self.engine.attn_backend
        self._select_meta(meta)
        backend._initialize_metadata_once(meta)
        return backend.plan_info(meta)

    def run_attn_group(self, layer: int, locals_, batches, key) -> None:
        """One full-attention layer for consecutive chunks of a window: per chunk input norm,
        QKV, rope and KV store; ONE prefill call over all their query rows against KV [0, end of
        the last chunk) when that is bit-exact (else one call per chunk, i.e. k=1); then per
        chunk output gate, o_proj, post norm and MoE."""
        inner, store = self.inner, self.store
        rows = [b.input_ids.shape[0] for b in batches]
        total = sum(rows)
        qbuf, obuf, gbuf = self._qbuf[:total], self._obuf[:total], self._gbuf[:total]
        off = 0
        for local, batch, n in zip(locals_, batches, rows):
            hid, res = store.hidden_view(local, n), store.residual_view(local, n)
            with self.ctx.forward_batch(batch), self.model.forward_host_ctx(batch, False):
                if layer == 0:
                    res.copy_(inner.layer_major_embed(batch))
                    q, gate, r = inner.layer_major_attn_pre(0, res, None)
                else:
                    q, gate, r = inner.layer_major_attn_pre(layer, hid, res)
            _keep(res, r)
            qbuf[off:off + n].copy_(q)
            gbuf[off:off + n].copy_(gate)
            off += n
        self._attend(layer, batches, rows, qbuf, obuf, key)
        off = 0
        for local, batch, n in zip(locals_, batches, rows):
            hid, res = store.hidden_view(local, n), store.residual_view(local, n)
            with self.ctx.forward_batch(batch), self.model.forward_host_ctx(batch, False):
                h, r = inner.layer_major_attn_post(layer, obuf[off:off + n], gbuf[off:off + n], res)
            _keep(res, r)
            _keep(hid, h)
            off += n

    def _group_bit_exact(self, batches, rows, key) -> bool:
        """The conditions under which one prefill call over the group returns, row for row, the
        bits of the per-chunk calls (FlashInfer FA2, causal, page_size 1): every member's own plan
        and the group plan use the same CTA_TILE_Q and no split-KV (so each row runs the same
        KV-tile sequence and online-softmax steps: N_t = ceil((pos + tile_rows) / 16) per tile),
        and every member starts on a q-tile boundary (CTA_TILE_Q / gqa positions) of the group.
        Each member's own plan is probed once (first grouped layer of its window): the planner's
        split decision depends on the kv length too (FlashInfer 0.6: kv > 128 tokens), so it is
        not cached by row count."""
        if not self._grouping_ok:
            return False
        backend = self.engine.attn_backend
        gqa = max(1, backend.qo_head_local // max(1, backend.kv_head_local))
        for c, batch in zip(key, batches):
            if c not in self._member_info:
                self._member_info[c] = self._plan_now(batch.attn_metadata)
                self.attn_stats["attn_member_probes"] += 1
            info = self._member_info[c]
            if info is None:  # planner without readable plan info: never group
                self._grouping_ok = False
                return False
            if info[1]:
                return False
            if self._member_cta is None:
                self._member_cta = info[0]
            if info[0] != self._member_cta or self._member_cta % gqa:
                return False
        tile = self._member_cta // gqa
        start = 0
        for n in rows[:-1]:
            start += n
            if start % tile:
                return False
        return True

    def _attend(self, layer: int, batches, rows, q, out, key) -> None:
        backend = self.engine.attn_backend
        kv_layer = self.inner.layer_major_attn_kv_layer(layer)
        if len(batches) > 1 and self._group_bit_exact(batches, rows, key):
            meta = self._group_meta.get(key)
            if meta is None:
                meta = self._group_meta[key] = backend.group_metadata(batches)
            info = self._plan_now(meta)
            if info is not None and not info[1] and info[0] == self._member_cta:
                backend.run_prefill(q, kv_layer, meta, out=out)
                self.attn_stats["attn_groups"] += 1
                self.attn_stats["attn_grouped_chunks"] += len(batches)
                return
            self._grouping_ok = False  # the group plan differs from the members': stay at k=1
            logger.warning(f"layer-major grouped attention disabled for this prompt: group plan {info}, "
                           f"member plan ({self._member_cta}, False)")
        if len(batches) > 1:
            self.attn_stats["attn_fallback_groups"] += 1
        off = 0
        for batch, n in zip(batches, rows):
            self._select_meta(batch.attn_metadata)
            backend.run_prefill(q[off:off + n], kv_layer, batch.attn_metadata, out=out[off:off + n])
            off += n

    def drop_group_metadata(self, first: int, end: int) -> None:
        """Forget the group plans and member probes of chunks [first, end) (window done)."""
        for key in [k for k in self._group_meta if first <= k[0] < end]:
            del self._group_meta[key]
        for c in [c for c in self._member_info if first <= c < end]:
            del self._member_info[c]

    def run_task(self, layer: int, local: int, batch) -> None:
        rows = batch.input_ids.shape[0]
        hid = self.store.hidden_view(local, rows)
        res = self.store.residual_view(local, rows)
        inner = self.inner
        with self.ctx.forward_batch(batch), self.model.forward_host_ctx(batch, False):
            if inner.layer_major_uses_attention(layer):
                self._select_plan(batch)
            if layer == 0:
                # chunked: residual = embedding tensor, updated in place by every layer; here the
                # embedding is that same tensor's bytes, living in the store.
                res.copy_(inner.layer_major_embed(batch))
                h, r = inner.layer_major_layer(0, res, None)
            else:
                h, r = inner.layer_major_layer(layer, hid, res)
        _keep(res, r)
        _keep(hid, h)

    def finish(self, batches, sample_args_list):
        """Final norm + LM head of the last chunk, sampling exactly like Engine.forward_batch."""
        from freetoken.engine.engine import ForwardOutput
        from freetoken.engine.qualification import force

        last = batches[-1]
        local = (len(batches) - 1) - self.plan.windows()[-1][0]
        rows = last.input_ids.shape[0]
        with self.ctx.forward_batch(last), self.model.forward_host_ctx(last, False):
            x = self.inner.layer_major_final_norm(self.store.hidden_view(local, rows),
                                                  self.store.residual_view(local, rows))
            logits = self.model.layer_major_logits(x)
        if self.cuda:
            self.end = torch.cuda.Event(enable_timing=True)
            self.end.record(torch.cuda.current_stream(self.device))
        executor = getattr(self.engine, "cpu_moe_executor", None)
        if executor is not None:
            executor.raise_if_unhealthy()
        # RNG parity with the chunked run, which sampled (and discarded) one token per
        # intermediate chunk: same call, same shape, so the generator offset advances identically.
        for batch, args in zip(batches[:-1], sample_args_list[:-1]):
            if args.temperatures is None or args.seeds is not None:
                continue  # greedy / position-seeded sampling consumes no generator offset
            dummy = torch.zeros((batch.size, logits.shape[-1]), dtype=logits.dtype, device=logits.device)
            if force(dummy, batch) is None:
                self.engine.sampler.sample(dummy, args)
        for req in last.reqs:
            req.complete_one()
        batch_logits = logits[: last.size]
        forced = force(batch_logits, last)
        next_tokens_gpu = (
            self.engine.sampler.sample(batch_logits, sample_args_list[-1]) if forced is None else forced
        ).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        if self.cuda:
            copy_done = torch.cuda.Event()
            copy_done.record(torch.cuda.current_stream(self.device))
        else:
            copy_done = _DoneEvent()
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done)

    def close(self, *, drain: bool = False) -> None:
        """End the epoch and drop the store. ``drain``: wait for every enqueued task first (abort:
        the caller frees pages/state slots the tasks still write)."""
        if self._closed:
            return
        self._closed = True
        if self.cuda and self.cache is not None and self.cache.prefill_ready_events:
            # An abort mid-layer leaves copy(L+1) on the copy stream with no consumer; order all
            # later compute (decode may claim the buffer slots: usage 0) after it. Only the
            # double-buffer copies (their ready events), not refill copies queued behind them on
            # the same stream (those target RESERVED slots nothing reads). No host sync.
            current = torch.cuda.current_stream(self.device)
            for event in self.cache.prefill_ready_events:
                current.wait_event(event)
        if drain and self.cuda:
            torch.cuda.current_stream(self.device).synchronize()
        if self.cache is not None:
            self.cache.end_layer_major_prefill()
        self.store.release()
        self._group_meta.clear()
        self._member_info.clear()
        if self.attn_k > 1:
            self._qbuf = self._obuf = self._gbuf = None
        if self._fla_prev is not None:
            _set_fla_tensor_cache(self._fla_prev)
        self._nsys_end()

    # Nsight runs (QWEN36_NSYS_FORWARDS=n, see Engine.forward_batch): the whole layer-major
    # prefill counts as one profiled forward with one 'Qwen36/prefill' range.
    def _nsys_begin(self) -> None:
        self._nsys = int(os.environ.get("QWEN36_NSYS_FORWARDS", "0") or 0) if self.cuda else 0
        if not self._nsys:
            return
        engine = self.engine
        if not hasattr(engine, "_profile_count"):
            engine._profile_count = 0
            torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("Qwen36/prefill")

    def _nsys_end(self) -> None:
        if not getattr(self, "_nsys", 0):
            return
        torch.cuda.nvtx.range_pop()
        engine = self.engine
        if engine._profile_count < self._nsys:
            engine._profile_count += 1
            if engine._profile_count == self._nsys:
                torch.cuda.cudart().cudaProfilerStop()
        self._nsys = 0


def _keep(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Make ``dst`` hold ``src`` (bit copy unless it already is that memory)."""
    if src.data_ptr() != dst.data_ptr() or src.shape != dst.shape:
        dst.copy_(src)


class _DoneEvent:
    """CPU stand-in for the copy-done event (tests)."""

    def synchronize(self) -> None:
        return None

    def query(self) -> bool:
        return True


__all__ = ["LayerMajorExecutor", "attn_batch", "attn_group_layers", "attn_scratch_bytes", "enabled", "engine_reason",
           "min_tokens", "store_banks"]
