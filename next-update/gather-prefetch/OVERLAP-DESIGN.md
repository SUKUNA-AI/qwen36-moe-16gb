# Decode gather/shared expert overlap: IMPLEMENTED, GPU NOT_RUN

`QWEN36_DECODE_SHARED_OVERLAP=1` opts into the single-token Qwen GPU offload
path. Unset/0 preserves serial execution. Other raw values fail at cache setup.
The profile switch is `decode_shared_overlap`; qualified=true is explicitly
rejected until a separately reviewed qualification extension has component,
teacher and model A/B evidence. Candidate configs are `main-release-overlap`
and `main-release-shortroute-fla-wu-overlap`. No released profile changed.

## Ownership and happens-before

1. The scheduler's existing engine<-scheduler wait orders previous async commits
   and restores before forward. Router and unchanged fused_topk run on compute.
   ensure_experts reserves LRU slots and rewrites ids, then records ready on that
   same compute stream. The side stream waits on ready before reading miss count,
   src_indices/evict_slots or writing slots. Immutable pinned source banks and
   device pointer tables are cache-owned across all calls and graph lifetimes.
2. copy_missing runs unchanged on the side stream (default or tuned gather).
   The original shared MLP and sigmoid gate run on compute. Shared needs the
   original hidden input: routed GEMM may overwrite it, so it must come later.
3. done is recorded after the entire copy, and compute waits on done before routed
   GEMM. Final arithmetic stays `(routed + shared)` with the same dtype/order.
4. The next layer's ensure runs after routed consumption on compute; its ready
   edge orders reuse of shared miss buffers and destination slots. Every call
   rejoins; no background work survives a normal return or Python callback error.
   Prefill borrowing, cancellation and cache rebuild inherit this compute order.
   Rebuild already drains and destroys graphs before replacing cache addresses.
   Concurrent forwards on distinct compute streams are unsupported, as before.

Streams and CUDA event handles are created/materialized at cache construction,
before graph capture. Each ready wait forks capture and the done wait rejoins it
inside the same captured forward. Graph replay uses fixed bank, metadata and
slot addresses; it executes device miss counts, never a host snapshot. There is
no global synchronize in the new execution path. CUDA faults invalidate capture
and propagate; a Python finally cannot recover a failed CUDA context.

The path falls back to serial for prefill, batch != 1, TP != 1, CPU/hybrid or
unpinned layers, a non-fused copy layout, active layer-major epochs and refill. Configured refill disables the overlap
object before capture, even while no refill job is active.
The profile wrapper clears inherited flags. No new kernel or numerical method
was introduced. Source-byte identity remains a claim to test on SM120.

## Validation

`harness/test-decode-overlap-cpu.py` executes actual model/layer/helper control
flow with fake streams, exercises disabled/unsupported modes and callback errors,
and detects removal of either producer or consumer wait. It is not CUDA proof.
`freetoken/tests/moe/test_decode_overlap_gpu.py` uses actual ensure/LRU/copy and
byte consumers, real pinned six-bank NVFP4 sizes, default/tuned gather, 1/3 layers,
eager and twice-captured 180-replay sequences with changing routes plus cancel.
The model teacher gate tests actual expert arithmetic and prefix/cancel transitions.

The archived no-join negative control had 14,248 mismatches. The +3-4% decode
estimate is historical, not measured for this implementation. Owner must run
components -> teacher -> unprofiled rotating A/B, T3 alone then T1+T2+T3.
