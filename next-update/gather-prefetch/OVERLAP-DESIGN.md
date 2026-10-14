# Decode gather/shared expert overlap: DEFERRED

No new active path or flag is shipped. The inspected implementation still computes
router and shared expert before `experts.forward` in
`freetoken/python/freetoken/models/qwen3_5_moe/moe.py`. The routed offload path
performs ensure/copy/compute in `layers/moe.py`. The archived ceiling verdict
proposes overlap but does not implement a CUDA graph protocol. The documented
negative control without a join had 14,248 checksum mismatches; these are historical
results, not results of this branch.

Required happens-before protocol for a future implementation:

1. Router/top-k and `ensure_experts` finish producing miss metadata on compute.
   Source host banks remain pinned/owned and immutable until gather completes.
   A producer event must include async commits from the prior step, not only router
   completion. Destination slots must be reserved against LRU, prefill borrowing,
   refill and graph replay reuse before the side stream can write them.
2. A side stream waits on that producer event, then gathers the same routed bytes
   into fixed destination addresses. It records gather-done after every bank copy.
3. Compute runs the same shared MLP/gate on the original hidden input before the
   routed kernel can modify that input. Compute waits on gather-done before any
   routed kernel reads slots. Final combination stays `(routed + shared)` with the
   original dtype and arithmetic order. Shared output must remain alive until that
   consumer; no allocator reuse across streams without ownership tracking.
4. Reusing buffers or events needs a consumer-done dependency. Re-recording an event
   while an earlier replay still references it, or publishing LRU residency before
   byte readiness, is forbidden. Cancellation/drain must finish outstanding copies
   before releasing slots; the scheduler's existing commit waits remain mandatory.

Capture/replay must create streams/events before capture, fork and join inside
capture, keep router/miss metadata and bank/destination addresses stable and prove
that the side stream rejoins capture before capture ends. No uncaptured background
copy, dynamic allocation/event creation or host-dependent routing may escape the
captured graph. An unsupported shape or capture mode must use the old path.

Before enabling a future implementation, add a CPU trace of its actual control
flow with separate metadata-ready, source-ready, destination-owned, gather-done,
shared-done and reuse events; removing each wait must kill the test. On SM120 run
byte checks with 0..8 misses, adversarial slot reuse, repeated capture/replay,
alternating patterns, cancellation and prefill/decode transitions. The existing
`test_decode_gather.py::test_tuned_gather_graph_replay` checks serial transport only;
it is not evidence for the proposed overlap. Add a dedicated overlap test before
claiming completion. No global synchronize may be used as performance evidence.

Only after those gates: T3 alone -> run-bound teacher -> unprofiled rotating A/B;
then combinations with T1 and T2. CPU traces and compilation do not qualify CUDA
execution. This work is deferred because ownership/capture changes need a separate
implementation and target-device validation, after completed P0-P2.
