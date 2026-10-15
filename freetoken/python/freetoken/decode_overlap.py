"""Default-off, same-layer gather/shared fork-join; no model imports.

The caller owns immutable source banks, slot storage and routing metadata. It
serializes calls on one compute stream, including prior-step scheduler commits.
All side work rejoins that stream even when a Python callback raises. CUDA errors
invalidate capture and must propagate; this is not recovery from device failure.
"""
import os

ENV = "QWEN36_DECODE_SHARED_OVERLAP"


def enabled():
    raw = os.environ.get(ENV, "0")
    if raw not in ("", "0", "1"):
        raise ValueError(f"{ENV}={raw!r}: expected unset, 0 or 1")
    return raw == "1"


class DecodeSharedOverlap:
    def __init__(self, cuda, device):
        self.cuda = cuda
        self.device = device
        self.stream = cuda.Stream(device=device)
        self.ready = cuda.Event()
        self.done = cuda.Event()
        # Materialize CUDA handles before capture; construction follows cache init.
        producer = cuda.current_stream(device)
        self.ready.record(producer)
        self.stream.wait_event(self.ready)
        self.done.record(self.stream)
        producer.wait_event(self.done)

    def run(self, copy_missing, shared):
        producer = self.cuda.current_stream(self.device)
        # Includes ensure_experts and the previous layer's slot consumer. Scheduler
        # engine<-commit ordering also reaches this edge; it must not be removed.
        self.ready.record(producer)
        try:
            with self.cuda.stream(self.stream):
                self.stream.wait_event(self.ready)
                try:
                    copy_missing()
                finally:
                    self.done.record(self.stream)
            return shared()
        finally:
            # Join before routed GEMM, next-layer metadata/slot reuse or cancellation.
            producer.wait_event(self.done)
