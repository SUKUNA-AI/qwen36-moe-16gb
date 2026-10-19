<!-- READY, NOT POSTED. Target: comment on FlashML-org/FreeToken#644 (non-streaming requests keep decoding after the client disconnects). Checked against upstream main 4d2e2e2 (2026-10-19). Replaces the cancellation part of 11-context-admission-streamed-overflow.md. Paste everything below this line. -->

A data point from a different deployment: a single RTX 5070 Ti 16 GB, FreeToken 0.1.3 plus a fork, serving only
`/v1/chat/completions` behind our own ASGI wrapper. We worked around the same root cause from outside FreeToken, and
it agrees with the diagnosis here.

**Still present on main 4d2e2e2.** The non-streaming branches await `generate_full` without watching the connection
(`server/openai_api.py:212-215`, `server/anthropic_api.py:139-142`). `stream_with_cancellation` is used only on the
streaming branches (`openai_api.py:209`, `anthropic_api.py:136`, `responses_api.py:172`). `_record_request_middleware`
is still an `@app.middleware("http")` (`api_server.py:485`), i.e. a `BaseHTTPMiddleware`. The earlier comment says
`/v1/messages` and `/v1/responses` are wrapped on main; on 4d2e2e2 we only see the wrapper on their streaming
branches. We may be looking at the wrong place.

**What we run** (`serving/guard.py`, `serving/serve_guarded.py` in our repo; not part of FreeToken):

- A pure ASGI middleware, added with `app.add_middleware(...)` so it sits outside the `BaseHTTPMiddleware`. It reads
  the request body itself, replays it to the app, and runs a watcher task on the real `receive()`. On
  `http.disconnect` it dispatches `state.abort_user(uid)` and only then cancels the app task. Our first version
  cancelled first; the non-streaming waiter then unwound before the abort was dispatched.
- Abort dispatch runs in its own retained, de-duplicated task and is awaited for a bounded time
  (`asyncio.shield`). The cleanup task is itself shielded, so a second server-side cancellation cannot abandon it.
- `openai_api.generate_full` is wrapped so that a `CancelledError` also dispatches the abort from its own task. An
  AnyIO cancel scope can cancel every awaited checkpoint inside the handler, including an inline `await abort_user`.

**Evidence.**

- A CPU unit test (`serving/test-guard.py`): a fake `/v1/chat/completions` app that sleeps is cancelled when the
  client sends `http.disconnect`, and the generation slot is released.
- On the GPU: in our 2-hour soak, 29/29 client cancellations passed, and the next request was served after a median
  of 0.37 s from the disconnect (max 0.38 s).
- A live non-streaming disconnect check exists in our runbook but has not been run, so the non-streaming case is
  covered only by the unit test above.

We think the pure-ASGI recorder from the linked commits plus an explicit watcher on the non-streaming branches is
the right fix inside FreeToken; our wrapper is just evidence that the approach works. One detail from our side: send
the abort before cancelling the waiter.
