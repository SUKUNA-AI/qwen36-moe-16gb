<!-- READY, NOT POSTED. Target: comment on FlashML-org/FreeToken#567 (silent stream through a long prefill; disconnect noticed only at the first token). Covers point 2 only. Checked against upstream main 4d2e2e2 (2026-10-19). Paste everything below this line. -->

A data point for point 2 (a client that left is not noticed until the first token), from a single RTX 5070 Ti 16 GB
running FreeToken 0.1.3 plus a fork, with prompts up to 250K tokens. We do not have anything for point 1 (keepalive on
`/v1/chat/completions`).

**Still present on main 4d2e2e2.** `FrontendManager.stream_with_cancellation` (`server/api_server.py:364-374`) still
calls `request.is_disconnected()` only inside `async for chunk in generator`. During a cold prefill the generator
yields nothing after the role chunk, so the disconnect is seen only at the first token. `with_keepalive`
(`server/generation.py:482`) is used by `/v1/messages` and `/v1/responses` (`anthropic_api.py:513`,
`responses_api.py:605`) but not by `/v1/chat/completions`.

**What we run** (outside FreeToken, `serving/guard.py` in our repo): a pure ASGI middleware that replays the request
body to the app and runs its own watcher task on `receive()`. The watcher does not depend on chunks being produced:
on `http.disconnect` it dispatches `abort_user(uid)` and then cancels the app task. The same middleware covers the
non-streaming case of #644.

**Evidence** (30-minute 256K stability run, `results/long256-30min/l256short-r1-analysis.json`):

- Load: 6 streamed requests with a 250,029-token prompt and no cached prefix. Headers arrived after about 0.34 s, and
  the client disconnected 8, 30 or 55 s after sending, i.e. in the middle of a prefill whose TTFT at 250K is about 77 s.
- Results: 6/6 were cancelled in the engine mid-prefill, with 0 completion tokens and 0 decode forwards.
- The KV pool was empty after each case.
- The client's next request was served 1.64 s (p50) after the disconnect (min 1.15 s, max 1.89 s).

Caveat: our long prefill is a fork-specific layer-major prefill that checks for an abort between its steps. Upstream's
chunked prefill would check between chunks instead. The detection part (a watcher on `receive()` rather than a check
per chunk) is independent of that, and it is the part that matters for this issue.
