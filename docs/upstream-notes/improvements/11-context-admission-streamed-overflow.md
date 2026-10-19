<!-- READY, NOT POSTED. Target: FlashML-org/FreeToken, new issue (bug). Paste from the "# " title line down. Checked against upstream main 4d2e2e2 (2026-10-19). -->

# [Bug] Context overflow is checked only after tokenization in the worker: a streamed request gets 200 plus an in-stream error, and an oversized `max_tokens` is silently clamped

## Summary

The context-length check runs only in the scheduler, after the worker has rendered and tokenized the prompt. Two
behaviours follow:

1. A **streamed** request whose prompt does not fit gets `HTTP 200`, an assistant role chunk, and then an SSE
   `{"error": ...}` event, not a 400 `context_length_exceeded`. The `prerender_error` docstring describes the client
   side of this: some agents show only "empty response" for an in-stream error.
2. A request whose prompt fits but whose `prompt + max_tokens` does not has its `max_tokens` lowered silently (a log
   line only). The client asked for N tokens, cannot know it got fewer, and receives `finish_reason: "length"` early.
   The scheduler's own error text already treats prompt plus generation as the limit ("... maximum (prompt +
   generation)").

We fixed both by tokenizing once in the frontend, before any response header, and passing those exact IDs to the
worker.

Other fixes of ours from the same list, checked against main 4d2e2e2: overlap-scheduler output budget (requests
stopped at 127 of 128 tokens) is **fixed upstream** by #546 (`scheduler.py:353`). Request-local `seed` is still
missing on main, but #693 already reports it (our notes go there as a comment). Our disconnect handling is posted as
comments on #644 and #567. Our in-engine abort changes apply only to our layer-major prefill and do not concern
upstream.

## Setup

- FreeToken 0.1.3 `cac247a8` plus our fork; `nvidia/Qwen3.6-35B-A3B-NVFP4`; context 131072; OpenAI
  `/v1/chat/completions`
- The behaviour is in the HTTP/scheduler layer; GPU and model do not matter.

## Problem in upstream today (main 4d2e2e2)

- `python/freetoken/server/openai_api.py:194-199`: for `stream: true`, `prerender_error` runs first, but
  `server/generation.py:358-387` only **renders** the template (`manager.render_prompt`) and never counts tokens.
- `openai_api.py:201-210`: `submit_generation` (`generation.py:297-315`) sends the raw messages to the worker and
  returns a uid. `StreamingResponse` is then returned, which commits `200`.
- `openai_api.py:250-256`: the stream yields the role chunk before any reply. If the worker then rejects the prompt,
  `generation.py:700-701` raises `GenerationError` inside the stream, and `openai_api.py:270-276` sends it as an SSE
  error event.
- `python/freetoken/scheduler/scheduler.py:514-539`: the only length check (`max_output_len <= 0`), which produces the
  `context_length_exceeded` reply.
- `scheduler.py:540-544`: `if msg.sampling_params.max_tokens > max_output_len:` it sets
  `max_tokens = max_output_len` and only logs a warning.
- The prompt is rendered twice (frontend pre-check, then worker) and tokenized once, in the worker.

## What we changed

- `freetoken/python/freetoken/server/generation.py` (`submit_generation`): for text-only requests the frontend
  tokenizer (`state.frontend_tokenizer`) encodes the conversation once, in a worker thread. If `count < 1` or
  `count + max_tokens > max_seq_len`, it raises `GenerationError(..., code="context_length_exceeded")`. Every adapter
  already turns that into a 400 before a `StreamingResponse` exists. Template errors become a 400 the same way.
- `message/tokenizer.py`: `TokenizeMsg.prepared_input_ids`. `tokenizer/tokenize.py` uses those IDs as they are, so the
  worker does not render or tokenize again, and the IDs it serves are exactly the ones that were counted.
- `scheduler/scheduler.py`: `max_tokens > max_output_len` now returns `ErrorReplyMsg(code="context_length_exceeded")`
  instead of clamping. The scheduler stays the authoritative check, for image requests and other paths.

## Measured impact

Boundary gate on the 128K profile (`results/context-gates/final-f1-c128-a1-evaluation.json`):

- prompt 130,396 tokens with `max_tokens` 676 (sum exactly 131,072) -> 200, `finish_reason: "stop"` after 28 tokens;
- the same request plus one token -> 400 `context_length_exceeded`.

Not measured: the extra frontend tokenization time per request (the worker no longer tokenizes, so the total work is
roughly unchanged), and the 256K/64K profiles separately (their gates use the same check).

## Correctness evidence

- The worker serves exactly the IDs the frontend counted, so the admission count and `usage.prompt_tokens` cannot
  disagree.
- Our release soak runs through this path (2-hour soak: 891 client requests, 0 errors; `docs/RESULTS.md` §3).

## Limitations / not measured

- Requests with images keep the old path (the frontend check is skipped and only the scheduler check applies).
- When the client omits `max_tokens`, our check uses the default output budget (`DEFAULT_MAX_OUTPUT_TOKENS`, 32768,
  unless configured). A long prompt without `max_tokens` is then rejected where upstream would clamp. Upstream may
  prefer to clamp only an implicit default and reject only an explicit `max_tokens`.
- Only `/v1/chat/completions` was exercised; `/v1/completions`, `/v1/messages` and `/v1/responses` share
  `submit_generation`, but we did not test them.

## Proposal

A PR along the lines above: count tokens in the frontend before committing the stream, pass `prepared_input_ids`,
and reject (rather than clamp) an explicit `max_tokens` that does not fit. Open question: should an omitted
`max_tokens` keep today's clamp?
