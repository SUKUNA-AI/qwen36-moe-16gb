# Tool-use E2E cycle: https://api.example.invalid/v1/chat/completions (model qwen36-35b)

## Findings (reviewer notes; the generated report follows)

The run was production main128, instance 48d56077, 2026-10-14 20:27-20:37Z, 623 s. It covered 16 synthetic tasks under 10 conditions and made 357 rounds.
- **Protocol/API contract: PASS.**
  - All 14 unsupported-option requests (required, named, parallel_tool_calls=false and non-bool, strict at function and tool level, legacy functions; each stream and non-stream) returned 400 with a JSON body and no SSE.
  - No protocol or transport errors in 357 rounds, streaming or not.
  - All 228 tool-call arguments were well-formed JSON and schema-valid, and every call id was valid and echoed back.
  - tool_choice=none was never violated (20/20).
  - Zero busy-429s; TLS was verified; the instance was the same before and after.
- **Model behaviour:** strict success was 106/120 = 88.3% [81.4, 92.9]. The final answer was correct in 117/120 = 97.5%. Results by group:
  - chains (2-4 dependent calls): 30/30;
  - choosing the right tool: 10/10;
  - parallel: 19/20, and all 20 parallel runs emitted 2-3 calls in one turn;
  - long16k_en: 10/10; long16k_ru: 5/10.
- **No call when one was required (11 runs, all with thinking off):**
  - fixture_add_ru answered "42" directly in 5/5 thinking-off runs, at T=0 and at every T=0.7 seed. It made the call in 5/5 thinking-on runs. This is the recorded near-tie, now consistent on this instance.
  - More important: on lookup prompts the model **invented catalog values** without calling the tool. It answered "150" in none_followup_en 4/5 and "$150.00" in single_lookup_en 2/5, so 6 of 10 thinking-off lookup runs. With thinking on this happened 0/10 times.
- **Incomplete chain (5 runs):** in long16k_ru with thinking on, the model looked the part up but multiplied 7.36x40 itself instead of using the calculator. The answer was correct in 4 runs. In 1 run the final content was empty: the answer was only in the reasoning.
- **"wrong_tool" (2 runs):** these were benign extra calculator calls, and both answers were correct.
- **Cosmetic stream/non-stream mismatch:** with thinking on, the streamed final content starts with "\n\n" in 47/48 runs. Non-stream content never does (0/32).
- **Latency per round (TLS route; GET /v1/models baseline about 9 ms):**
  - Thinking off: streaming TTFT median 0.57 s; total median 0.75-0.77 s, p90 about 1.0 s.
  - Thinking on: TTFT to the first reasoning token 0.54 s; total median 1.56 s, p90 3.7 s, max 14.7 s.
  - long16k round 1: TTFT 1.58 s uncached, 0.62 s cached.
  - Run wall time: median 1.5 s (thinking off) and 3.7 s (thinking on). Maximum 32 s (3038 completion tokens).
- **Tokens:** 1.17M prompt tokens, 0.94M of them cached; 46.6K completion tokens.

Started 2026-10-14T20:27:19.562183+00:00, elapsed 623.5 s; runs 160/160 (NOT_RUN 0, budget-cut 0); instance same: True; TLS verified: True; proxies ignored; no redirects.

**Protocol verdict: PASS** unsupported_options_all_400=ok, no_protocol_or_transport_errors=ok, tool_choice_none_never_called=ok, all_arguments_json_wellformed=ok, same_instance=ok

## Success (strict: required calls with correct name/args in dependency order + correct final answer + no protocol error; Wilson 95%)

- all cases: 142/160 = 88.8% [82.9, 92.8]
- tool tasks: 106/120 = 88.3% [81.4, 92.9]
- tool tasks excl. fixture_add_ru (known near-tie): 101/110 = 91.8% [85.2, 95.6]
- final answer correct (tool tasks): 117/120 = 97.5% [92.9, 99.2]

**By category:** chain 30/30 = 100.0% [88.6, 100.0]; choose 10/10 = 100.0% [72.2, 100.0]; long 15/20 = 75.0% [53.1, 88.8]; no_tool 10/10 = 100.0% [72.2, 100.0]; none 26/30 = 86.7% [70.3, 94.7]; parallel 19/20 = 95.0% [76.4, 99.1]; single 32/40 = 80.0% [65.2, 89.5]

**By stream:** nonstream 71/80 = 88.8% [80.0, 94.0]; stream 71/80 = 88.8% [80.0, 94.0]

**By thinking:** thinking_off 68/80 = 85.0% [75.6, 91.2]; thinking_on 74/80 = 92.5% [84.6, 96.5]

**By temperature:** T0 57/64 = 89.1% [79.1, 94.6]; T>0 (0.6/0.7 seeded) 85/96 = 88.5% [80.6, 93.5]

**By language:** en 72/80 = 90.0% [81.5, 94.8]; ru 70/80 = 87.5% [78.5, 93.1]

**By condition:** t0-off-ns 13/16 = 81.2% [57.0, 93.4]; t0-off-s 14/16 = 87.5% [64.0, 96.5]; t0-on-ns 15/16 = 93.8% [71.7, 98.9]; t0-on-s 15/16 = 93.8% [71.7, 98.9]; t06-on-s1-s 15/16 = 93.8% [71.7, 98.9]; t06-on-s2-ns 15/16 = 93.8% [71.7, 98.9]; t06-on-s3-s 14/16 = 87.5% [64.0, 96.5]; t07-off-s1-ns 14/16 = 87.5% [64.0, 96.5]; t07-off-s2-s 13/16 = 81.2% [57.0, 93.4]; t07-off-s3-ns 14/16 = 87.5% [64.0, 96.5]

**By task:** chain2_en 10/10 = 100.0% [72.2, 100.0]; chain3_ru 10/10 = 100.0% [72.2, 100.0]; chain4_en 10/10 = 100.0% [72.2, 100.0]; choose_temp_ru 10/10 = 100.0% [72.2, 100.0]; fixture_add_ru 5/10 = 50.0% [23.7, 76.3]; long16k_en 10/10 = 100.0% [72.2, 100.0]; long16k_ru 5/10 = 50.0% [23.7, 76.3]; no_tool_ru 10/10 = 100.0% [72.2, 100.0]; none_en 10/10 = 100.0% [72.2, 100.0]; none_followup_en 6/10 = 60.0% [31.3, 83.2]; none_ru 10/10 = 100.0% [72.2, 100.0]; parallel_en 9/10 = 90.0% [59.6, 98.2]; parallel_ru 10/10 = 100.0% [72.2, 100.0]; single_convert_en 9/10 = 90.0% [59.6, 98.2]; single_files_ru 10/10 = 100.0% [72.2, 100.0]; single_lookup_en 8/10 = 80.0% [49.0, 94.3]

## Tool calls

- calls 228; JSON well-formed 228/228 = 100.0% [98.3, 100.0]; schema-valid 228/228 = 100.0% [98.3, 100.0]; acceptable tool name 224/228 = 98.2% [95.6, 99.3]; args match an expected step 224/228 = 98.2% [95.6, 99.3]; call id valid 228/228 = 100.0% [98.3, 100.0]
- parallel: {'runs': 20, 'runs_with_parallel_calls_in_one_turn': 20, 'max_calls_in_one_turn': 3, 'any_task_parallel_turns': 20}
- unsupported options -> HTTP 400 JSON: 14/14 = 100.0% [78.5, 100.0]
- busy 429 retries (single-slot guard, no queue): 0

## Failure taxonomy (primary code per failed run)

- {'no_call_when_required': 11, 'incomplete_chain_or_premature_answer': 5, 'wrong_tool': 2}
- all codes: {'no_call_when_required': 11, 'wrong_final_answer': 7, 'incomplete_chain_or_premature_answer': 5, 'wrong_tool': 2}; warnings: none; protocol issues: none

## Latency per round (s) and tokens

- thinking_off/nonstream: {"rounds": 86, "ttft_s": null, "total_s": {"n": 86, "median": 0.7498, "p90": 0.9713, "max": 1.9624}, "completion_tokens": {"n": 86, "median": 26.0, "p90": 55.0, "max": 105}}
- thinking_off/stream: {"rounds": 60, "ttft_s": {"n": 60, "median": 0.5738, "p90": 0.6064, "max": 0.6182}, "total_s": {"n": 60, "median": 0.7703, "p90": 1.0091, "max": 1.446}, "completion_tokens": {"n": 60, "median": 26.5, "p90": 55.1, "max": 105}}
- thinking_on/nonstream: {"rounds": 62, "ttft_s": null, "total_s": {"n": 62, "median": 1.562, "p90": 3.6682, "max": 12.2035}, "completion_tokens": {"n": 62, "median": 116.0, "p90": 368.6, "max": 1339}}
- thinking_on/stream: {"rounds": 94, "ttft_s": {"n": 94, "median": 0.5402, "p90": 0.5705, "max": 0.6217}, "total_s": {"n": 94, "median": 1.5689, "p90": 3.643, "max": 14.7132}, "completion_tokens": {"n": 94, "median": 113.0, "p90": 351.9, "max": 1570}}
- long16k_round1_uncached: {"ttft_s": {"n": 5, "median": 1.5829, "p90": 1.6099, "max": 1.6176}, "total_s": {"n": 10, "median": 3.4481, "p90": 4.4369, "max": 4.4998}}
- long16k_round1_cached: {"ttft_s": {"n": 5, "median": 0.6173, "p90": 0.6429, "max": 0.6448}, "total_s": {"n": 10, "median": 2.0534, "p90": 7.3961, "max": 29.1852}}
- long16k_later_rounds: {"ttft_s": {"n": 17, "median": 0.5805, "p90": 0.6983, "max": 0.7302}, "total_s": {"n": 35, "median": 1.0238, "p90": 7.0048, "max": 11.1091}}
- run_wall_s_by_thinking: {"off": {"n": 80, "median": 1.5233, "p90": 4.2333, "max": 4.703}, "on": {"n": 80, "median": 3.7035, "p90": 11.5787, "max": 31.9958}}
- note: ttft_s = first token-bearing SSE delta (reasoning, content or tool call) incl. TLS route; non-stream rounds have no TTFT (total only)
- tokens: {'prompt': 1170020, 'completion': 46613, 'cached': 937945, 'rounds': 357, 'long16k_prompt_tokens_round1': [16814, 16816, 16823, 16825]}

## fixture_add_ru (network-acceptance prompt; known T=0/thinking-off near-tie)

- [{'condition': 't0-off-ns', 'success': False, 'failure_code': 'no_call', 'round1_cached_tokens': 0}, {'condition': 't0-off-s', 'success': False, 'failure_code': 'no_call', 'round1_cached_tokens': 0}, {'condition': 't07-off-s1-ns', 'success': False, 'failure_code': 'no_call', 'round1_cached_tokens': 0}, {'condition': 't07-off-s2-s', 'success': False, 'failure_code': 'no_call', 'round1_cached_tokens': 0}, {'condition': 't07-off-s3-ns', 'success': False, 'failure_code': 'no_call', 'round1_cached_tokens': 0}, {'condition': 't0-on-s', 'success': True, 'failure_code': None, 'round1_cached_tokens': 0}, {'condition': 't06-on-s1-s', 'success': True, 'failure_code': None, 'round1_cached_tokens': 0}, {'condition': 't0-on-ns', 'success': True, 'failure_code': None, 'round1_cached_tokens': 0}, {'condition': 't06-on-s2-ns', 'success': True, 'failure_code': None, 'round1_cached_tokens': 0}, {'condition': 't06-on-s3-s', 'success': True, 'failure_code': None, 'round1_cached_tokens': 0}]

## Failed runs

- single_lookup_en@t0-off-ns: ['no_call', 'wrong_final'] rounds=1 calls=[] final=<redacted>
- fixture_add_ru@t0-off-ns: ['no_call'] rounds=1 calls=[] final=<redacted>
- none_followup_en@t0-off-ns: ['no_call', 'wrong_final'] rounds=1 calls=[] final=<redacted>
- fixture_add_ru@t0-off-s: ['no_call'] rounds=1 calls=[] final=<redacted>
- none_followup_en@t0-off-s: ['no_call', 'wrong_final'] rounds=1 calls=[] final=<redacted>
- fixture_add_ru@t07-off-s1-ns: ['no_call'] rounds=1 calls=[] final=<redacted>
- none_followup_en@t07-off-s1-ns: ['no_call', 'wrong_final'] rounds=1 calls=[] final=<redacted>
- single_lookup_en@t07-off-s2-s: ['no_call', 'wrong_final'] rounds=1 calls=[] final=<redacted>
- single_convert_en@t07-off-s2-s: ['wrong_tool'] rounds=4 calls=[('convert_units', {'value': 26.2, 'from_unit': 'mi', 'to_unit': 'km'}), ('calculator', {'operation': 'multiply', 'a': 42.1648, 'b': 100}), ('calculator', {'operation': 'divide', 'a': 4216.48, 'b': 100})] final=<redacted>
- fixture_add_ru@t07-off-s2-s: ['no_call'] rounds=1 calls=[] final=<redacted>
- fixture_add_ru@t07-off-s3-ns: ['no_call'] rounds=1 calls=[] final=<redacted>
- none_followup_en@t07-off-s3-ns: ['no_call', 'wrong_final'] rounds=1 calls=[] final=<redacted>
- long16k_ru@t0-on-s: ['incomplete_chain'] rounds=2 calls=[('lookup_part', {'part_id': 'QX-1182'})] final=<redacted>
- long16k_ru@t06-on-s1-s: ['incomplete_chain'] rounds=2 calls=[('lookup_part', {'part_id': 'QX-1182'})] final=<redacted>
- long16k_ru@t0-on-ns: ['incomplete_chain'] rounds=2 calls=[('lookup_part', {'part_id': 'QX-1182'})] final=<redacted>
- long16k_ru@t06-on-s2-ns: ['incomplete_chain'] rounds=2 calls=[('lookup_part', {'part_id': 'QX-1182'})] final=<redacted>
- parallel_en@t06-on-s3-s: ['wrong_tool'] rounds=3 calls=[('lookup_part', {'part_id': 'ZX-4471'}), ('lookup_part', {'part_id': 'QX-1182'}), ('lookup_part', {'part_id': 'MT-0716'}), ('calculator', {'operation': 'subtract', 'a': 18.75, 'b': 7.36}), ('calculator', {'operation': 'subtract', 'a': 7.36, 'b': 3.95})] final=<redacted>
- long16k_ru@t06-on-s3-s: ['incomplete_chain', 'wrong_final'] rounds=2 calls=[('lookup_part', {'part_id': 'QX-1182'})] final=<redacted>
