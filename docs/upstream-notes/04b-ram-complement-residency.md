<!-- DRAFT ONLY, NOT POSTED. Local file; the owner decides whether to post anything. Recommendation: do NOT post. This is a negative result for a feature upstream does not have; there is no upstream defect. Kept for the record. -->

# Note: a "RAM complement" expert residency (host RAM holds only the non-resident experts) was 2-3x slower than FreeToken's full host banks. Negative result, not adopted

## Context

With `--moe-strategy offload`, FreeToken keeps **all** experts in pinned host banks and caches a subset on the GPU,
so the cached experts exist twice. For `nvidia/Qwen3.6-35B-A3B-NVFP4` the host banks take about 16.9 GiB. We
prototyped a residency mode downstream that would save host RAM (a target of about 20%). This is our own code
(`moe/complement*.py`, CUDA transport); FreeToken `cac247a8` has no such mode.

- **complement**: every GPU slot owns its expert, and host RAM stores only the complement (the experts that are not
  resident). An eviction swaps bytes GPU <-> host through registers or a persistent CUDA transport. 2E scratch slots
  are kept for prefill. Policies: LRU, adaptive, decayed frequency.
- **fixed**: a static resident set, with the same host layout.

## Evidence

- Correctness: byte preservation and slot ownership pass at the real expert size. The tests covered fixed, LRU and
  decayed-frequency policies, 128 eager decode steps, 32 CUDA-graph replays and 32 prefill gathers, with both the
  Triton and the persistent CUDA transport. Model outputs had the same hashes as the baseline on the screening
  prompts.
- Speed (single screening series, 128K context, chunk 2048, synthetic prompts, T=0.6, 128 output tokens):

| Prompt | Baseline full host banks: TTFT / decode | complement: TTFT / decode | fixed: TTFT / decode |
|---|---|---|---|
| 64 tokens | 0.58 s / 103 tok/s | 1.32 s / 35 tok/s | 1.32 s / 36 tok/s |
| 4K | 1.80 s / 110 | 4.83 s / 41 | 4.80 s / 36 |
| 16K | 5.83 s / 99 | 15.58 s / 33 | 15.58 s / 36 |
| 64K | 24.52 s / 92 | 61.72 s / 37 | 61.56 s / 34 |
| 120K | 50.70 s / 81.5 | 117.61 s / 36 | 117.55 s / 33 |

- The persistent CUDA transport was faster than the Triton complement exchange, but it was still slower than full
  host banks (component tests; no new whole-model numbers).

## Why it is slower (our reading)

FreeToken's prefill streams the whole layer, all 256 experts, from host to GPU for every chunk. Decode misses are
on the critical path. A complement layout turns every eviction into a two-way exchange on the compute stream, so
copy and compute serialise. WSL2 also exposes a single copy engine (`asyncEngineCount=1`). Saving host RAM is not
worth a 2.3x TTFT and a 2-3x decode loss on this hardware.

## Readiness

Not an upstream issue, and the evidence comes from a single screening series. Do not post. If upstream ever asks
about host-RAM use under `--moe-strategy offload`, the table above can be shared as a data point.
