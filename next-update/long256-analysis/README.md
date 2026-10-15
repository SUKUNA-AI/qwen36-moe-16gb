# long256 read-only analysis (cloud delivery, 2026-10-14)

CPU-only sizing and arithmetic over historical evidence, delivered next to the
`agent/cloud-next-update` commits and run against its head `7c30166`. Not a GPU
benchmark; every number marked "derived" or "extrapolation" was not measured.

- Layer-major at 250K-262K needs 2 windows with long256's 1481 slots. One window
  would need 1736-1840 slots, i.e. +433-610 MiB of VRAM (attention batch 2-4).
- Modelling attention as N^2 from the 120K trace gives ~80% of 250K prefill
  in attention (extrapolation). The second window re-streams the 18.25 GB bank
  (36.49 vs 18.25 GB H2D), roughly 0.4 s at the 42 GB/s PCIe ceiling against
  ~77 s TTFT (derived here, not measured).
- Config mismatch: the profile schema accepted `layer_major_attn_batch` 8 while
  the engine ran k=1. Since `cee127b` the schema accepts 1-4 only, so the script's
  `validate_profile_schema(..., attn_batch=8)` step now raises by design.

Reproduce at the analysed head: `python long256-analysis.py <checkout at 7c30166>`.
