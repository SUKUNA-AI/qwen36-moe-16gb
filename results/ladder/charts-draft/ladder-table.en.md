> **SYNTHETIC DRAFT: do not publish as a result.**

### TTFT, s

| Step | Lever | 64 | 2K | 4K | 16K | 120K |
|---|---|---|---|---|---|---|
| L0 | main-release (f1) | 0.601 | n/a | 0.606 | 1.36 | 21.0 |
| L1 | +T1 hit-D2D + layer-major от 2049 | 0.446 (−21.8 %) | 0.585 (+0.7 %) | 0.582 (+0.3 %) | n/a | n/a |
| L2 | +T2 GDN FLA wu | n/a | n/a | n/a | 1.38 (−3.0 %) | 21.7 (−0.9 %) |
| L3 | +T3 decode_shared_overlap | n/a | n/a | n/a | n/a | n/a |
| L4 | +эмбеддинги на хосте | 0.334 (+0.9 %) | 0.312 (−0.6 %) | 0.461 (+7.2 %) | 1.31 (+1.0 %) | 21.4 (+0.1 %) |
| L5 | +memory_ratio 0.85 | 0.252 (−24.6 %) | 0.252 (−19.2 %) | 0.401 (−13.0 %) | 1.21 (−7.3 %) | 20.1 (−5.9 %) |
| L6 | +ядра S3 (o-proj, MoE decode) | n/a | n/a | n/a | n/a | n/a |
| L7 | +C0 приоритет копий | 0.353 (−18.6 %) | n/a | n/a | n/a | 21.8 (−0.2 %) |
| L8 | +C1 только выбранные эксперты (кандидат) | 0.279 (−21.0 %) | n/a | n/a | n/a | n/a |
| T3only | f1 + только T3 | n/a | n/a | n/a | n/a | n/a |

### Decode, tok/s

| Step | Lever | 64 | 2K | 4K | 16K | 120K |
|---|---|---|---|---|---|---|
| L0 | main-release (f1) | 133.7 | n/a | 135.0 | 129.8 | 97.5 |
| L1 | +T1 hit-D2D + layer-major от 2049 | 132.0 (−0.1 %) | 144.1 (+0.4 %) | 143.4 (−0.2 %) | n/a | n/a |
| L2 | +T2 GDN FLA wu | n/a | n/a | n/a | n/a | n/a |
| L3 | +T3 decode_shared_overlap | n/a | n/a | n/a | n/a | n/a |
| L4 | +эмбеддинги на хосте | 141.0 (+4.1 %) | 145.9 (+5.3 %) | 144.9 (+5.4 %) | 138.7 (+4.7 %) | 100.0 (+3.7 %) |
| L5 | +memory_ratio 0.85 | 164.8 (+16.9 %) | 167.3 (+14.7 %) | 166.0 (+14.6 %) | 158.7 (+14.4 %) | 111.2 (+11.2 %) |
| L6 | +ядра S3 (o-proj, MoE decode) | 172.1 (+8.6 %) | n/a | n/a | 159.4 (+7.6 %) | n/a |
| L7 | +C0 приоритет копий | n/a | n/a | n/a | n/a | n/a |
| L8 | +C1 только выбранные эксперты (кандидат) | n/a | n/a | n/a | n/a | n/a |
| T3only | f1 + только T3 | n/a | n/a | n/a | n/a | n/a |

### Teacher and memory

| Step | Teacher | memory_ratio | Slots | Process peak, MiB | Min free, MiB | Misses per token |
|---|---|---|---|---|---|---|
| L0 | pass 125/125, max_abs 0.0 | 0.75 | 3164 | n/a | 436 | 26.9 |
| L1 | pending | 0.75 | n/a | n/a | n/a | n/a |
| L2 | pending | 0.75 | n/a | n/a | n/a | n/a |
| L3 | pending | 0.75 | 3164 | n/a | 2603 | 39.7 |
| L4 | pending | 0.75 | 3729 | n/a | 2562 | 28.8 |
| L5 | pending | 0.85 | 4611 | n/a | 1128 | 16.2 |
| L6 | pending | 0.85 | n/a | n/a | n/a | n/a |
| L7 | pending | 0.85 | n/a | n/a | n/a | n/a |
| L8 | pending | 0.85 | n/a | n/a | n/a | n/a |
| T3only | not_run | n/a | n/a | n/a | n/a | n/a |

In brackets: the step's contribution, the median of paired ratios against its parent; `*` marks a ratio of medians (no pairs).
