"""ncu_pick2.py TARGETS.json -> 'label scope rank' lines for the six wanted kernels (rank = index inside the scope list, 1-based)."""
import json, sys
rows = json.load(open(sys.argv[1]))['candidates']
WANT = [('moe_decode', 'fresh_16k', 'decode', ['moe_decode_nvfp4']),
        ('fp8_gemv', 'fresh_16k', 'decode', ['dense_fp8_gemv']),
        ('oproj', 'fresh_16k', 'decode', ['dense_fp8_oproj_gemv', 'other_dense_gemm_gemv_reduction']),
        ('attention_decode', 'fresh_16k', 'decode', ['full_attention']),
        ('gather', 'fresh_16k', 'decode', ['expert_mapped_host_gather']),
        ('moe_prefill_gemm', 'fresh_16k', 'prefill', ['moe_prefill_nvfp4'])]
for label, scope, phase, cats in WANT:
    inscope = [r for r in rows if r.get('scope') == 'request:' + scope]
    for cat in cats:      # preference order: the S3 kernel first, else the cuBLAS class
        hit = next((i + 1 for i, r in enumerate(inscope) if r.get('phase') == phase and r.get('category') == cat), None)
        if hit:
            print(label, scope, hit, cat); break
