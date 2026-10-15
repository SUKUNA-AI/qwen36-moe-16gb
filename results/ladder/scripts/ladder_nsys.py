"""CPU-only: analyze-model-nsys.py public JSON per ladder step -> prefill/decode critical-path buckets (chart-friendly).

python3 -I ladder_nsys.py --analysis-dir DIR --steps steps.json --out nsys.json
DIR holds <step-config>.analysis.json (public analyzer output, no symbols). Buckets come from the
latest-finishing-predecessor critical chain per forward (overlapped work is never summed); every
request's buckets add up to its critical total. Decode buckets are per decode step (period; the last
step of a request is excluded by the analyzer). Timings were taken under Nsight: diagnostic only.
"""
import argparse, json, pathlib

BUCKETS = {
    'moe': ('moe_prefill_nvfp4', 'moe_decode_nvfp4', 'moe_router_alignment_reduction', 'swiglu', 'expert_lru_metadata'),
    'fp8_dense_oproj': ('dense_sm120_fp8_gemm', 'dense_fp8_gemv', 'dense_fp8_oproj_gemv', 'static_fp8_activation_quantize',
                        'attention_gate_fp8_fused', 'other_dense_gemm_gemv_reduction', 'nvfp4_linear_gemv'),
    'gather': ('expert_mapped_host_gather',),
    'attention': ('full_attention', 'qk_rope_kv_store'),
    'gdn': ('gdn_recurrence_conv_chunk', 'gdn_output_gated_norm'),
    'h2d_exposed': ('memcpy_h2d',),
    'copies_other': ('memcpy_d2d', 'memcpy_d2h', 'memset', 'tensor_layout_dtype_copy'),
    'norm_sampling_misc': ('rmsnorm', 'sampling', 'embedding_host_rows', 'other_elementwise_index', 'other'),
}
ORDER = list(BUCKETS) + ['idle']
LOOKUP = {cat: bucket for bucket, cats in BUCKETS.items() for cat in cats}


def buckets(phase):
    out = dict.fromkeys(ORDER, 0.0)
    unmapped = {}
    for row in phase.get('by_category', []):
        bucket = LOOKUP.get(row['category'])
        if bucket is None:
            unmapped[row['category']] = row['critical_ms']
            bucket = 'norm_sampling_misc'
        out[bucket] += row['critical_ms']
    out['idle'] = phase.get('gpu_idle_inside_forward_ms', 0.0) + phase.get('gpu_idle_between_forwards_ms', 0.0)
    return out, unmapped


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis-dir', type=pathlib.Path, required=True)
    p.add_argument('--steps', type=pathlib.Path, required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    a = p.parse_args()
    steps = json.loads(a.steps.read_text())['steps']
    out = {'schema': 'ladder-nsys-1', 'bucket_order': ORDER, 'buckets': {k: list(v) for k, v in BUCKETS.items()},
           'method': 'analyze-model-nsys.py critical chain per forward; buckets sum to the request phase critical total; '
                     'decode buckets are ms per decode step; timings under Nsight are diagnostic only',
           'steps': []}
    for s in steps:
        path = a.analysis_dir / f"{s['config']}.analysis.json"
        st = {'id': s['id'], 'config': s['config'], 'lever_short': s['lever_short']}
        if not path.is_file():
            st['status'] = 'NOT_RUN'
            out['steps'].append(st)
            continue
        d = json.loads(path.read_text())
        st['status'] = 'OK'
        st['input_sha256'] = d.get('input_sha256')
        reqs = {r['label']: r for r in d.get('requests', [])}
        st['requests'] = {}
        for entry in d['critical_path']['by_request']:
            label = entry['label']
            meta = reqs.get(label, {})
            pre, pre_unmapped = buckets(entry['phases']['prefill'])
            dec, dec_unmapped = buckets(entry['phases']['decode'])
            steps_in_chain = entry['phases']['decode'].get('instances_in_chain_total') or 0
            per_step = {k: v / steps_in_chain for k, v in dec.items()} if steps_in_chain else None
            st['requests'][label] = {
                'capture_complete': meta.get('capture_complete'),
                'prefill_critical_ms': entry['phases']['prefill']['critical_total_ms'],
                'prefill_buckets_ms': pre,
                'decode_steps_in_chain': steps_in_chain,
                'decode_step_period_ms_p50': (meta.get('decode_step_period_ms') or {}).get('p50_ms'),
                'decode_step_critical_ms': (entry['phases']['decode']['critical_total_ms'] / steps_in_chain) if steps_in_chain else None,
                'decode_buckets_ms_per_step': per_step,
                'transport': entry.get('transport'),
                'unmapped_categories_ms': {'prefill': pre_unmapped, 'decode': dec_unmapped},
            }
        out['steps'].append(st)
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps([(s['id'], s['status']) for s in out['steps']]))


if __name__ == '__main__':
    main()
