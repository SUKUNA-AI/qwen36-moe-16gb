"""Print the compare-quality summary and the FP8 serving facts (clipping, slots, KV bytes) after S1."""
import json
R = '/home/USER/qwen36-api/optimization/reports/'
q = json.load(open(R + 'quality-main-release-fp8kv.json'))
print(json.dumps({k: q.get(k) for k in ('relative_perplexity_growth', 'relative_growth_ci95', 'ppl_gate_pass', 'logit_comparisons',
                                        'forced_targets_complete', 'new_functional_failures', 'quality_gate_pass')}))
rows = [json.loads(line) for line in open(R + 'engine.jsonl') if line.strip()]
fp8 = [r for r in rows if r.get('kv_dtype') == 'torch.float8_e4m3fn']
print(json.dumps({'fp8_requests': len(fp8), 'max_kv_clipped_values_total': max((r.get('kv_clipped_values_total') or 0 for r in fp8), default=None),
                  'expert_slots': sorted({r.get('expert_slots') for r in fp8}), 'kv_bytes': sorted({r.get('kv_bytes') for r in fp8})}))
