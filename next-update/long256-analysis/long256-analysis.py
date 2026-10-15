"""Read-only sizing and arithmetic analysis; no CUDA, inference or speed benchmark.

Run with the isolated CPU Python and pass the source checkout as argv[1].
"""
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
def read(name):
    return json.loads((root / name).read_text())

def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, root / relative)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

planner_path = 'freetoken/python/freetoken/moe/layer_major.py'
planner = module('long256_source_planner', planner_path)
row_bytes = [1048576, 131072, 4096, 524288, 65536, 8192]
plans = []
for tokens in (120000, 250000, 261468, 262144):
    for k in (2, 4):
        chunks = math.ceil(tokens / 2048)
        scratch = 3 * k * 2048 * 16 * 256 * 2
        def plan(slots):
            return planner.plan_store(row_bytes, slots - 512, chunks,
                                      2048, 2048, 2, scratch_bytes=scratch)
        minimum = next(s for s in range(1481, 2500) if plan(s).num_windows == 1)
        current, single = plan(1481), plan(minimum)
        plans.append(dict(tokens=tokens, attention_batch=k, current=current.__dict__,
                          minimum_total_slots_for_one_window=minimum,
                          extra_slot_MiB=(minimum-1481)*sum(row_bytes)/2**20,
                          unborrowed_nonbuffer_slots_current=1481-512-current.borrow_slots,
                          unborrowed_nonbuffer_slots_at_minimum=minimum-512-single.borrow_slots))

runtime_path = 'serving/profile-runtime.py'
runtime = module('long256_profile_schema', runtime_path)
profile = read('serving/profiles.json')['profiles']['long256']
profile['layer_major_attn_batch'] = 8
runtime.validate_profile_schema('long256', profile)
engine_path = 'freetoken/python/freetoken/engine/layer_major.py'
tree = ast.parse((root / engine_path).read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'attn_batch')
namespace = {'os': os}
exec(compile(ast.Module(body=[fn], type_ignores=[]), engine_path, 'exec'), namespace)
env_key = 'QWEN36_LAYER_MAJOR_ATTN_BATCH'
previous = os.environ.get(env_key)
try:
    os.environ[env_key] = '8'
    actual_k = namespace['attn_batch']()
finally:
    if previous is None:
        os.environ.pop(env_key, None)
    else:
        os.environ[env_key] = previous

stability = read('results/long256-30min/metrics.json')
fresh = stability['fresh_by_length']['250000']
ttft, tps = fresh['ttft_p50_s'], fresh['decode_tps_p50']
hit_ttft = stability['continuations']['prefix@250000']['ttft_p50_s']
prefill = read('results/nsight/dashboard.json')['scenarios']['prefill_120k']
attention120 = prefill['buckets_requested']['full_attention']['critical_ms']/1000
total120 = prefill['critical_path_total_ms']/1000
attention250 = attention120 * (250000/120000)**2
other250 = (total120-attention120) * (250000/120000)
sources = [planner_path, runtime_path, engine_path, 'serving/profiles.json',
           'results/long256-30min/metrics.json', 'results/metrics/metrics.json',
           'results/nsight/dashboard.json', 'results/experiments/fla-results.json',
           'next-update/gather-prefetch/src/ceiling_bench.cu']
result = dict(
    scope='CPU source planning + historical evidence arithmetic; NOT a GPU benchmark',
    head=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
    source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
    planner_assumptions=dict(experts_per_layer=256, chunk=2048, hidden=2048,
                             dtype_bytes=2, query_heads=16, head_dim=256,
                             row_bytes=row_bytes, baseline_total_slots=1481,
                             prefix_hit=0, refill=False),
    plans=plans,
    reproduced_config_mismatch=dict(schema_accepts_attention_batch=8, engine_uses=actual_k),
    historical_250k_stability=fresh,
    historical_long_contexts=read('results/metrics/metrics.json')['expert_streaming']['long_contexts'],
    derived_e2e_not_measured=[dict(output_tokens=n,
        cold_s=ttft+(n-1)/tps, cached_s=hit_ttft+(n-1)/tps,
        cold_decode_time_fraction=((n-1)/tps)/(ttft+(n-1)/tps)) for n in (128,512,2048,4096)],
    illustrative_extrapolation_not_measured=dict(
        model='attention N squared; other work N, extrapolated from main128 120K trace',
        attention250_s=attention250, other250_s=other250,
        attention_fraction=attention250/(attention250+other250),
        seconds_saved_if_attention_throughput_grows_10pct=attention250-attention250/1.1,
        fla_seconds_saved_if_per_call_cost_is_unchanged=123*30*(325.5-215.1)/1e6,
        gather_payload_seconds_at_42GBps=320*.3844*1781760/42e9,
        single_KV_read_seconds_at_841GBps=250000*20480/841e9),
    cuda_initialized=__import__('torch').cuda.is_initialized(),
)
print(json.dumps(result, ensure_ascii=False, indent=2))
