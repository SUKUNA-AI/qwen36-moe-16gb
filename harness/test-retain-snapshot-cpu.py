"""CPU regression runner; deliberately leaves GPU/native overlay untouched."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['QWEN36_RETAIN_CHUNK_SNAPSHOT'] = '0'
os.environ['QWEN36_ENGINE_METRICS'] = '0'
here = Path(__file__).resolve().parent
source = here.parents[2] / 'work/FreeToken-opt'
sys.path.insert(0, str(source/'python'))
import torch
torch.set_num_threads(1)

files = ['test_retain_chunk_snapshot.py', 'test_prefix_snapshot_identity.py',
         'test_hybrid_cache_manager.py', 'test_abort_inflight_prefill.py',
         'test_commit_repoints_page_table.py', 'test_scheduler_chunked_prefill.py']
results = []
for filename in files:
    namespace = runpy.run_path(str(source/'tests/scheduler'/filename))
    if filename == 'test_abort_inflight_prefill.py':
        # Older scheduler mock predates the metrics hook's engine argument.
        # Only supply that unused field; no scheduler/cache behavior is mocked.
        original_setup = namespace['_setup']
        def setup_with_engine(original=original_setup):
            values = original()
            values[-1].engine = None
            return values
        for function in namespace.values():
            if callable(function) and hasattr(function, '__globals__'):
                function.__globals__['_setup'] = setup_with_engine
    for name, function in namespace.items():
        if name.startswith('test_') and callable(function):
            assert not inspect.signature(function).parameters, (filename, name)
            t = time.perf_counter()
            function()
            results.append({'file': filename, 'name': name, 'status': 'PASS',
                            'elapsed_s': time.perf_counter()-t})
            print(filename, name, 'PASS', flush=True)
assert not torch.cuda.is_initialized()
runtime = ['core.py', 'attention/linear.py', 'scheduler/prefill.py', 'scheduler/cache.py']
report = {'status': 'CPU_PASS', 'cuda_initialized': False, 'tests': results,
          'test_count': len(results), 'model_gate': 'NOT_RUN',
          'scope': 'Exact synthetic state-prefix identity and state/KV ownership; no model arithmetic',
          'feature_default': 'off', 'extra_gpu_allocation_bytes': 0,
          'qwen36_state_slot_bytes': 64389120, 'qwen36_pool_slots_unchanged': 9,
          'source_sha256': {p: hashlib.sha256((source/'python/freetoken'/p).read_bytes()).hexdigest()
                            for p in runtime},
          'test_sha256': {p: hashlib.sha256((source/'tests/scheduler'/p).read_bytes()).hexdigest()
                          for p in files}}
(here/'retain-snapshot-cpu.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('tests', 'test_sha256')}, indent=2))
