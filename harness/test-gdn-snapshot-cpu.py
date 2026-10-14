"""Run model-independent prefix identity tests without pytest or CUDA startup."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

from cpu_imports import enable
enable()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
here = Path(__file__).resolve().parent
source = __import__("repo_paths").FREETOKEN
sys.path.insert(0, str(source / 'python'))
test = source / 'tests/scheduler/test_prefix_snapshot_identity.py'
namespace = runpy.run_path(str(test))
tests = []
for name, function in namespace.items():
    if name.startswith('test_') and callable(function):
        start = time.perf_counter()
        function()
        tests.append({'test': name, 'status': 'PASS', 'elapsed_s': time.perf_counter()-start})
import torch
assert not torch.cuda.is_initialized()
report = {'status': 'CPU_PASS', 'tests': tests, 'cuda_initialized': False,
          'scope': 'tokenprefix identity, ownership, and copy semantics; no model arithmetic',
          'margin_chunk_tail_combinations': 64, 'unrelated_requests_with_recycling': 20,
          'test_sha256': hashlib.sha256(test.read_bytes()).hexdigest()}
(Path(os.environ.get('QWEN36_CPU_REPORT_DIR',here))/'prefix-snapshot-identity.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
