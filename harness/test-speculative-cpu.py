"""Existing CPU primitives from the original CPU gate; no MTP/model execution."""
import inspect
import json
import runpy
from cpu_imports import enable
from repo_paths import FREETOKEN
enable()
namespace=runpy.run_path(str(FREETOKEN/'tests/engine/test_speculative.py'))
count=0
for name,fn in namespace.items():
    if name.startswith('test_') and callable(fn):
        if inspect.signature(fn).parameters:raise RuntimeError('unhandled test fixture')
        fn();count+=1
if not count:raise RuntimeError('zero CPU tests')
import torch
if torch.cuda.is_initialized():raise RuntimeError('unexpected CUDA initialization')
print(json.dumps({'test_count':count,'cuda_initialized':False,'result':'PASS'}))
