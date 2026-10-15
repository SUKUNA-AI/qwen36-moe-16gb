"""Existing CPU cache/daemon regressions, counted from pytest's JUnit result."""
import json
import os
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET
from cpu_imports import enable
from repo_paths import FREETOKEN

def main():
    enable()
    import pytest
    import torch
    suites=['kvcache/test_linear_state_pool_alloc.py','kvcache/radix','daemon']
    suites += ['engine/test_cache_budget.py', 'engine/test_host_embeddings.py']
    with tempfile.TemporaryDirectory(prefix='qwen36-core-cpu-') as temp:
        xml=Path(temp)/'result.xml'
        rc=pytest.main(['-q','--tb=short',*[str(FREETOKEN/'tests'/p) for p in suites],f'--junitxml={xml}'])
        root=ET.parse(xml).getroot()
        rows=list(root.iter('testsuite'))
        counts={key:sum(int(r.get(key,0)) for r in rows) for key in ('tests','failures','errors','skipped')}
    report={'tests_run':counts['tests'],'failed_count':counts['failures']+counts['errors'],
            'skipped':counts['skipped'],'cuda_initialized':torch.cuda.is_initialized()}
    print(json.dumps(report))
    return int(rc or report['failed_count'] or report['tests_run']<=report['skipped'] or report['cuda_initialized'])
if __name__=='__main__':raise SystemExit(main())
