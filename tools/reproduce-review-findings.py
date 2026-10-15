#!/usr/bin/env python3
"""Read-only synthetic reproductions of REVIEW.md findings; not a release gate."""
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'harness'),str(ROOT/'freetoken/python')]

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    tests=module('synthetic_quality',ROOT/'harness/test-quality-evidence-cpu.py')
    fixture=tests.fixtures()
    for doc in fixture[2:4]:doc['run']={'run_id':'unrelated-run','profile':'wrong-profile','source_files':{'wrong.py':'bad'}}
    report=tests.cmp.compare(*fixture,'teacher-bf128','teacher-candidate')
    result={'functional_wrong_run_accepted':report['bit_exact_pass'] is True}
    ftw=module('review_ftw',ROOT/'freetoken/python/freetoken/checkpoint/ftw.py')
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp);index={'format':'freetoken_weight','version':999,'shards':[],
                           'tensors':[{'name':'duplicate','nbytes':1},{'name':'duplicate','nbytes':2}]}
        (p/'freetoken_weight.json').write_text(json.dumps(index))
        r=ftw.FTWReader(str(p));result['ftw_unknown_version_accepted']=r.index['version']==999
        result['ftw_duplicate_name_silently_replaced']=r.tensors['duplicate']['nbytes']==2;r.close()
    profile=module('review_profile_requests',ROOT/'harness/profile-request.py')
    result['long256_profiler_max_document_tokens']=max(row[1] for row in profile.final_requests(2048,262144))
    code="import runpy; f=runpy.run_path(__import__('sys').argv[1])['plan_cache_budget']; print(f(300,100,10,4,4,False,10,4))"
    child=subprocess.run([sys.executable,'-O','-c',code,str(ROOT/'freetoken/python/freetoken/engine/cache_budget.py')],
                         env={**__import__('os').environ,'PYTHONPATH':str(ROOT/'freetoken/python')},capture_output=True,text=True)
    result['optimized_budget_accepts_500_bytes_in_300']=child.returncode==0 and child.stdout.strip()=='(4, 10, False)'
    print(json.dumps(result,indent=2))
    return 0 if all(result[k] for k in result if k!='long256_profiler_max_document_tokens') else 1

if __name__=='__main__':raise SystemExit(main())
