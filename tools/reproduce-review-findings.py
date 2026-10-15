#!/usr/bin/env python3
"""Read-only synthetic reproductions of REVIEW.md findings; not a release gate.

True = the defect still reproduces; False = the fixed code refuses the synthetic input.
"""
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
    """Each boolean means the defect still reproduces; a fixed finding is False (the check refuses the input)."""
    tests=module('synthetic_quality',ROOT/'harness/test-quality-evidence-cpu.py')
    from freetoken.quality_evidence import EvidenceError
    fixture=tests.fixtures()
    for doc in fixture[2:4]:doc['run']={'run_id':'unrelated-run','profile':'wrong-profile','source_files':{'wrong.py':'bad'}}
    try:result={'functional_wrong_run_accepted':tests.cmp.compare(*fixture,'teacher-bf128','teacher-candidate')['bit_exact_pass'] is True}
    except EvidenceError:result={'functional_wrong_run_accepted':False}
    ftw=module('review_ftw',ROOT/'freetoken/python/freetoken/checkpoint/ftw.py')
    refused=getattr(ftw,'FTWIndexError',AssertionError)
    for key,index in (('ftw_unknown_version_accepted',{'version':999,'tensors':[]}),
                      ('ftw_duplicate_name_silently_replaced',{'version':1,'tensors':[{'name':'duplicate','kind':'weight','dtype':'uint8','shape':[1],'global_off':0,'nbytes':1},
                                                                                       {'name':'duplicate','kind':'weight','dtype':'uint8','shape':[2],'global_off':4096,'nbytes':2}]})):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'freetoken_weight.json').write_text(json.dumps({'format':'freetoken_weight','align':4096,'shard_limit':8192,'total_bytes':8192,
                'shards':[{'file':'freetoken-00000.ftw','global_off':0,'nbytes':8192}],**index}))
            try:
                r=ftw.FTWReader(str(p))
                result[key]=r.index['version']==999 if key=='ftw_unknown_version_accepted' else r.tensors['duplicate']['nbytes']==2
                r.close()
            except refused:result[key]=False
    profile=module('review_profile_requests',ROOT/'harness/profile-request.py')
    result['long256_profiler_max_document_tokens']=max(row[1] for row in profile.final_requests(2048,262144))
    code="import runpy; f=runpy.run_path(__import__('sys').argv[1])['plan_cache_budget']; print(f(300,100,10,4,4,False,10,4))"
    child=subprocess.run([sys.executable,'-O','-c',code,str(ROOT/'freetoken/python/freetoken/engine/cache_budget.py')],
                         env={**__import__('os').environ,'PYTHONPATH':str(ROOT/'freetoken/python')},capture_output=True,text=True)
    result['optimized_budget_accepts_500_bytes_in_300']=child.returncode==0 and child.stdout.strip()=='(4, 10, False)'
    print(json.dumps(result,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
