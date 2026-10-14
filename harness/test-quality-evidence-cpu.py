"""Synthetic, public fixtures; no model imports or reference regeneration."""
import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest
from freetoken.quality_evidence import *

def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), pathlib.Path(__file__).with_name(name+'.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

cmp = load('compare-quality')

def fixtures(clean=False):
    records = {}
    def report(profile, run):
        receipt = {'profile':profile, 'run_id':run, 'started_utc':'synthetic', 'configuration':{'context':131072,'chunk':2048,'env':{}},
                   'source_sha256':__import__('hashlib').sha256(json.dumps({'synthetic.py':'a'*64},sort_keys=True).encode()).hexdigest(),
                   'source_files':{'synthetic.py':'a'*64}, 'service_source_files':{k:'b'*64 for k in ('experiment.py','serve_guarded.py','guard.py','ingress.py','request_metrics.py')},
                   'packages':{k:'synthetic' for k in ('torch','triton','flashinfer-python','sglang-kernel','transformers')},
                   'checkpoint':'synthetic', 'checkpoint_revision':'synthetic', 'native_libraries':{}}
        out = {'fixture_version':1, 'run':receipt, 'fixture_sha256':'f'*64, 'targets':{}, 'rows':[]}
        if clean:
            cold = dict(receipt, run_id=run+'-cold', started_utc='later')
            out['clean_cache'] = {'mode':'restart_same_profile', 'cases':['long-120000'], 'runs':[cold]}
        for i, case in enumerate(TEACHER_CASES):
            cold_case = clean and case == 'long-120000'
            active = cold if cold_case else receipt
            identity = {'profile':'teacher-bf128', 'run_id':'base-cold' if cold_case else 'base', 'case':case,
                        'fixture_sha256':'f'*64, 'targets_sha256':'e'*64, 'files':{'0':'0'*64,'8':'1'*64}}
            out['targets'][case] = {'count':9, 'sha256':'e'*64}
            out['rows'].append({'case':case, 'run_id':active['run_id'], 'clean_cache':cold_case, 'status':200,
                                'forced_text_matches':True, 'expected_tokens':9, 'usage':{'completion_tokens':9}})
            records[(profile, active['run_id'], case)] = {'profile':profile, 'run_id':active['run_id'], 'case':case,
                'seed':900000+i, 'fixture_sha256':'f'*64, 'targets_sha256':'e'*64, 'receipt_sha256':digest(active),
                'nll':[1.0]*9, 'reference':identity,
                'logit_comparisons':[{'position':p,'finite':True,'byte_equal':True,'max_abs':0.0,'relative_l2':0.0} for p in (0,8)]}
        return out
    a,b = report('teacher-bf128','base'),report('teacher-candidate','new')
    def functional(length=120000):
        return {'fixture_version':1, 'functional_matrix':{'version':1,'long':length},
                'rows':[{'case':c,'pass':True,'status':200} for c in ('russian','executable-code','tool-cycle','long-facts-16384',f'long-facts-{length}')]}
    return [a,b,functional(),functional(),records]

class QualityEvidence(unittest.TestCase):
    def compare(self, f):
        return cmp.compare(*f, 'teacher-bf128', 'teacher-candidate')

    def test_clean_restart_and_mixed_old_runs(self):
        f = fixtures(True)
        old = dict(f[4][('teacher-candidate','new','ru-0')], run_id='historical', nll=[99.]*9)
        f[4][('teacher-candidate','historical','ru-0')] = old
        self.assertTrue(self.compare(f)['bit_exact_pass'])

    def test_missing_case_cannot_use_historical_row(self):
        f = fixtures(); row=f[4].pop(('teacher-candidate','new','ru-0'))
        f[4][('teacher-candidate','old','ru-0')] = dict(row,run_id='old')
        with self.assertRaisesRegex(EvidenceError,'missing teacher row'):self.compare(f)

    def test_wrong_case_run_binding(self):
        f=fixtures(True);f[1]['rows'][0]['run_id']='new-cold'
        with self.assertRaisesRegex(EvidenceError,'case->run'):self.compare(f)

    def test_invalid_teacher_data_both_arms(self):
        mutations = [lambda r:r.pop('fixture_sha256'), lambda r:r.update(fixture_sha256='wrong'),
                     lambda r:r.update(nll=[]),lambda r:r.update(nll=[float('nan')]*9),
                     lambda r:r.update(nll=[float('inf')]*9),lambda r:r.update(logit_comparisons=[]),
                     lambda r:r['logit_comparisons'][0].update(max_abs=float('nan')),
                     lambda r:r.update(receipt_sha256='wrong'),lambda r:r.pop('reference')]
        for profile,run in (('teacher-bf128','base'),('teacher-candidate','new')):
            for mutate in mutations:
                with self.subTest(profile=profile, mutation=mutate):
                    f=fixtures();mutate(f[4][(profile,run,'ru-0')])
                    with self.assertRaises(EvidenceError):self.compare(f)

        for arm in (0,1):
            for value in (None,True,2):
                f=fixtures();f[arm]['fixture_version']=value
                with self.assertRaises(EvidenceError):self.compare(f)

    def test_rotated_reference_and_duplicate_rules(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'teacher.jsonl'; row=next(iter(fixtures()[4].values())); line=json.dumps(row)
            p.with_suffix('.jsonl.1').write_text(line+'\n');p.write_text(line+'\n')
            self.assertEqual(len(teacher_records(p)),1)
            p.write_text(json.dumps(dict(row,nll=[2.]*9))+'\n')
            with self.assertRaisesRegex(EvidenceError,'conflicting'):teacher_records(p)
            for damaged in ('{', '{}', line+'\n{', line.replace('1.0','NaN',1)):
                p.write_text(damaged)
                with self.assertRaises(EvidenceError):teacher_records(p)

    def test_empty_incomplete_duplicate_extra_and_types_functional(self):
        for arm in (2,3):
            for mode in ('empty','missing','extra','duplicate','type','row-type','matrix','status'):
                with self.subTest(arm=arm,mode=mode):
                    f=fixtures();doc=f[arm]
                    if mode=='empty':doc['rows']=[]
                    elif mode=='missing':doc['rows'].pop(0)
                    elif mode=='extra':doc['rows'].append({'case':'other','pass':True})
                    elif mode=='duplicate':doc['rows'].append(dict(doc['rows'][0]))
                    elif mode=='type':doc['rows'][0]['pass']=1
                    elif mode=='row-type':doc['rows'][0]='bad'
                    elif mode=='status':doc['rows'][0]['status']=True
                    else:doc.pop('functional_matrix')
                    with self.assertRaises(EvidenceError):self.compare(f)

    def test_run_snapshots_survive_expired_rotation(self):
        f=fixtures()
        for report in f[:2]:
            report['teacher_evidence']=[row for key,row in f[4].items() if key[0]==report['run']['profile']]
        with tempfile.TemporaryDirectory() as d:
            path=pathlib.Path(d)/'teacher.jsonl'
            records=evidence_records(path,f[:2])
            self.assertEqual(records,f[4])
            report=copy.deepcopy(f[0]);report['teacher_evidence'].append(report['teacher_evidence'][0])
            with self.assertRaises(EvidenceError):evidence_records(path,[report])
            path.write_text(json.dumps(dict(f[0]['teacher_evidence'][0],nll=[2.]*9))+'\n')
            with self.assertRaises(EvidenceError):evidence_records(path,f[:2])

    def test_real_engine_reference_lookup_rotated_and_file_hash(self):
        import os, types, sys
        import numpy as np
        import torch
        from unittest.mock import patch
        from repo_paths import FREETOKEN
        f=fixtures()
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);reports=root/'reports';reports.mkdir();refs=reports/'logits';refs.mkdir()
            fixture=root/'fixture.json'
            fixture.write_text(json.dumps({'targets':{str(900000+i):{'case':c,'ids':[0]*9} for i,c in enumerate(TEACHER_CASES)}}))
            f[0]['fixture_sha256']=file_digest(fixture)
            for c in TEACHER_CASES:
                f[0]['targets'][c]['sha256']=digest([0]*9)
                row=f[4][('teacher-bf128','base',c)]
                row.update(fixture_sha256=file_digest(fixture),targets_sha256=digest([0]*9))
                row['reference'].update(fixture_sha256=file_digest(fixture),targets_sha256=digest([0]*9))
            for pos in (0,8):
                file=refs/f'900000-{pos}.npy';np.save(file,np.zeros(4,dtype=np.float32))
                f[4][('teacher-bf128','base','ru-0')]['reference']['files'][str(pos)]=file_digest(file)
            (reports/'teacher.jsonl.1').write_text('\n'.join(json.dumps(v) for k,v in f[4].items() if k[0]=='teacher-bf128')+'\n')
            reference=reports/'baseline.json';reference.write_text(json.dumps(f[0]))
            active=dict(f[1]['run'],profile='teacher-main-release-shortroute')
            (root/'active-run.json').write_text(json.dumps(active))
            spec=importlib.util.spec_from_file_location('engine_teacher_reference_test',FREETOKEN/'python/freetoken/engine/qualification.py')
            engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
            prefill=types.ModuleType('freetoken.scheduler.prefill');prefill.ChunkedReq=type('ChunkedReq',(),{})
            req=types.SimpleNamespace(uid=1,sampling_params=types.SimpleNamespace(seed=900000))
            batch=types.SimpleNamespace(size=1,reqs=[req])
            env={'QWEN36_TEACHER_FIXTURE':str(fixture),'QWEN36_TEACHER_REFERENCE_REPORT':str(reference),
                 'QWEN36_LOGIT_REFERENCE':str(refs),'QWEN36_PROFILE':active['profile'],
                 'QWEN36_RUN_ID':active['run_id'],'QWEN36_DATA_ROOT':str(root)}
            with patch.dict(os.environ,env),patch.dict(sys.modules,{'freetoken.scheduler.prefill':prefill}):
                self.assertEqual(engine.force(torch.zeros(1,4),batch).item(),0)
                self.assertEqual(engine._results[1]['logit_comparisons'][0]['max_abs'],0)
                np.save(refs/'900000-0.npy',np.ones(4,dtype=np.float32));req.uid=2
                with self.assertRaisesRegex(EvidenceError,'missing/replaced'):engine.force(torch.zeros(1,4),batch)
            self.assertEqual(set(refs.iterdir()),{refs/'900000-0.npy',refs/'900000-8.npy'})

    def test_zero_numeric_error_without_byte_equality_is_not_bit_exact(self):
        f=fixtures();f[4][('teacher-candidate','new','ru-0')]['logit_comparisons'][0]['byte_equal']=False
        report=self.compare(f)
        self.assertTrue(report['quality_gate_pass']);self.assertFalse(report['bit_exact_pass'])
        with self.assertRaises(ValueError):load('make-qualification').check_quality_bitexact_report(report)

    def test_fixture_preparation_never_overwrites_or_implicitly_creates(self):
        import fixtures as source
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);(root/'config').mkdir()
            path=root/'config/teacher-fixture.json'
            cases=[{'case':'synthetic','seed':900000,'ids':[0,1]}]
            with patch.object(source,'ROOT',root),patch.object(source,'teacher_cases',return_value=cases):
                with self.assertRaisesRegex(RuntimeError,'missing'):source.prepare(None)
                self.assertFalse(path.exists())
                source.prepare(None,create=True);before=path.read_bytes();source.prepare(None)
                self.assertEqual(path.read_bytes(),before)
                path.write_text('{"fixture_version":1,"targets":{}}');before=path.read_bytes()
                with self.assertRaisesRegex(RuntimeError,'refusing'):source.prepare(None,create=True)
                self.assertEqual(path.read_bytes(),before)

    def test_context_matrices_and_distinct_verdicts(self):
        f=fixtures(); f[3]['functional_matrix']['long']=250000; f[3]['rows'][-1]['case']='long-facts-250000'
        self.assertTrue(self.compare(f)['bit_exact_pass'])
        f[4][('teacher-candidate','new','ru-0')]['logit_comparisons'][0]['max_abs']=0.1
        f[4][('teacher-candidate','new','ru-0')]['logit_comparisons'][0]['byte_equal']=False
        result=self.compare(f);self.assertTrue(result['evidence_valid']);self.assertTrue(result['quality_gate_pass'])
        self.assertFalse(result['bit_exact_pass'])
        gate=load('make-qualification')
        with self.assertRaises(ValueError):gate.check_quality_bitexact_report(result)
        result['logit_comparisons']['max_abs']=0;result['logit_comparisons']['all_byte_equal']=True;result['bit_exact_pass']=True
        gate.check_quality_bitexact_report(result)
        for key in ('schema','evidence_valid','bit_exact_pass'):
            bad=dict(result);bad.pop(key)
            with self.assertRaises(ValueError):gate.check_quality_bitexact_report(bad)

if __name__=='__main__':unittest.main(verbosity=2)
