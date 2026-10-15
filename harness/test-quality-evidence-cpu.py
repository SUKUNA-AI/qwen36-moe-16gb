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

def make_receipt(profile, run, configuration=None, source_files=None):
    source_files = source_files or {'synthetic.py':'a'*64}
    # Teacher configs historically omit 'flags'; functional configs carry the list.
    return {'profile':profile, 'run_id':run, 'started_utc':'synthetic',
            'configuration':configuration or {'context':131072,'chunk':2048,'env':{}},
            'source_sha256':__import__('hashlib').sha256(json.dumps(source_files,sort_keys=True).encode()).hexdigest(),
            'source_files':source_files, 'service_source_files':{k:'b'*64 for k in ('experiment.py','serve_guarded.py','guard.py','ingress.py','request_metrics.py')},
            'packages':{k:'synthetic' for k in ('torch','triton','flashinfer-python','sglang-kernel','transformers')},
            'checkpoint':'synthetic', 'checkpoint_revision':'synthetic', 'native_libraries':{}}

def functional_report(profile, run, length=120000):
    receipt = make_receipt(profile, run, {'context':131072,'chunk':2048,'flags':[],'env':{}})
    return {'fixture_version':1, 'functional_matrix':{'version':1,'long':length}, 'run':receipt,
            'rows':[{'case':c,'pass':True,'status':200,'run_id':run,'serving_run_id':run}
                    for c in ('russian','executable-code','tool-cycle','long-facts-16384',f'long-facts-{length}')]}

def fixtures(clean=False):
    records = {}
    def report(profile, run):
        receipt = make_receipt(profile, run)
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
    return [a,b,functional_report('bf128','base-functional'),functional_report('candidate','new-functional'),records]

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
            key=(row['profile'],row['run_id'],row['case']); other=dict(row,run_id='unrelated')
            p.with_suffix('.jsonl.1').write_text(line+'\n');p.write_text(line+'\n')
            self.assertEqual(len(teacher_records(p)),1)
            # A conflicting duplicate poisons its own identity only; unrelated runs stay usable.
            p.write_text(json.dumps(dict(row,nll=[2.]*9))+'\n'+json.dumps(other)+'\n')
            records=teacher_records(p)
            self.assertNotIn(key,records);self.assertEqual(records.conflicts,{key})
            self.assertIn((row['profile'],'unrelated',row['case']),records)
            # Damaged or identity-less lines can never be selected: listed, not fatal.
            for damaged in ('{', '{}', '{"profile":"p","run_id":null,"case":"c"}', line.replace('1.0','NaN',1)):
                p.write_text(damaged+'\n'+json.dumps(other)+'\n')
                records=teacher_records(p)
                self.assertEqual(len(records.unreadable),1);self.assertEqual(records.conflicts,set())
                self.assertIn(key,records);self.assertIn((row['profile'],'unrelated',row['case']),records)
            p.unlink();p.with_suffix('.jsonl.1').unlink()
            with self.assertRaisesRegex(EvidenceError,'no teacher evidence'):teacher_records(p)

    def test_conflicting_duplicate_fails_only_the_selected_identity(self):
        f=fixtures();key=('teacher-candidate','new','ru-0')
        records=TeacherRecords(f[4]);records.pop(key);records.conflicts.add(key);f[4]=records
        with self.assertRaisesRegex(EvidenceError,'conflicting'):self.compare(f)
        f=fixtures();records=TeacherRecords(f[4]);records.conflicts.add(('teacher-candidate','other-run','ru-0'))
        f[4]=records;self.assertTrue(self.compare(f)['bit_exact_pass'])
        f=fixtures(True);records=TeacherRecords(f[4]);records.conflicts.add(('teacher-candidate','new-cold','ru-0'))
        f[4]=records
        with self.assertRaisesRegex(EvidenceError,'clean-cache run served other cases'):self.compare(f)
        with tempfile.TemporaryDirectory() as d:
            path=pathlib.Path(d)/'teacher.jsonl';f=fixtures();snapshot=f[1]
            snapshot['teacher_evidence']=[row for k,row in f[4].items() if k[0]=='teacher-candidate']
            row=snapshot['teacher_evidence'][0]
            path.write_text(json.dumps(row)+'\n'+json.dumps(dict(row,nll=[2.]*9))+'\n')
            with self.assertRaisesRegex(EvidenceError,'conflicting'):evidence_records(path,[snapshot])

    def test_empty_incomplete_duplicate_extra_and_types_functional(self):
        for arm in (2,3):
            for mode in ('empty','missing','extra','duplicate','type','row-type','matrix','status'):
                with self.subTest(arm=arm,mode=mode):
                    f=fixtures();doc=f[arm]
                    if mode=='empty':doc['rows']=[]
                    elif mode=='missing':doc['rows'].pop(0)
                    elif mode=='extra':doc['rows'].append({'case':'other','pass':True,'status':200,'run_id':doc['run']['run_id']})
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

    def test_baseline_reference_created_only_on_opt_in_and_never_replaced(self):
        import os, types, sys
        import numpy as np
        import torch
        from unittest.mock import patch
        from repo_paths import FREETOKEN
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);refs=root/'reports/logits';refs.parent.mkdir()
            fixture=root/'fixture.json'
            fixture.write_text(json.dumps({'targets':{'900000':{'case':'ru-0','ids':[0]*9}}}))
            active=dict(fixtures()[0]['run'])
            (root/'active-run.json').write_text(json.dumps(active))
            prefill=types.ModuleType('freetoken.scheduler.prefill');prefill.ChunkedReq=type('ChunkedReq',(),{})
            env={'QWEN36_TEACHER_FIXTURE':str(fixture),'QWEN36_LOGIT_REFERENCE':str(refs),'QWEN36_PROFILE':'teacher-bf128',
                 'QWEN36_RUN_ID':active['run_id'],'QWEN36_DATA_ROOT':str(root)}
            def engine():
                spec=importlib.util.spec_from_file_location('engine_teacher_create_test',FREETOKEN/'python/freetoken/engine/qualification.py')
                m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
            def force(m,uid):
                req=types.SimpleNamespace(uid=uid,sampling_params=types.SimpleNamespace(seed=900000))
                return m.force(torch.arange(4,dtype=torch.float32).reshape(1,4),types.SimpleNamespace(size=1,reqs=[req]))
            file=refs/'900000-0.npy'
            with patch.dict(os.environ,env),patch.dict(sys.modules,{'freetoken.scheduler.prefill':prefill}):
                os.environ.pop('QWEN36_TEACHER_CREATE_REFERENCE',None)
                with self.assertRaisesRegex(EvidenceError,'preexisting'):force(engine(),1)
                self.assertFalse(file.exists())
                os.environ['QWEN36_TEACHER_CREATE_REFERENCE']='1'
                m=engine();force(m,1)
                self.assertTrue(np.array_equal(np.load(file),np.arange(4,dtype=np.float32)))
                self.assertEqual(m._results[1]['reference']['files'],{'0':file_digest(file)})
                self.assertTrue(m._results[1]['logit_comparisons'][0]['byte_equal'])
                np.save(file,np.ones(4,dtype=np.float32));before=file.read_bytes()
                with self.assertRaisesRegex(EvidenceError,'differs from pinned'):force(engine(),2)
                self.assertEqual(file.read_bytes(),before)

    def test_qualify_and_quality_refuse_before_any_gpu_work(self):
        import sys, types
        from unittest.mock import patch
        qualify=load('qualify');quality=load('quality')
        f=fixtures()
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);reports=root/'reports';reports.mkdir()
            (reports/'teacher.jsonl').write_text(''.join(json.dumps(v)+'\n' for k,v in f[4].items() if k[0]=='teacher-bf128'))
            (reports/'ref.json').write_text(json.dumps(f[0]));(reports/'fb.json').write_text(json.dumps(f[2]))
            (reports/'legacy.json').write_text(json.dumps({'rows':[]}))
            args=lambda **kw:types.SimpleNamespace(**{'baseline_teacher':False,'reference_report':'ref','functional_baseline':'fb',**kw})
            with patch.object(qualify,'ROOT',root):
                qualify.preflight(args(),['new-teacher','new-functional','quality-new'])
                (reports/'quality-new.json').write_text('{}')
                with self.assertRaisesRegex(SystemExit,'already exist'):qualify.preflight(args(),['new-teacher','quality-new'])
                for bad in (dict(reference_report='legacy'),dict(reference_report='absent'),dict(functional_baseline='legacy')):
                    with self.subTest(bad=bad),self.assertRaisesRegex(SystemExit,'not run-bound'):qualify.preflight(args(**bad),['other'])
                with self.assertRaisesRegex(SystemExit,'already exist'):qualify.preflight(args(baseline_teacher=True),['x','ref'])
            with patch.object(quality,'ROOT',root),patch.object(sys,'argv',['quality.py','--name','ref']):
                with self.assertRaisesRegex(SystemExit,'already exists'):quality.main()

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

    def test_final_qcheck_uses_the_same_gate_under_isolation_and_optimization(self):
        import subprocess, sys
        helper=pathlib.Path(__file__).parent/'gates/final-helpers.py'
        good=self.compare(fixtures())
        with tempfile.TemporaryDirectory() as d:
            path=pathlib.Path(d)/'quality.json'
            for mode in ('good','legacy','byte-mismatch','missing-cases','nonzero'):
                report=copy.deepcopy(good)
                if mode=='legacy':report.pop('schema')
                elif mode=='byte-mismatch':report['logit_comparisons']['all_byte_equal']=False
                elif mode=='missing-cases':report['rows'].pop()
                elif mode=='nonzero':report['logit_comparisons']['max_abs']=0.1
                path.write_text(json.dumps(report))
                result=subprocess.run([sys.executable,'-I','-O',str(helper),'qcheck',str(path)],capture_output=True,text=True,timeout=10)
                self.assertEqual(result.returncode,0 if mode=='good' else 3,result.stdout+result.stderr)
                self.assertEqual(json.loads(result.stdout)['pass'],mode=='good')
        driver=helper.with_name('final-gates.sh').read_text()
        self.assertNotIn('then st=PASS_EQUIV_R1B256; q=0',driver)

    def test_functional_evidence_is_bound_to_the_teacher_arm(self):
        # H1: functional rows count only when served by their own receipt, and that receipt served the same
        # sources, packages, libraries, checkpoint and configuration as the arm's teacher run.
        def resource(doc, **changes):
            run = doc['run']
            doc['run'] = make_receipt(run['profile'], run['run_id'], changes.pop('configuration', run['configuration']),
                                      changes.pop('source_files', run['source_files']))
            doc['run'].update(changes)
        library = {'path':'/lib/x.so','sha256':'c'*64}
        def same_run(d, arm):
            d['run']['run_id'] = {2:'base', 3:'new'}[arm]
            for r in d['rows']: r.update(run_id=d['run']['run_id'], serving_run_id=None)
        mutations = {
            'unrelated-run': lambda d, arm: d.update(run={'run_id':'unrelated-run','profile':'wrong-profile','source_files':{'wrong.py':'bad'}}),
            'missing-run': lambda d, arm: d.pop('run'),
            'wrong-profile': lambda d, arm: d['run'].update(profile='teacher-candidate'),
            'teacher-profile-as-functional': lambda d, arm: d['run'].update(profile='teacher-'+d['run']['profile']),
            'row-run': lambda d, arm: d['rows'][2].update(run_id='other'),
            'row-run-missing': lambda d, arm: d['rows'][2].pop('run_id'),
            'served-by-other': lambda d, arm: d['rows'][4].update(serving_run_id='other'),
            'clean-cache': lambda d, arm: d.update(clean_cache={'runs':[]}),
            'source': lambda d, arm: resource(d, source_files={'synthetic.py':'d'*64}),
            'service': lambda d, arm: d['run']['service_source_files'].update({'guard.py':'d'*64}),
            'package': lambda d, arm: d['run']['packages'].update(torch='other'),
            'checkpoint': lambda d, arm: d['run'].update(checkpoint_revision='other'),
            'env': lambda d, arm: d['run']['configuration']['env'].update(QWEN36_PREFIX_MARGIN='8'),
            'library': lambda d, arm: (d['run']['configuration']['env'].update(QWEN36_FP8_GEMV_LIBRARY='/lib/x.so'),
                                       d['run']['native_libraries'].update(QWEN36_FP8_GEMV_LIBRARY=library)),
            'flags': lambda d, arm: d['run']['configuration'].update(flags=['--moe-prefill-hit-d2d']),
            'memory-ratio': lambda d, arm: d['run']['configuration'].update(memory_ratio=0.8),
            'context': lambda d, arm: d['run']['configuration'].update(context=262144),
            'extra-field': lambda d, arm: d['run'].update(kernel_inventory_sha256='d'*64),
            'same-run-as-teacher': same_run,
        }
        for arm in (2, 3):
            for name, mutate in mutations.items():
                with self.subTest(arm=arm, mutation=name):
                    f = fixtures(); mutate(f[arm], arm)
                    with self.assertRaises(EvidenceError): self.compare(f)
        # A request whose metrics join timed out has no serving run; that alone is not a mismatch.
        f = fixtures(); f[3]['rows'][0]['serving_run_id'] = None; del f[2]['rows'][1]['serving_run_id']
        self.assertTrue(self.compare(f)['bit_exact_pass'])
        self.assertEqual(self.compare(fixtures())['functional_runs'], {'baseline':'base-functional', 'candidate':'new-functional'})

    def test_teacher_knobs_and_missing_flags_are_not_identity(self):
        f = fixtures()
        knobs = {k:'/synthetic/'+k for k in TEACHER_DIAGNOSTIC_ENV}
        for arm, teacher in ((0, 'teacher-bf128'), (1, 'teacher-candidate')):
            f[arm]['run']['configuration']['env'].update(knobs)
            for row in f[4].values():
                if row['profile'] == teacher: row['receipt_sha256'] = digest(f[arm]['run'])
        self.assertNotIn('flags', f[1]['run']['configuration'])
        self.assertTrue(self.compare(f)['bit_exact_pass'])
        # Only the three declared knobs are ignored; creating a reference is a different served run.
        f[1]['run']['configuration']['env']['QWEN36_TEACHER_CREATE_REFERENCE'] = '1'
        for row in f[4].values():
            if row['profile'] == 'teacher-candidate': row['receipt_sha256'] = digest(f[1]['run'])
        with self.assertRaisesRegex(EvidenceError, 'serving identity differs'): self.compare(f)

    def test_teacher_knobs_match_the_qualification_tool(self):
        self.assertEqual(TEACHER_DIAGNOSTIC_ENV, load('make-qualification').RUN_DIAGNOSTIC_ENV)
        self.assertEqual(serving_profile('teacher-main-release'), 'main-release')
        for bad in ('main-release', 'teacher-', None):
            with self.subTest(bad=bad), self.assertRaises(EvidenceError): serving_profile(bad)

    def test_qualify_preflight_binds_the_functional_baseline(self):
        import types
        from unittest.mock import patch
        qualify = load('qualify'); f = fixtures()
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d); reports = root/'reports'; reports.mkdir()
            (reports/'teacher.jsonl').write_text(''.join(json.dumps(v)+'\n' for k,v in f[4].items() if k[0]=='teacher-bf128'))
            (reports/'ref.json').write_text(json.dumps(f[0]))
            other = copy.deepcopy(f[2]); other['run']['packages']['torch'] = 'other'
            (reports/'fb.json').write_text(json.dumps(f[2])); (reports/'fb-other.json').write_text(json.dumps(other))
            (reports/'fb-teacher.json').write_text(json.dumps(dict(f[2], run=dict(f[2]['run'], profile='teacher-bf128'))))
            args = lambda **kw: types.SimpleNamespace(**{'baseline_teacher':False,'reference_report':'ref','functional_baseline':'fb',**kw})
            with patch.object(qualify, 'ROOT', root):
                qualify.preflight(args(), ['new'])
                with self.assertRaisesRegex(SystemExit, 'serving identity differs'): qualify.preflight(args(functional_baseline='fb-other'), ['new'])
                # A new reference is created in the same call: only the functional report's own binding is checked now.
                qualify.preflight(args(baseline_teacher=True, functional_baseline='fb-other'), ['new'])
                with self.assertRaisesRegex(SystemExit, 'not run-bound'):
                    qualify.preflight(args(baseline_teacher=True, functional_baseline='fb-teacher'), ['new'])

    def test_quality_functional_pass_rechecks_the_receipt_after_each_request(self):
        from unittest.mock import patch
        quality = load('quality')
        receipt = make_receipt('candidate', 'live')
        state = {}
        def generate(*a, **k):
            return {'status':200, 'serving_run_id':state['served']}, state['text']
        def request(*a, **k):
            message = {'tool_calls':[{'id':'c1','function':{'name':'get_fixture_weather','arguments':'{"city": "Казань"}'}}]}
            if state['tool'] == 1: message = {'content':'17'}
            state['tool'] += 1
            return {'choices':[{'message':message}]}, state['served']
        def run_receipt():
            state['reads'] += 1
            return dict(receipt, run_id='restarted') if state['reads'] == state['change_at'] else receipt
        def run(change_at=None, served='live'):
            state.update(reads=0, change_at=change_at, served=served, tool=0, text='Москва')
            with patch.multiple(quality, generate=generate, request=request, run_receipt=run_receipt,
                                document=lambda *a: 'x'*30, code_check=lambda text: True):
                return quality.functional(None, 120000, receipt)
        rows = run()
        self.assertEqual([r['case'] for r in rows], ['russian','executable-code','tool-cycle','long-facts-16384','long-facts-120000'])
        self.assertTrue(all(r['run_id'] == r['serving_run_id'] == 'live' for r in rows))
        self.assertEqual(state['reads'], 5)
        with self.assertRaisesRegex(RuntimeError, 'changed during functional pass'): run(change_at=3)
        with self.assertRaisesRegex(RuntimeError, 'served by another run'): run(served='other')
        self.assertTrue(all(r['serving_run_id'] is None for r in run(served=None)))

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
