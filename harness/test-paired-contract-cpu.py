"""Paired harness V6 CPU contract tests (V4/V5 contracts adapted + V6 statistics/launcher/VRAM).

Never starts services, never contacts HTTP (requests.post is mocked), never touches
the native ROOT or the real GPU lock (temporary roots/locks only), never runs
nvidia-smi or systemd (stubs only) and never initializes CUDA.
Run: CUDA_VISIBLE_DEVICES= python test-paired-contract-cpu.py [--receipt PATH]
"""
import copy,hashlib,importlib.util,io,json,math,os,pathlib,shutil,statistics,subprocess,sys,tempfile,time,types,unittest
from contextlib import redirect_stderr,redirect_stdout
from unittest.mock import MagicMock,patch
import requests
import bench
import tool_fixture_v5 as fx

HERE=pathlib.Path(__file__).resolve().parent


def load(name):
    spec=importlib.util.spec_from_file_location(name.replace('-','_'),HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
comparison=load('compare-ab');matrix=load('matrix');tool_task=load('tool-task-bench');pilot_runner=load('tool-pilot')

PROFILES=['bf128','main-retain','main-retain2048']
FILES={'engine/engine.py':'1'*64}
SOURCE=hashlib.sha256(json.dumps(FILES,sort_keys=True).encode()).hexdigest()
FAKE_HARNESS={name:'3'*64 for name in bench.HARNESS_FILES}
FAKE_LAUNCHER={name:'6'*64 for name in bench.LAUNCHER_FILES}
LAUNCHER_ENV={'CUDA_HOME':'/home/USER/qwen36-api/cuda-13.4','PYTHONPATH':'/home/USER/qwen36-api/optimization/python',
              'QWEN36_DATA_ROOT':'/home/USER/qwen36-api/optimization','OMP_NUM_THREADS':'16',
              'LD_LIBRARY_PATH':'/home/USER/qwen36-api/cuda-13.4/lib:/home/USER/qwen36-api/cuda-13.4/lib64:/usr/lib/wsl/lib:'}
GIB=1<<30
REAL_HARNESS={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in bench.HARNESS_FILES}
V4_ENGLISH_DIAGNOSTIC={'first_finish':'stop','completion_tokens':3,'tool_calls':0}


def sha(text):return hashlib.sha256(text.encode()).hexdigest()


def make_manifest():
    common={'source_sha256':SOURCE,'service_source_files':{k:'2'*64 for k in bench.SERVICE_FILES},
            'checkpoint':bench.CHECKPOINT,'checkpoint_revision':bench.CHECKPOINT_REVISION,
            'packages':{k:'1.0' for k in bench.PACKAGES}}
    manifest={'schema_version':1,'harness_version':bench.HARNESS_VERSION,'bench_fixture_version':5,'profiles':PROFILES,'pairs':5,
              'lengths':list(bench.DEFAULT_LENGTHS),'output':128,'harness_source_files':dict(FAKE_HARNESS),'arms':{},
              'launcher_files':dict(FAKE_LAUNCHER),'windows_reserve_bytes':bench.WINDOWS_RESERVE_BYTES,
              'schedule':matrix.make_schedule(PROFILES,5),'tool_fixture':fx.describe(),
              'tool_pilot':{'name':'fixture-pilot','report_sha256':'8'*64,'profile':'bf128','repeats':3,'run_id':'9'*32}}
    for arm in PROFILES:
        config={'context':131072,'chunk':2048,'memory_ratio':.73,'flags':[],'env':{}};libs={}
        if arm!='bf128':
            config['env']['QWEN36_FUSED_ATTN_GATE_LIBRARY']='/private/gate.so'
            libs={'QWEN36_FUSED_ATTN_GATE_LIBRARY':{'path':'/private/gate.so','sha256':'4'*64}}
        manifest['arms'][arm]=dict(common,configuration=config,native_libraries=libs)
    return manifest


def receipt_for(manifest,arm,run_id):
    return dict(copy.deepcopy(manifest['arms'][arm]),source_files=dict(FILES),profile=arm,run_id=run_id)


def launcher_doc(env=None,problems=()):
    env=dict(LAUNCHER_ENV if env is None else env)
    return {'schema_version':1,'files':dict(FAKE_LAUNCHER),'environment':env,'environment_sha256':bench.canonical_hash(env),
            'cmdline':['/home/USER/qwen36-api/.venv/bin/python','/home/USER/qwen36-api/optimization/experiment.py','x'],
            'main_pid':4242,'invocation_id':'inv','problems':list(problems)}


def tool_steps(run_id,elapsed):
    steps=[]
    for index in range(2):
        steps.append({'status':200,'engine_metrics_status':'joined','serving_run_id':run_id,'engine':{'run_id':run_id},
            'peaks':{'gpu_total_used_bytes':12*GIB,'min_gpu_free_bytes':GIB},'cached_prompt_tokens':0,'cache_hit_observed':False,
            'usage':{'prompt_tokens':300+index*40,'completion_tokens':30-index*20,'total_tokens':330+index*20},
            'sampling':{**fx.SAMPLING,'max_tokens':fx.STEP_LIMITS[index],'thinking':False,'ignore_eos':False},
            'reasoning_chars':0,'finish_reason':'tool_calls' if index==0 else 'stop','elapsed_s':elapsed})
    return steps


def fixture(root,name='fixture'):
    """Synthetic complete V6 matrix: baseline gets ZERO cached tokens on every history task."""
    (root/'reports').mkdir(exist_ok=True)
    manifest=make_manifest();matrix_rows=[]
    for plan in manifest['schedule']:
        arm,repeat=plan['profile'],plan['repeat'];stem=f'{name}-{repeat}-{arm}'
        run_id=hashlib.sha256(stem.encode()).hexdigest()[:32];run=receipt_for(manifest,arm,run_id);rows=[]
        for case,length in sorted(bench.expected_cases(bench.DEFAULT_LENGTHS)):
            cached=0 if case=='fresh' or arm=='bf128' else 4096
            rows.append({'case':case,'length':length,'repeat':repeat,'status':200,'cache_requirement':bench.cache_requirement(case),
                'engine_metrics_status':'joined','serving_run_id':run_id,
                'engine':{'run_id':run_id,'prefill_gpu_ms':3.,'decode_gpu_ms':4.,'engine_span_s':1.},
                'peaks':{'gpu_total_used_bytes':12*GIB,'min_gpu_free_bytes':GIB},'messages_sha256':sha(f'{repeat}-{case}-{length}'),
                'request_fixture_sha256':sha(f'r{repeat}-{case}-{length}'),'content_sha256':sha(f'c{repeat}-{case}-{length}'),
                'usage':{'prompt_tokens':length+80,'completion_tokens':128,'total_tokens':length+208},
                'sampling':{'seed':100+repeat,'temperature':.6,'top_p':.95,'max_tokens':128,'ignore_eos':True,'thinking':False},
                'cached_prompt_tokens':cached,'cache_hit_observed':cached>0,
                'reasoning_chars':0,'finish_reason':'length','ttft_content_s':.3 if arm=='bf128' else .24,
                'elapsed_s':1. if arm=='bf128' else .8,'decode_client_tps':100. if arm=='bf128' else 125.})
            if case in ('last_message_edit','tool_result_edit'):
                primer=copy.deepcopy(rows[-1]);primer['cached_prompt_tokens']=0;primer['cache_hit_observed']=False
                rows[-1]['primer']=primer
        report={'name':stem,'run':run,'launcher':launcher_doc(),
                'benchmark':{'schema_version':1,'harness_version':bench.HARNESS_VERSION,'fixture_version':5,'output':128,
                'lengths':list(bench.DEFAULT_LENGTHS),'repeat_start':repeat,'repeats':1,'source_sha256':'3'*64},
                'rows':[{'case':'warmup','status':200}]+rows}
        (root/'reports'/f'{stem}.json').write_text(json.dumps(report))
        steps=tool_steps(run_id,.5 if arm=='bf128' else .4)
        task={'name':stem,'mode':'matrix','run':run,'schema_version':2,'repeat':repeat,'status':'complete',
              'fixture':fx.describe(),'source_sha256':'3'*64,'failure_code':None,'failure_codes':[],
              'row':{'case':'tool_cycle','repeat':repeat,'mode':'matrix','fixture_id':fx.FIXTURE_ID,'fixture_version':5,
                     'fixture_sha256':fx.fixture_sha256(),'task_fixture_sha256':fx.task_fixture_sha256(),
                     'correct':True,'failure_code':None,'failure_codes':[],'tool_result_matches_call_id':True,'safe_tool_result':42,
                     'returned_call_id_sha256':'5'*64,'second_history_normalized_sha256':'7'*64,
                     'steps':steps,'task_elapsed_s':sum(s['elapsed_s'] for s in steps)}}
        (root/'reports'/f'{stem}-tool-task.json').write_text(json.dumps(task))
        launcher=launcher_doc()
        matrix_rows.append(dict(plan,status='complete',run_id=run_id,harness_version=bench.HARNESS_VERSION,arm_verdict='VALID',
            launcher={key:launcher[key] for key in ('files','environment_sha256','problems','main_pid','invocation_id')},
            vram={'windows_reserve_bytes':bench.WINDOWS_RESERVE_BYTES,'process_peak_bytes':9*GIB,'budget_bytes':15*GIB,
                  'peak_gpu_used_bytes':12*GIB,'min_gpu_free_bytes':GIB,'gate':'PASS','code':None,
                  'pre_start_gpu':{'used_bytes':3*GIB,'free_bytes':13*GIB,'total_bytes':16*GIB},
                  'foreign_gpu_used_bytes_pre_start':3*GIB}))
    (root/'reports'/f'{name}-manifest.json').write_text(json.dumps(manifest))
    (root/'reports'/f'{name}-matrix.json').write_text(json.dumps(matrix_rows))
    return manifest


def compare(root,mod=None,**kw):return (mod or comparison).compare(root,'fixture','main-retain','bf128',verify_current=False,**kw)


def edit(root,stem,mutate):
    path=root/'reports'/f'{stem}.json';doc=json.loads(path.read_text());mutate(doc);path.write_text(json.dumps(doc))


# ---- mocked HTTP for the tool fixture -----------------------------------------------------
def call_message(call_id='call_fixture-add_0_1a2b3c4d',name='fixture_add',arguments='{"a": 17, "b": 25}',**extra):
    call={'id':call_id,'index':0,'type':'function','function':{'name':name,'arguments':arguments}};call.update(extra)
    return {'role':'assistant','content':'','tool_calls':[call]}


def final_message(content='Результат: 42.'):return {'role':'assistant','content':content}


def response(message=None,finish='tool_calls',status=200,usage=None,request_id='rid'):
    r=MagicMock();r.status_code=status;r.headers={'x-qwen36-request-id':request_id}
    r.json.return_value={'choices':[{'message':message,'finish_reason':finish}],
                         'usage':usage or {'prompt_tokens':300,'completion_tokens':25,'total_tokens':325}}
    return r


def cycle(first=None,second=None,first_finish='tool_calls',second_finish='stop'):
    return [response(first or call_message(),first_finish),response(second or final_message(),second_finish,request_id='rid2')]


class Http:
    """Context: mocked POST + no metrics files + no key file."""
    def __init__(self,responses):self.responses=list(responses);self.payloads=[]
    def post(self,url,headers=None,json=None,timeout=None):
        self.payloads.append(copy.deepcopy(json))
        item=self.responses.pop(0)
        if isinstance(item,Exception):raise item
        return item
    def __enter__(self):
        self.patches=[patch.object(tool_task.requests,'post',side_effect=self.post),
                      patch.object(tool_task,'headers',return_value={}),
                      patch.object(tool_task,'attach_metrics',side_effect=lambda step:step.update(engine_metrics_status='joined'))]
        for item in self.patches:item.start()
        return self
    def __exit__(self,*exc):
        for item in self.patches:item.stop()


def quiet(function,*args,**kw):
    with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):return function(*args,**kw)


# =============================================================================================
class V4Adapted(unittest.TestCase):
    """The 13 V4 contract checks, adapted to V5 semantics."""
    def test_v4_01_valid_contract_and_distinct_candidate_libraries(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);fixture(root);result=compare(root)
        self.assertTrue(result['performance_gate_pass'],result['failures'])
        self.assertEqual(result['verdict'],'PASS')
        self.assertEqual(len(result['rows']),24)
        self.assertEqual(result['gain_comparison_family_size'],18)

    def test_v4_02_all_old_false_accepts_rejected(self):
        defects=('output64','stale_engine','stale_http','service_hash','library_hash','profile','repeat','missing_case',
                 'duplicate_case','reasoning','benchmark_hash','teacher')
        for defect in defects:
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as folder:
                root=pathlib.Path(folder);fixture(root)
                def mutate(doc):
                    row=doc['rows'][1]
                    if defect=='output64':row['usage']['completion_tokens']=row['sampling']['max_tokens']=64
                    elif defect=='stale_engine':row['engine']['run_id']='a'*32
                    elif defect=='stale_http':row['serving_run_id']='a'*32
                    elif defect=='service_hash':doc['run']['service_source_files']['guard.py']='5'*64
                    elif defect=='library_hash':doc['run']['native_libraries']['QWEN36_FUSED_ATTN_GATE_LIBRARY']['sha256']='5'*64
                    elif defect=='profile':doc['run']['profile']='bf128'
                    elif defect=='repeat':row['repeat']=99
                    elif defect=='missing_case':doc['rows'].pop()
                    elif defect=='duplicate_case':doc['rows'].append(copy.deepcopy(row))
                    elif defect=='reasoning':row['reasoning_chars']=1
                    elif defect=='benchmark_hash':doc['benchmark']['source_sha256']='5'*64
                    elif defect=='teacher':doc['run']['configuration']['env']['QWEN36_TEACHER_FIXTURE']='/private/teacher.json'
                edit(root,'fixture-2-main-retain',mutate);result=compare(root)
                self.assertFalse(result['performance_gate_pass']);self.assertFalse(result['contract_pass'])
                self.assertNotEqual(result['verdict'],'PASS')

    def test_v4_03_wrong_expected_dimensions_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);fixture(root)
            for kw in ({'output':64},{'lengths':(64,)}):
                result=compare(root,**kw)
                self.assertFalse(result['performance_gate_pass']);self.assertEqual(result['verdict'],'INVALID')

    def test_v4_04_manifest_resume_idempotent_and_refuses_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path=pathlib.Path(folder)/'manifest.json'
            matrix.freeze_manifest(path,{'output':128,'harness_version':5});original=path.read_bytes()
            matrix.freeze_manifest(path,{'output':128,'harness_version':5},True)
            with self.assertRaises(RuntimeError):matrix.freeze_manifest(path,{'output':64,'harness_version':5},True)
            self.assertEqual(path.read_bytes(),original)
            with self.assertRaises(RuntimeError):matrix.freeze_manifest(path.with_name('missing.json'),{},True)

    def test_v4_05_three_arm_positions_and_two_arm_alternation(self):
        schedule=matrix.make_schedule(['a','b','c'],5)
        for arm in ('a','b','c'):
            positions=[row['pair_position'] for row in schedule if row['profile']==arm]
            self.assertTrue(all(1<=positions.count(position)<=2 for position in (0,1,2)))
        self.assertEqual([row['profile'] for row in matrix.make_schedule(['a','b'],3)],['a','b','b','a','a','b'])

    def test_v4_06_uncertain_regression_does_not_pass_on_mean(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);fixture(root)
            for repeat,ratio in enumerate((.9,.96,1.,1.04,1.08)):
                edit(root,f'fixture-{repeat}-main-retain',lambda doc,ratio=ratio:doc['rows'][1].update(elapsed_s=1/ratio))
            result=compare(root)
        self.assertTrue(result['contract_pass'])
        self.assertFalse(result['no_regression_over_3pct_full_request_975'])
        self.assertFalse(result['performance_gate_pass'])
        self.assertEqual(result['verdict'],'INCONCLUSIVE')
        self.assertIn('noninferiority_not_established',result['verdict_reasons'])

    def test_v4_07_t_interval_scale_and_bad_values(self):
        stats=comparison.summarize([1.2]*5,18)
        self.assertTrue(all(abs(stats[key]-1.2)<1e-12 for key in ('geomean','lower_975','upper_975','gain_lower_bonferroni')))
        for values in ([1]*4,[1,1,1,1,float('nan')],[1,1,1,1,0],[1,1,1,1,-1],[1]*6):
            with self.assertRaises(ValueError):comparison.summarize(values)

    def test_v4_08_teacher_and_profiler_configs_rejected(self):
        for key in ('QWEN36_TEACHER_FIXTURE','QWEN36_LOGIT_REFERENCE','QWEN36_NSYS_FORWARDS','QWEN36_KV_CALIBRATE'):
            with self.assertRaises(ValueError):bench.validate_speed_config({'context':1,'chunk':1,'flags':[],'env':{key:'on'}})
        with self.assertRaises(ValueError):bench.validate_speed_config({'context':1,'chunk':1,'flags':['--profiler'],'env':{}})

    def test_v4_09_failed_response_does_not_read_or_store_raw_error_body(self):
        response400=MagicMock();response400.status_code=400;response400.__enter__.return_value=response400
        with patch.object(bench,'headers',return_value={}),patch.object(bench.requests,'post',return_value=response400):
            result,_=bench.generate([],128)
        response400.json.assert_not_called();response400.iter_lines.assert_not_called()
        self.assertEqual(result,{'status':400,'error_code':'http_request_failed'})
        failed=response(status=400)
        with Http([failed]):row=tool_task.execute(0)
        failed.json.assert_not_called()
        self.assertEqual(row['failure_codes'],['http_status']);self.assertNotIn('usage',row['steps'][0])

    def test_v4_10_fixed_edited_history_uses_distinct_primers_and_auto_or_none(self):
        fixtures=list(bench.edited_fixtures('synthetic',2))
        self.assertEqual([x[0] for x in fixtures],['last_message_edit','tool_result_edit'])
        for case,old,new,tools,choice in fixtures:
            self.assertNotEqual(old,new);self.assertIn(choice,(None,'none'))
            self.assertEqual(bench.canonical_hash(new),bench.canonical_hash(list(bench.edited_fixtures('synthetic',2))[0 if case=='last_message_edit' else 1][2]))

    def test_v4_11_tool_cycle_actual_tokens_and_correctness_required(self):
        defects=('tokens','correct','run','input','second','source','sampling','time','fixture','mode','codes')
        for defect in defects:
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as folder:
                root=pathlib.Path(folder);fixture(root)
                def mutate(doc):
                    row=doc['row']
                    if defect=='tokens':row['steps'][0]['usage']['completion_tokens']+=1;row['steps'][0]['usage']['total_tokens']+=1
                    elif defect=='correct':row['correct']=False
                    elif defect=='run':row['steps'][0]['engine']['run_id']='a'*32
                    elif defect=='input':row['task_fixture_sha256']='a'*64
                    elif defect=='second':row['second_history_normalized_sha256']='a'*64
                    elif defect=='source':doc['source_sha256']='a'*64
                    elif defect=='sampling':row['steps'][0]['sampling']['temperature']=.5
                    elif defect=='time':row['task_elapsed_s']=.000001
                    elif defect=='fixture':doc['fixture']=dict(doc['fixture'],sha256='a'*64)
                    elif defect=='mode':doc['mode']='pilot'
                    elif defect=='codes':row['failure_codes']=['bad_finish']
                edit(root,'fixture-1-main-retain-tool-task',mutate);result=compare(root)
                self.assertFalse(result['performance_gate_pass']);self.assertEqual(result['verdict'],'INVALID')

    def test_v4_12_tool_call_validation_and_id_normalization(self):
        message=call_message('abc')
        codes,call,value=fx.validate_first(message,'tool_calls',0)
        self.assertEqual((codes,value),([],42))
        history=[message,{'role':'tool','tool_call_id':call['id'],'content':'{"result":42}'}]
        normalized=tool_task.normalized_history(history)
        self.assertEqual(normalized[0]['tool_calls'][0]['id'],normalized[1]['tool_call_id'])
        self.assertEqual(message['tool_calls'][0]['id'],'abc')
        message['tool_calls'][0]['function']['arguments']='{"a":true,"b":25}'
        codes,call,value=fx.validate_first(message,'tool_calls',0)
        self.assertEqual(codes,['bad_args']);self.assertIsNone(call);self.assertIsNone(value)

    def test_v4_13_cgroup_peak_rejects_other_unit_and_changed_invocation(self):
        with patch.object(matrix,'state',side_effect=['abc','/user.slice/other.service']):
            self.assertIsNone(matrix.cgroup_peak('abc'))
        with patch.object(matrix,'state',return_value='different'):
            self.assertIsNone(matrix.cgroup_peak('abc'))


# =============================================================================================
class V5Continuation(unittest.TestCase):
    def test_history_continuation_zero_cache_is_valid_and_not_a_hit(self):
        self.assertIn(('history_continuation',4096),bench.expected_cases(bench.DEFAULT_LENGTHS))
        self.assertNotIn('prefix',{case for case,_ in bench.expected_cases(bench.DEFAULT_LENGTHS)})
        self.assertEqual(bench.cache_requirement('fresh'),'must_be_zero')
        self.assertEqual(bench.cache_requirement('history_continuation'),'observed_not_required')
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);fixture(root);result=compare(root)
        self.assertEqual(result['verdict'],'PASS')
        history=[o for o in result['cache_observations'] if o['case']=='history_continuation']
        self.assertEqual(len(history),5)
        for item in history:
            self.assertEqual((item['baseline_cached_tokens'],item['baseline_hit']),(0,False))
            self.assertEqual((item['candidate_cached_tokens'],item['candidate_hit']),(4096,True))
            self.assertEqual(item['cache_requirement'],'observed_not_required')
        self.assertEqual(len(result['cache_observations']),5*8)
        self.assertEqual(len(result['tool_cache_observations']),5*2*2)

    def test_cache_label_and_requirement_defects(self):
        cases={'zero_labelled_hit':('history_continuation','INVALID',lambda r:r.update(cached_prompt_tokens=0,cache_hit_observed=True)),
               'fresh_hit':('fresh','INVALID',lambda r:r.update(cached_prompt_tokens=5,cache_hit_observed=True)),
               'requirement':('history_continuation','INVALID',lambda r:r.update(cache_requirement='must_hit')),
               'missing_count':('history_continuation','INCONCLUSIVE',lambda r:r.pop('cached_prompt_tokens')),
               'v4_case_name':('history_continuation','INVALID',lambda r:r.update(case='prefix'))}
        for label,(case,verdict,change) in cases.items():
            with self.subTest(label=label),tempfile.TemporaryDirectory() as folder:
                root=pathlib.Path(folder);fixture(root)
                def mutate(doc):change(next(r for r in doc['rows'] if r.get('case')==case))
                edit(root,'fixture-3-bf128',mutate);result=compare(root)
                self.assertEqual(result['verdict'],verdict,result['failures']);self.assertFalse(result['performance_gate_pass'])

    def test_attach_metrics_observed_hit_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);(root/'logs').mkdir();(root/'reports').mkdir()
            values={'zero':0,'hit':4096,'bad':'4096'}
            (root/'logs/requests.jsonl').write_text(''.join(json.dumps({'request_id':key,'engine_uid':i,'run_id':'r'*32,
                'cached_prompt_tokens':value,'peaks':{'x':1}})+'\n' for i,(key,value) in enumerate(values.items())))
            (root/'reports/engine.jsonl').write_text(''.join(json.dumps({'uid':i,'run_id':'r'*32,'pid':1})+'\n' for i in range(3)))
            out={}
            with patch.object(bench,'ROOT',root),patch.object(bench.time,'sleep'):
                for key in values:out[key]={'request_id':key};bench.attach_metrics(out[key])
        self.assertIs(out['zero']['cache_hit_observed'],False);self.assertIs(out['hit']['cache_hit_observed'],True)
        self.assertNotIn('cache_hit_observed',out['bad'])
        self.assertTrue(all(out[key]['engine_metrics_status']=='joined' for key in values))


# =============================================================================================
class V5ToolFixture(unittest.TestCase):
    def test_fixture_identity_auto_only_and_v4_retired_as_fail(self):
        self.assertEqual(fx.FIXTURE_VERSION,5);self.assertEqual(fx.TOOL_CHOICE,'auto')
        self.assertEqual([m['role'] for m in fx.first_messages()],['user'])
        self.assertIn('fixture_add',fx.first_messages()[0]['content'])
        copy_=fx.first_messages();copy_[0]['content']='x';self.assertNotEqual(fx.first_messages(),copy_)
        self.assertEqual(fx.fixture_sha256(),fx.fixture_sha256())
        retired=fx.RETIRED_FIXTURES['tool-cycle-en-v4']
        self.assertEqual(retired['status'],'FAIL');self.assertIs(retired['inherits_pass'],False)
        self.assertEqual({k:retired['observed'][k] for k in V4_ENGLISH_DIAGNOSTIC},V4_ENGLISH_DIAGNOSTIC)
        self.assertNotIn(fx.task_fixture_sha256(),retired['task_fixture_sha256_by_repeat'])
        diagnostic=HERE/'tool-task-v4-diagnostic.json'
        if diagnostic.exists():
            doc=json.loads(diagnostic.read_text())
            self.assertEqual((doc['first_finish'],doc['call_count'],doc['usage']['completion_tokens']),('stop',0,3))
        with Http(cycle()) as http:tool_task.execute(0)
        self.assertEqual([p['tool_choice'] for p in http.payloads],['auto','auto'])
        self.assertTrue(all(p['tools']==fx.TOOLS and p['temperature']==0 and p['stream'] is False
                            and p['chat_template_kwargs']=={'enable_thinking':False} for p in http.payloads))
        self.assertEqual([p['max_tokens'] for p in http.payloads],list(fx.STEP_LIMITS))

    def test_static_failure_codes_per_check(self):
        T='tool_calls'
        first=[('no_call',final_message('42'),'stop'),
               ('wrong_count',{'role':'assistant','tool_calls':call_message()['tool_calls']*2},T),
               ('bad_type',call_message(type='tool'),T),
               ('bad_id',call_message(''),T),('bad_id',call_message('call 1'),T),('bad_id',call_message(5),T),
               ('bad_name',call_message(name='fixture_sub'),T),
               ('bad_json',call_message(arguments='{"a":17,'),T),
               ('bad_json',call_message(arguments={'a':17,'b':25}),T),
               ('bad_json',call_message(arguments='{"a":17,"a":17,"b":25}'),T),
               ('bad_json',call_message(arguments='{"a":NaN,"b":25}'),T),
               ('bad_args',call_message(arguments='{"a":"17","b":25}'),T),
               ('bad_args',call_message(arguments='{"a":true,"b":25}'),T),
               ('bad_args',call_message(arguments='{"a":17.0,"b":25}'),T),
               ('bad_args',call_message(arguments='{"a":17,"b":25,"c":1}'),T),
               ('bad_args',call_message(arguments='{"a":17,"b":26}'),T),
               ('bad_args',call_message(arguments='[17,25]'),T),
               ('bad_finish',call_message(),'stop'),
               ('bad_response_schema',{'role':'user'},T),('bad_response_schema',None,T)]
        for index,(expected,message,finish) in enumerate(first):
            with self.subTest(index=index,expected=expected):
                codes,call,value=fx.validate_first(message,finish,0)
                self.assertIn(expected,codes);self.assertTrue(set(codes)<=set(fx.FAILURE_CODES))
                self.assertEqual(call is None,expected!='bad_finish');self.assertEqual(value is None,expected!='bad_finish')
        codes,_,_=fx.validate_first(call_message(),'tool_calls',7);self.assertEqual(codes,['reasoning_present'])
        self.assertEqual(fx.validate_final(final_message(),'stop',0),[])
        self.assertEqual(fx.validate_final(final_message('17 + 25 = 42'),'stop',0),[])
        final={'final_missing_42':(final_message('Результат: 41'),'stop',0),'final_missing_420':(final_message('420'),'stop',0),
               'final_missing_none':({'role':'assistant','content':None},'stop',0),
               'final_unexpected_tool_call':(dict(final_message(),tool_calls=call_message()['tool_calls']),'stop',0),
               'final_bad_finish':(final_message(),'length',0),'final_reasoning_present':(final_message(),'stop',3),
               'final_bad_response_schema':(None,'stop',0)}
        for label,(message,finish,reasoning) in final.items():
            with self.subTest(label=label):
                codes=fx.validate_final(message,finish,reasoning)
                expected='final_missing_42' if label.startswith('final_missing') else label
                self.assertIn(expected,codes);self.assertTrue(set(codes)<=set(fx.FAILURE_CODES))
        self.assertEqual(fx.ordered(['bad_finish','no_call','bad_finish']),['no_call','bad_finish'])
        with self.assertRaises(RuntimeError):fx.ordered(['ValueError'])

    def test_execute_success_cycle_hashes_only(self):
        with Http(cycle()) as http:row=tool_task.execute(0)
        self.assertTrue(row['correct'],row['failure_codes']);self.assertEqual(row['failure_codes'],[])
        self.assertEqual(len(row['steps']),2);self.assertEqual(row['safe_tool_result'],42)
        self.assertIs(row['tool_result_matches_call_id'],True)
        self.assertAlmostEqual(row['task_elapsed_s'],sum(s['elapsed_s'] for s in row['steps']))
        tool_message=http.payloads[1]['messages'][-1]
        self.assertEqual(tool_message,{'role':'tool','tool_call_id':'call_fixture-add_0_1a2b3c4d','content':'{"result":42}'})
        self.assertEqual(http.payloads[1]['messages'][1],call_message())
        with Http(cycle(first=call_message('call_fixture-add_0_ffffffff'))):other=tool_task.execute(1)
        self.assertEqual(row['second_history_normalized_sha256'],other['second_history_normalized_sha256'])
        self.assertNotEqual(row['returned_call_id_sha256'],other['returned_call_id_sha256'])
        text=json.dumps(row,ensure_ascii=False)
        for secret in ('Результат','Вызови','"a": 17','call_fixture-add_0_1a2b3c4d'):self.assertNotIn(secret,text)

    def test_execute_failure_rows_are_always_built(self):
        v4_like=response(final_message('42'),'stop',usage={'prompt_tokens':337,'completion_tokens':3,'total_tokens':340})
        scenarios={'no_call':([v4_like],'no_call',1),
                   'http_status':([response(status=500)],'http_status',1),
                   'transport_error':([requests.ConnectionError('down')],'transport_error',1),
                   'bad_json':([response(call_message(arguments='{oops'))],'bad_json',1),
                   'bad_finish':(cycle(first_finish='stop'),'bad_finish',2),
                   'final_http_status':([response(call_message()),response(status=503)],'final_http_status',2),
                   'final_transport_error':([response(call_message()),requests.Timeout('slow')],'final_transport_error',2),
                   'final_missing_42':(cycle(second=final_message('Готово.')),'final_missing_42',2),
                   'final_unexpected_tool_call':(cycle(second=call_message('call_x_1_aaaaaaaa'),second_finish='tool_calls'),'final_unexpected_tool_call',2)}
        for label,(responses,code,steps) in scenarios.items():
            with self.subTest(label=label),Http(responses):
                row=tool_task.execute(0)
                self.assertFalse(row['correct']);self.assertEqual(row['failure_code'],code)
                self.assertEqual(len(row['steps']),steps);self.assertIn(code,fx.FAILURE_CODES)
                self.assertEqual(row['task_fixture_sha256'],fx.task_fixture_sha256())
        self.assertEqual(scenarios['no_call'][0][0].json.call_count,1)

    def test_matrix_task_report_on_failure_and_never_overwrites(self):
        manifest=make_manifest();receipt=receipt_for(manifest,'bf128','b'*32)
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);(root/'reports').mkdir()
            with patch.object(tool_task,'run_receipt',return_value=receipt),Http([response(final_message('42'),'stop')]):
                code=quiet(tool_task.matrix_task,root,'arm-0-bf128',0)
            doc=json.loads((root/'reports/arm-0-bf128-tool-task.json').read_text())
            self.assertEqual(code,1);self.assertEqual(doc['status'],'failed');self.assertEqual(doc['failure_code'],'no_call')
            self.assertEqual(doc['row']['failure_codes'],['no_call','bad_finish']);self.assertEqual(doc['fixture'],fx.describe())
            with patch.object(tool_task,'run_receipt',return_value=receipt),Http(cycle()):
                with self.assertRaises(FileExistsError):quiet(tool_task.matrix_task,root,'arm-0-bf128',0)
            with patch.object(tool_task,'run_receipt',return_value=receipt),Http(cycle()):
                self.assertEqual(quiet(tool_task.matrix_task,root,'arm-1-bf128',1),0)
            self.assertEqual(json.loads((root/'reports/arm-1-bf128-tool-task.json').read_text())['status'],'complete')
            with patch.object(tool_task,'run_receipt',return_value={'source_receipt':'not_recorded_in_earlier_run'}),Http([]) as http:
                self.assertEqual(quiet(tool_task.matrix_task,root,'arm-2-bf128',2),1)
            self.assertEqual(http.payloads,[])
            self.assertEqual(json.loads((root/'reports/arm-2-bf128-tool-task.json').read_text())['failure_codes'],['receipt_invalid'])

    def test_pilot_pass_fail_profile_and_generation_guards(self):
        manifest=make_manifest();receipt=receipt_for(manifest,'bf128','b'*32)
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);(root/'reports').mkdir()
            with patch.object(tool_task,'run_receipt',return_value=receipt),Http(cycle()*3):
                self.assertEqual(quiet(tool_task.pilot,root,'p-pass',3,'bf128'),0)
            doc=json.loads((root/'reports/p-pass-tool-pilot.json').read_text())
            self.assertEqual((doc['status'],doc['passes'],doc['mode'],doc['repeats']),('PASS',3,'pilot',3))
            self.assertTrue(doc['first_step_identical_across_repeats']);self.assertEqual(doc['retired_fixtures']['tool-cycle-en-v4']['status'],'FAIL')
            with patch.object(tool_task,'run_receipt',return_value=receipt),Http(cycle()+[response(final_message('42'),'stop')]+cycle()):
                self.assertEqual(quiet(tool_task.pilot,root,'p-fail',3,'bf128'),1)
            doc=json.loads((root/'reports/p-fail-tool-pilot.json').read_text())
            self.assertEqual((doc['status'],doc['passes']),('FAIL',2));self.assertEqual(doc['failure_code_counts'],{'no_call':1,'bad_finish':1})
            with patch.object(tool_task,'run_receipt',return_value=receipt),Http([]) as http:
                self.assertEqual(quiet(tool_task.pilot,root,'p-profile',3,'main-retain'),1)
            self.assertEqual(http.payloads,[])
            self.assertEqual(json.loads((root/'reports/p-profile-tool-pilot.json').read_text())['failure_codes'],['receipt_invalid'])
            later=dict(receipt,run_id='c'*32)
            with patch.object(tool_task,'run_receipt',side_effect=[receipt,later]),Http(cycle()*3):
                self.assertEqual(quiet(tool_task.pilot,root,'p-gen',3,'bf128'),1)
            self.assertIn('serving_generation_changed',json.loads((root/'reports/p-gen-tool-pilot.json').read_text())['failure_codes'])
            with self.assertRaises(FileExistsError):tool_task.pilot(root,'p-pass',3,'bf128')

    def test_bench_refuses_to_overwrite_receipt(self):
        manifest=make_manifest();receipt=receipt_for(manifest,'bf128','b'*32)
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);(root/'reports').mkdir();(root/'reports/old.json').write_text('{}')
            with patch.object(bench,'ROOT',root),patch.object(bench,'run_receipt',return_value=receipt),\
                 patch.object(sys,'argv',['bench.py','--name','old']),patch.object(bench,'generate',side_effect=AssertionError):
                with self.assertRaises(RuntimeError):bench.main()
            self.assertEqual((root/'reports/old.json').read_text(),'{}')


# =============================================================================================
class V5Lock(unittest.TestCase):
    def test_gpu_lock_exclusive_nonblocking_and_released(self):
        with tempfile.TemporaryDirectory() as folder:
            path=pathlib.Path(folder)/'run/gpu-exclusive.lock'
            fd=bench.acquire_gpu_lock(path)
            self.assertEqual(path.stat().st_mode&0o777,0o600);self.assertFalse(os.get_inheritable(fd))
            started=time.monotonic()
            with self.assertRaises(bench.GpuLockBusy):bench.acquire_gpu_lock(path)
            self.assertLess(time.monotonic()-started,1)
            probe=('import fcntl,os,sys\nfd=os.open(sys.argv[1],os.O_RDWR)\n'
                   'try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);print("acquired")\nexcept BlockingIOError:print("busy")')
            out=subprocess.run([sys.executable,'-I','-c',probe,str(path)],capture_output=True,text=True,timeout=30,
                               env={'CUDA_VISIBLE_DEVICES':'','PATH':os.environ.get('PATH','')}).stdout.strip()
            self.assertEqual(out,'busy')
            os.close(fd)
            again=bench.acquire_gpu_lock(path);os.close(again)
            self.assertEqual(path.read_bytes(),b'')

    def test_matrix_and_pilot_runner_exit_when_lock_busy_without_side_effects(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);(root/'reports').mkdir();lock=root/'gpu.lock';held=bench.acquire_gpu_lock(lock)
            try:
                # matrix.subprocess and pilot_runner.subprocess are the same module: patch it once.
                guards=[patch.object(matrix,'units_inactive',side_effect=AssertionError('must not probe units')),
                        patch.object(pilot_runner.matrix,'units_inactive',side_effect=AssertionError('must not probe units')),
                        patch.object(subprocess,'run',side_effect=AssertionError('must not run anything'))]
                for item in guards:item.start()
                try:
                    err=io.StringIO()
                    with redirect_stderr(err),redirect_stdout(io.StringIO()):
                        code=matrix.main(['--profiles',','.join(PROFILES),'--name','m1','--tool-pilot','p'],root=root,lock_path=lock)
                    self.assertEqual(code,75);self.assertEqual(json.loads(err.getvalue())['reason'],'gpu_lock_busy')
                    err=io.StringIO()
                    with redirect_stderr(err),redirect_stdout(io.StringIO()):
                        code=pilot_runner.main(['--name','p1'],root=root,lock_path=lock)
                    self.assertEqual(code,75);self.assertEqual(json.loads(err.getvalue())['reason'],'gpu_lock_busy')
                finally:
                    for item in reversed(guards):item.stop()
            finally:os.close(held)
            self.assertEqual(sorted(p.name for p in (root/'reports').iterdir()),[])


# =============================================================================================
class ScreenPairs(unittest.TestCase):
    def test_fewer_than_three_pairs_refused_before_the_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);(root/'reports').mkdir();lock=root/'gpu.lock'
            for pairs in ('2','1','0'):
                with self.subTest(pairs=pairs),self.assertRaisesRegex(ValueError,'>=3 pairs'):
                    matrix.main(['--profiles',','.join(PROFILES),'--pairs',pairs,'--name','m1','--tool-pilot','p'],root=root,lock_path=lock)
            self.assertFalse(lock.exists())
            held=bench.acquire_gpu_lock(lock)
            try:
                with redirect_stderr(io.StringIO()),redirect_stdout(io.StringIO()):
                    code=matrix.main(['--profiles',','.join(PROFILES),'--pairs','3','--name','m1','--tool-pilot','p'],root=root,lock_path=lock)
                self.assertEqual(code,75)   # three pairs pass the dimension check and reach the lock
            finally:os.close(held)

    def test_screen_flag_only_off_five_pairs(self):
        with patch.object(matrix,'current_run_spec',side_effect=lambda *args:{'arms':{},'harness_source_files':{}}),\
             patch.object(matrix,'validate_tool_pilot',return_value={}):
            three=matrix.build_spec(None,PROFILES,bench.DEFAULT_LENGTHS,128,3,'p')
            five=matrix.build_spec(None,PROFILES,bench.DEFAULT_LENGTHS,128,5,'p')
        self.assertIs(three['screen'],True);self.assertNotIn('screen',five)
        self.assertEqual(len(three['schedule']),3*len(PROFILES))

    def test_screen_summary_lists_every_pair_and_the_median(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);fixture(root)
            rows=[dict(row,report=f"fixture-{row['repeat']}-{row['profile']}.json")
                  for row in json.loads((root/'reports/fixture-matrix.json').read_text()) if row['repeat']<3]
            summary=matrix.screen_summary(root,'fixture',rows)
        self.assertTrue(summary['screen']);self.assertFalse(summary['gate'])
        self.assertEqual(sorted(summary['profiles']),sorted(PROFILES))
        cell=summary['profiles'][PROFILES[0]]['fresh_64']['elapsed_s']
        self.assertEqual((len(cell['pairs']),cell['repeats']),(3,[0,1,2]))
        self.assertEqual(cell['median'],sorted(cell['pairs'])[1])


# =============================================================================================
class V5MatrixResume(unittest.TestCase):
    def test_refuses_v4_manifest_stale_receipts_and_failed_workload(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);manifest=fixture(root)
            v4=copy.deepcopy(manifest);[v4.pop(k) for k in ('harness_version','bench_fixture_version','tool_fixture','tool_pilot')]
            path=root/'reports/final3-v4-manifest.json';path.write_text(json.dumps(v4));before=path.read_bytes()
            with self.assertRaisesRegex(RuntimeError,'harness version'):matrix.freeze_manifest(path,manifest,True)
            self.assertEqual(path.read_bytes(),before)
            args=types.SimpleNamespace(name='final3-v4',output=128,pairs=5,tool_pilot='p')
            with patch.object(matrix,'units_inactive',return_value=True),patch.object(matrix,'build_spec',return_value=manifest),\
                 patch.object(matrix,'workspace_config_drift',return_value=False),\
                 patch.object(matrix,'run',side_effect=AssertionError('must not start')):
                with self.assertRaisesRegex(RuntimeError,'harness version'):matrix.run_matrix(args,PROFILES,bench.DEFAULT_LENGTHS,root)
                # Fresh name colliding with an existing arm receipt is refused before any start.
                (root/'reports/stale-0-bf128.json').write_text('{}')
                with self.assertRaisesRegex(RuntimeError,'already exist'):
                    matrix.run_matrix(types.SimpleNamespace(name='stale',output=128,pairs=5,tool_pilot='p'),PROFILES,bench.DEFAULT_LENGTHS,root)
            self.assertFalse((root/'reports/stale-manifest.json').exists())
            plan0=manifest['schedule'][0]
            for step,allowed in (('tool_task',False),('bench',False),('receipt',False),('readiness',True),('start',True),('preflight',True)):
                rows=[dict(plan0,status='failed',failed_step=step)]
                with self.subTest(step=step):
                    if allowed:self.assertEqual(matrix.check_start_or_resume(root,'fresh',manifest,rows,bench.DEFAULT_LENGTHS,128),set())
                    else:
                        with self.assertRaisesRegex(RuntimeError,'cannot be resumed'):
                            matrix.check_start_or_resume(root,'fresh',manifest,rows,bench.DEFAULT_LENGTHS,128)
            completed=[dict(plan,status='complete') for plan in manifest['schedule']]
            self.assertEqual(matrix.check_start_or_resume(root,'fixture',manifest,completed,bench.DEFAULT_LENGTHS,128),
                             {(plan['repeat'],plan['profile']) for plan in manifest['schedule']})
            with self.assertRaisesRegex(RuntimeError,'Unaccounted'):
                matrix.check_start_or_resume(root,'fixture',manifest,completed[:1],bench.DEFAULT_LENGTHS,128)
            for label,stem,change in (('bench_v4',f'fixture-0-{plan0["profile"]}',lambda d:d['benchmark'].update(fixture_version=4)),
                                      ('tool_pilot_mode',f'fixture-0-{plan0["profile"]}-tool-task',lambda d:d.update(mode='pilot')),
                                      ('tool_fixture',f'fixture-0-{plan0["profile"]}-tool-task',lambda d:d.update(fixture={'version':4}))):
                with self.subTest(label=label):
                    path=root/'reports'/f'{stem}.json';original=path.read_text();edit(root,stem,change)
                    with self.assertRaises(RuntimeError):matrix.check_start_or_resume(root,'fixture',manifest,completed,bench.DEFAULT_LENGTHS,128)
                    path.write_text(original)

    def test_matrix_requires_passing_matching_pilot(self):
        manifest=make_manifest();arms=manifest['arms'];receipt=receipt_for(manifest,'bf128','b'*32)
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);(root/'reports').mkdir()
            with patch.object(tool_task,'run_receipt',return_value=receipt),Http(cycle()*3):quiet(tool_task.pilot,root,'pilot-ok',3,'bf128')
            with patch.object(tool_task,'run_receipt',return_value=receipt),Http(cycle()*2):quiet(tool_task.pilot,root,'pilot-two',2,'bf128')
            with patch.object(tool_task,'run_receipt',return_value=receipt),Http(cycle()*2+[response(final_message('42'),'stop')]):
                quiet(tool_task.pilot,root,'pilot-fail',3,'bf128')
            binding=matrix.validate_tool_pilot(root,'pilot-ok',PROFILES,arms,REAL_HARNESS)
            self.assertEqual((binding['profile'],binding['repeats'],binding['run_id']),('bf128',3,'b'*32))
            self.assertEqual(binding['report_sha256'],hashlib.sha256((root/'reports/pilot-ok-tool-pilot.json').read_bytes()).hexdigest())
            other_arms=copy.deepcopy(arms);other_arms['bf128']['configuration']['chunk']=4096
            path=root/'reports/pilot-ok-tool-pilot.json';doc=json.loads(path.read_text())
            (root/'reports/pilot-fx-tool-pilot.json').write_text(json.dumps(dict(doc,name='pilot-fx',fixture=dict(doc['fixture'],sha256='0'*64))))
            for name,profiles,arms_,harness,code in (('missing',PROFILES,arms,REAL_HARNESS,'tool_pilot_missing'),
                                                       ('pilot-fail',PROFILES,arms,REAL_HARNESS,'tool_pilot_not_passing'),
                                                       ('pilot-two',PROFILES,arms,REAL_HARNESS,'tool_pilot_not_passing'),
                                                       ('pilot-fx',PROFILES,arms,REAL_HARNESS,'tool_pilot_fixture_mismatch'),
                                                       ('pilot-ok',PROFILES,arms,dict(REAL_HARNESS,**{'tool-task-bench.py':'0'*64}),'tool_pilot_harness_mismatch'),
                                                       ('pilot-ok',['main-retain','main-retain2048'],arms,REAL_HARNESS,'tool_pilot_profile_not_in_matrix'),
                                                       ('pilot-ok',PROFILES,other_arms,REAL_HARNESS,'tool_pilot_runtime_mismatch')):
                with self.subTest(code=code,name=name):
                    with self.assertRaisesRegex(RuntimeError,code):matrix.validate_tool_pilot(root,name,profiles,arms_,harness)

    def test_matrix_serial_scoped_stop_and_failure_classification(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);spec=fixture(root)
            def harness(fail_tool_at=None,foreign_at=None,interrupt_at=None):
                calls=[];stops=[];current={'inv':'','count':0}
                def fake_run(args,timeout=1200):
                    calls.append(args)
                    if args[0]=='bash':
                        current['count']+=1;current['inv']=f'inv-{current["count"]}';profile=args[2]
                        (root/'active-run.json').write_text(json.dumps(receipt_for(spec,profile,f'{current["count"]:032x}')))
                    elif args[:3]==['systemctl','--user','stop']:stops.append(current['inv']);current['inv']=''
                    elif str(args[1]).endswith('bench.py') and current['count']==interrupt_at:
                        matrix.terminate_cleanly(15,None)
                    elif str(args[1]).endswith('tool-task-bench.py'):
                        if current['count']==foreign_at:current['inv']='foreign';raise RuntimeError('python: exit 1')
                        if current['count']==fail_tool_at:
                            (root/'reports'/f'{args[3]}-tool-task.json').write_text(json.dumps({'failure_codes':['no_call','bad_finish','ValueError']}))
                            raise RuntimeError('python: exit 1')
                    return ''
                def fake_state(prop,unit='qwen36-experiment'):
                    if prop=='InvocationID':return current['inv'] if unit=='qwen36-experiment' else ''
                    if prop=='ActiveState':return 'active' if current['inv'] and unit=='qwen36-experiment' else 'inactive'
                    return ''
                return calls,stops,fake_run,fake_state
            for name,fail_at,foreign_at,expected_rows,expected_rc in (('live',None,None,15,0),('toolfail',4,None,4,1),('foreign',None,2,2,1)):
                with self.subTest(name=name):
                    calls,stops,fake_run,fake_state=harness(fail_at,foreign_at)
                    args=types.SimpleNamespace(name=name,output=128,pairs=5,tool_pilot='p')
                    with patch.object(matrix,'run',side_effect=fake_run),patch.object(matrix,'state',side_effect=fake_state),\
                         patch.object(matrix,'build_spec',return_value=spec),patch.object(matrix,'cgroup_peak',return_value=None),\
                         patch.object(matrix,'workspace_config_drift',return_value=False),\
                         patch.object(bench,'gpu_memory_snapshot',return_value=None),\
                         patch.object(bench,'launcher_receipt',side_effect=lambda profile,root:dict(launcher_doc(),invocation_id=fake_state('InvocationID'))):
                        rc=quiet(matrix.run_matrix,args,PROFILES,bench.DEFAULT_LENGTHS,root)
                    rows=json.loads((root/'reports'/f'{name}-matrix.json').read_text())
                    self.assertEqual((rc,len(rows)),(expected_rc,expected_rows))
                    starts=[c for c in calls if c[0]=='bash']
                    self.assertEqual([r['profile'] for r in rows],[s[2] for s in starts])
                    self.assertEqual([r['profile'] for r in rows],[p['profile'] for p in spec['schedule'][:expected_rows]])
                    self.assertNotIn('foreign',stops)
                    if name=='live':
                        self.assertEqual(stops,[f'inv-{i}' for i in range(1,16)]);self.assertTrue(all(r['status']=='complete' for r in rows))
                    elif name=='toolfail':
                        self.assertEqual(stops,[f'inv-{i}' for i in range(1,5)])
                        self.assertEqual((rows[-1]['status'],rows[-1]['failed_step'],rows[-1]['tool_failure_codes']),('failed','tool_task',['no_call','bad_finish']))
                        with self.assertRaisesRegex(RuntimeError,'cannot be resumed'):
                            matrix.check_start_or_resume(root,name,spec,rows,bench.DEFAULT_LENGTHS,128)
                    else:
                        self.assertEqual(stops,['inv-1']);self.assertEqual(rows[-1]['failed_step'],'tool_task')
            # SIGTERM/SIGHUP during an arm: own unit stopped, arm recorded, exit propagates, no resume.
            calls,stops,fake_run,fake_state=harness(interrupt_at=3)
            args=types.SimpleNamespace(name='sigterm',output=128,pairs=5,tool_pilot='p')
            with patch.object(matrix,'run',side_effect=fake_run),patch.object(matrix,'state',side_effect=fake_state),\
                 patch.object(matrix,'build_spec',return_value=spec),patch.object(matrix,'cgroup_peak',return_value=None),\
                 patch.object(matrix,'workspace_config_drift',return_value=False),\
                 patch.object(bench,'gpu_memory_snapshot',return_value=None),\
                 patch.object(bench,'launcher_receipt',side_effect=lambda profile,root:dict(launcher_doc(),invocation_id=fake_state('InvocationID'))):
                with self.assertRaises(SystemExit) as caught:quiet(matrix.run_matrix,args,PROFILES,bench.DEFAULT_LENGTHS,root)
            self.assertEqual(caught.exception.code,143);self.assertEqual(stops,['inv-1','inv-2','inv-3'])
            rows=json.loads((root/'reports/sigterm-matrix.json').read_text())
            self.assertEqual((len(rows),rows[-1]['failed_step'],rows[-1]['interrupted']),(3,'bench',True))
            with self.assertRaisesRegex(RuntimeError,'cannot be resumed'):
                matrix.check_start_or_resume(root,'sigterm',spec,rows,bench.DEFAULT_LENGTHS,128)

    def test_managed_pilot_runner_scoped_stop_and_receipts(self):
        manifest=make_manifest()
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);(root/'reports').mkdir()
            def scenario(name,status='PASS',interrupt=False,foreign=False):
                current={'inv':''};stops=[]
                def fake_run(args,timeout=1200):
                    if args[0]=='bash':
                        current['inv']='inv-'+name;(root/'active-run.json').write_text(json.dumps(receipt_for(manifest,args[2],'d'*32)))
                    elif args[:3]==['systemctl','--user','stop']:stops.append(current['inv']);current['inv']=''
                    return ''
                def fake_state(prop,unit='qwen36-experiment'):
                    mine=unit=='qwen36-experiment'
                    if prop=='InvocationID':return current['inv'] if mine else ''
                    if prop=='ActiveState':return 'active' if mine and current['inv'] else 'inactive'
                    return ''
                def fake_subprocess(args,**kw):
                    self.assertEqual(args[2:4],['--pilot','--name']);self.assertIn('--expect-profile',args)
                    if interrupt:matrix.terminate_cleanly(1,None)
                    if foreign:current['inv']='foreign'
                    (root/'reports'/f'{name}-tool-pilot.json').write_text(json.dumps(
                        {'status':status,'passes':3 if status=='PASS' else 2,'failure_code_counts':{} if status=='PASS' else {'no_call':1},'run':{'run_id':'d'*32}}))
                    return types.SimpleNamespace(returncode=0 if status=='PASS' else 1,stdout='',stderr='')
                with patch.object(pilot_runner.matrix,'run',side_effect=fake_run),patch.object(pilot_runner.matrix,'state',side_effect=fake_state),\
                     patch.object(pilot_runner,'current_run_spec',return_value={'arms':{'bf128':manifest['arms']['bf128']}}),\
                     patch.object(subprocess,'run',side_effect=fake_subprocess):
                    try:rc=quiet(pilot_runner.run_pilot,root,'bf128',3,name)
                    except SystemExit as error:rc=('exit',error.code)
                return rc,stops,json.loads((root/'reports'/f'{name}-tool-pilot-runner.json').read_text())
            rc,stops,runner=scenario('ok')
            self.assertEqual((rc,stops,runner['status'],runner['stopped_own_invocation']),(0,['inv-ok'],'PASS',True))
            rc,stops,runner=scenario('bad','FAIL')
            self.assertEqual((rc,stops,runner['status'],runner['failure_code_counts']),(1,['inv-bad'],'FAIL',{'no_call':1}))
            rc,stops,runner=scenario('int',interrupt=True)
            self.assertEqual((rc,stops,runner['status'],runner['failed_step'],runner['interrupted']),(('exit',129),['inv-int'],'FAIL','pilot',True))
            rc,stops,runner=scenario('foreign',foreign=True)
            self.assertEqual(stops,[]);self.assertNotIn('stopped_own_invocation',runner)
            with self.assertRaisesRegex(RuntimeError,'already exist'):pilot_runner.run_pilot(root,'bf128',3,'ok')


# =============================================================================================
class V5Compare(unittest.TestCase):
    def test_v4_manifest_rejected_and_no_comparison_written(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);manifest=fixture(root)
            v4=copy.deepcopy(manifest);[v4.pop(k) for k in ('harness_version','bench_fixture_version','tool_fixture','tool_pilot')]
            (root/'reports/fixture-manifest.json').write_text(json.dumps(v4))
            result=compare(root);self.assertEqual(result['verdict'],'INVALID');self.assertIn('manifest_not_harness_v6',result['failures'])
            lock=root/'gpu.lock'
            with patch.object(bench,'ROOT',root),patch.object(comparison,'current_run_spec',return_value={}):
                code=quiet(comparison.main,['--name','fixture','--candidate','main-retain'],lock_path=lock)
            self.assertEqual(code,3);self.assertFalse((root/'reports/fixture-main-retain-vs-bf128-comparison.json').exists())
            (root/'reports/fixture-manifest.json').write_text(json.dumps(manifest))
            with patch.object(bench,'ROOT',root),patch.object(comparison,'current_run_spec',return_value={}):
                code=quiet(comparison.main,['--name','fixture','--candidate','main-retain'],lock_path=lock)
            path=root/'reports/fixture-main-retain-vs-bf128-comparison.json'
            self.assertEqual(code,0);self.assertEqual(json.loads(path.read_text())['verdict'],'PASS')
            self.assertEqual(path.stat().st_mode&0o777,0o600)

    def test_missing_pairs_reports_and_metrics_are_inconclusive(self):
        def drop(name):return lambda root:(root/'reports'/name).unlink()
        def metric(stem,change):return lambda root:edit(root,stem,change)
        scenarios={'missing_pair':drop('fixture-4-main-retain.json'),
                   'missing_tool_report':drop('fixture-2-bf128-tool-task.json'),
                   'missing_matrix':drop('fixture-matrix.json'),
                   'missing_manifest':drop('fixture-manifest.json'),
                   'engine_timeout':metric('fixture-1-main-retain',lambda d:d['rows'][2].update(engine_metrics_status='engine_row_timeout',engine={})),
                   'no_peaks':metric('fixture-1-bf128',lambda d:d['rows'][3].pop('peaks')),
                   'no_timing':metric('fixture-0-bf128',lambda d:d['rows'][1].update(elapsed_s=None)),
                   'no_gpu_used':metric('fixture-2-main-retain',lambda d:[r.get('peaks',{}).pop('gpu_total_used_bytes',None) for r in d['rows']]),
                   'tool_metrics':metric('fixture-3-main-retain-tool-task',lambda d:d['row']['steps'][1].update(engine_metrics_status='request_row_timeout',engine={}))}
        for label,damage in scenarios.items():
            with self.subTest(label=label),tempfile.TemporaryDirectory() as folder:
                root=pathlib.Path(folder);fixture(root);damage(root);result=compare(root)
                self.assertEqual(result['verdict'],'INCONCLUSIVE',[f for f in result['failures'] if not f.startswith('missing_')])
                self.assertFalse(result['performance_gate_pass']);self.assertTrue(any(f.startswith('missing_') for f in result['failures']))

    def test_wrong_run_source_library_output_and_tool_failure_invalid(self):
        scenarios={'wrong_run':('fixture-2-main-retain',lambda d:d['rows'][1].update(serving_run_id='f'*32)),
                   'wrong_source':('fixture-2-main-retain',lambda d:d['run'].update(source_sha256='f'*64)),
                   'wrong_library':('fixture-2-main-retain',lambda d:d['run']['native_libraries']['QWEN36_FUSED_ATTN_GATE_LIBRARY'].update(path='/other.so')),
                   'output_changed':('fixture-2-main-retain',lambda d:[r['usage'].update(completion_tokens=127,total_tokens=r['usage']['prompt_tokens']+127)
                                                                     or r['sampling'].update(max_tokens=127) for r in d['rows'][1:2]]),
                   'sampling_changed':('fixture-2-main-retain',lambda d:d['rows'][1]['sampling'].update(temperature=.7)),
                   'input_changed':('fixture-2-main-retain',lambda d:d['rows'][1].update(messages_sha256='f'*64)),
                   'bench_harness_v4':('fixture-2-main-retain',lambda d:d['benchmark'].update(harness_version=4)),
                   'tool_cycle_failed':('fixture-2-main-retain-tool-task',lambda d:d.update(status='failed',failure_codes=['no_call','bad_finish'])),
                   'matrix_row':('fixture-matrix',lambda d:d[0].pop('harness_version'))}
        for label,(stem,change) in scenarios.items():
            with self.subTest(label=label),tempfile.TemporaryDirectory() as folder:
                root=pathlib.Path(folder);fixture(root);edit(root,stem,change);result=compare(root)
                self.assertEqual(result['verdict'],'INVALID',result['failures']);self.assertFalse(result['performance_gate_pass'])
                if label=='tool_cycle_failed':
                    self.assertEqual(result['tool_failures'],[{'repeat':2,'arm':'main-retain','failure_codes':['no_call','bad_finish']}])

    def test_verdicts_pass_fail_regression_and_no_gain(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);fixture(root)
            for repeat in range(5):
                def slower(doc):
                    for row in doc['rows'][1:]:row.update(ttft_content_s=.3,elapsed_s=1.25,decode_client_tps=100.)
                edit(root,f'fixture-{repeat}-main-retain',slower)
            result=compare(root)
        self.assertEqual(result['verdict'],'FAIL');self.assertTrue(any(r.startswith('regression_over_3pct_established') for r in result['verdict_reasons']))
        def same_rows(doc):
            for row in doc['rows'][1:]:row.update(ttft_content_s=.3,elapsed_s=1.,decode_client_tps=100.)
        def same_tool(doc):
            for step in doc['row']['steps']:step['elapsed_s']=.5
            doc['row']['task_elapsed_s']=1.
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);fixture(root)
            for repeat in range(5):
                edit(root,f'fixture-{repeat}-main-retain',same_rows);edit(root,f'fixture-{repeat}-main-retain-tool-task',same_tool)
            result=compare(root)
        self.assertEqual(result['verdict'],'FAIL');self.assertIn('five_pct_gain_excluded',result['verdict_reasons'])
        self.assertTrue(result['no_regression_over_3pct_full_request_975']);self.assertFalse(result['performance_gate_pass'])


# =============================================================================================
# Harness V6: statistics (t-intervals, elapsed-only gate, Bonferroni), mutation kills C01-C14 + V6,
# compare main exit codes / O_EXCL / lock, launcher identity, VRAM floor, start-experiment.sh.
COMPARE_SOURCE=(HERE/'compare-ab.py').read_text(encoding='utf-8')
T975,T95,T995=2.7764451051977987,2.131846786326649,4.604094871415897
MUTATION_RESULTS={}


def t_closed(p):
    """Independent closed form of the df=4 quantile (trigonometric inverse), for cross-checks."""
    a=4*p*(1-p);q=math.cos(math.acos(math.sqrt(a))/3)/math.sqrt(a);return 2*math.sqrt(q-1)


def vec(mean,se):
    """Five paired ratios with geometric mean `mean` and log standard error `se`."""
    step=se*math.sqrt(5)/math.sqrt(2.5)
    return [math.exp(math.log(mean)+step*k) for k in (-2,-1,0,1,2)]


def apply_scenario(root,elapsed=None,default=None,tool=None,decode=None):
    """Candidate (main-retain) vs baseline (bf128, elapsed 1.0, tool 1.0): ratio r -> candidate time 1/r."""
    for repeat in range(5):
        def change(doc):
            for row in doc['rows'][1:]:
                ratios=(elapsed or {}).get((row['case'],row['length']))
                if ratios is None and default is not None:ratios=[default]*5
                if ratios is not None:row['elapsed_s']=1.0/ratios[repeat]
                if decode is not None:row['decode_client_tps']=100.*decode
        edit(root,f'fixture-{repeat}-main-retain',change)
        if tool is not None:
            def change_tool(doc):
                for step in doc['row']['steps']:step['elapsed_s']=.5/tool[repeat]
                doc['row']['task_elapsed_s']=sum(step['elapsed_s'] for step in doc['row']['steps'])
            edit(root,f'fixture-{repeat}-main-retain-tool-task',change_tool)


def set_launcher_env(root,stem,env,problems=()):
    edit(root,stem,lambda doc:doc.update(launcher=launcher_doc(env,problems)))


def edit_matrix_row(root,repeat,arm,change):
    path=root/'reports/fixture-matrix.json';rows=json.loads(path.read_text())
    for row in rows:
        if row['repeat']==repeat and row['profile']==arm:change(row)
    path.write_text(json.dumps(rows))


def over_budget(doc):
    # Synthetic: 3 GiB foreign before start, so the footprint passes total - 1 GiB only above 18 GiB used.
    doc['rows'][3]['peaks']['gpu_total_used_bytes']=18*GIB+1


# name -> (setup(root), expected verdict, expected reasons or None, expected failure code or None)
SCENARIOS={
    'pass':(lambda r:None,'PASS',['noninferior_all_gating_cases','gain_at_least_5pct_bonferroni'],None),
    'bench_gain_needs_bonferroni':(lambda r:apply_scenario(r,default=1.0,tool=[1.0]*5,elapsed={('fresh',65536):vec(1.15,.02)}),
                                   'INCONCLUSIVE',['gain_not_established'],None),
    'tool_gain_needs_bonferroni':(lambda r:apply_scenario(r,default=1.0,tool=vec(1.15,.02)),'INCONCLUSIVE',['gain_not_established'],None),
    'noninferiority_borderline':(lambda r:apply_scenario(r,elapsed={('fresh',4096):vec(1.0,.0112)}),
                                 'INCONCLUSIVE',['noninferiority_not_established'],None),
    'tool_regression':(lambda r:apply_scenario(r,tool=[.95]*5),'FAIL',['regression_over_3pct_established:tool_cycle_None_task_elapsed_s'],None),
    'bench_regression':(lambda r:apply_scenario(r,elapsed={('fresh',120000):[.95]*5}),'FAIL',
                        ['regression_over_3pct_established:fresh_120000_elapsed_s'],None),
    'gain_excluded':(lambda r:apply_scenario(r,default=1.04,tool=[1.04]*5),'FAIL',['five_pct_gain_excluded'],None),
    'gain_open_lower_below':(lambda r:apply_scenario(r,default=1.0,tool=[1.0]*5,elapsed={('history_continuation',4096):vec(1.06,.02)}),
                             'INCONCLUSIVE',['gain_not_established'],None),
    'decode_only_gain':(lambda r:apply_scenario(r,default=1.0,tool=[1.0]*5),'FAIL',['five_pct_gain_excluded'],None),
    'decode_regression_not_gating':(lambda r:apply_scenario(r,decode=.8),'PASS',None,None),
    'vram_over_budget':(lambda r:edit(r,'fixture-2-main-retain',over_budget),'INVALID',None,'process_vram_over_budget_2_main-retain'),
    'vram_tool_step_over_budget':(lambda r:edit(r,'fixture-1-bf128-tool-task',
                                  lambda d:d['row']['steps'][1]['peaks'].update(gpu_total_used_bytes=18*GIB+1)),
                                  'INVALID',None,'process_vram_over_budget_1_bf128'),
    'low_free_vram_within_budget':(lambda r:edit(r,'fixture-2-main-retain',
                                   lambda d:d['rows'][3]['peaks'].update(min_gpu_free_bytes=int(.1*GIB))),'PASS',None,None),
    'launcher_env_differs':(lambda r:(set_launcher_env(r,'fixture-3-main-retain',dict(LAUNCHER_ENV,OMP_NUM_THREADS='8')),
                                      edit_matrix_row(r,3,'main-retain',lambda row:row['launcher'].update(
                                          environment_sha256=bench.canonical_hash(dict(LAUNCHER_ENV,OMP_NUM_THREADS='8'))))),
                            'INVALID',None,'launcher_environment_differs_between_arms'),
    'launcher_problems':(lambda r:set_launcher_env(r,'fixture-0-bf128',LAUNCHER_ENV,['pythonpath']),'INVALID',None,'launcher_identity_0_bf128'),
    'matrix_arm_invalid':(lambda r:edit_matrix_row(r,4,'main-retain',lambda row:(row.update(arm_verdict='INVALID'),
                                                                                 row['vram'].update(gate='INVALID'))),
                          'INVALID',None,'matrix_arm_invalid_4_main-retain'),
}


def build_scenarios(base):
    roots={}
    for name,(setup,*_) in SCENARIOS.items():
        root=base/name;root.mkdir();fixture(root);setup(root);roots[name]=root
    return roots


def run_main(mod,root,argv=None,lock=None):
    argv=['--name','fixture','--candidate','main-retain'] if argv is None else argv
    with patch.object(bench,'ROOT',root),patch.object(mod,'current_run_spec',return_value={}):
        try:return quiet(mod.main,argv,lock_path=lock or root/'gpu.lock')
        except SystemExit as error:return ('exit',error.code)


def battery(mod,roots,work):
    """Every check returns a failure label; [] means the module implements the V6 contract."""
    bad=[]
    def check(label,condition):
        if not condition:bad.append(label)
    try:
        for p,value in ((.975,T975),(.95,T95),(.995,T995)):check(f't_quantile_{p}',abs(mod.t_quantile(p)-value)<1e-9)
        for p in (.6,.9,.99,1-.025/18,1-.025/27,1-.025/9):check(f't_closed_{p}',abs(mod.t_quantile(p)-t_closed(p))<1e-8)
        ratios=[1.10,1.05,1.12,1.08,1.07];logs=[math.log(x) for x in ratios];m=sum(logs)/5
        se=math.sqrt(sum((x-m)**2 for x in logs)/4/5);tg=t_closed(1-.025/18);stats=mod.summarize(ratios,18)
        expected={'geomean':math.exp(m),'lower_975':math.exp(m-T975*se),'upper_975':math.exp(m+T975*se),
                  'gain_lower_bonferroni':math.exp(m-tg*se)}
        for key,value in expected.items():check(f'summarize_{key}',abs(stats[key]-value)<1e-12*value)
        check('summarize_family_size',stats.get('gain_family_size')==18)
    except Exception as error:bad.append('stats_error_'+type(error).__name__)
    for name,root in roots.items():
        _,verdict,reasons,code=SCENARIOS[name]
        try:result=compare(root,mod)
        except Exception as error:bad.append(f'{name}:error_{type(error).__name__}');continue
        check(f'{name}:verdict',result['verdict']==verdict)
        if reasons is not None:check(f'{name}:reasons',result['verdict_reasons']==reasons)
        if code is not None:check(f'{name}:code',code in result['failures'])
        check(f'{name}:gate_flag',result['performance_gate_pass']==(verdict=='PASS'))
        if name=='pass':
            check('family_size',result['gain_comparison_family_size']==18)
            check('gating_cases',len(result['gating_cases'])==9 and all('decode' not in c and 'ttft' not in c for c in result['gating_cases']))
        if name=='decode_regression_not_gating':
            check('decode_reported',any(r['metric']=='decode_client_tps' and not r['gating'] and abs(r['geomean']-.8)<1e-12 for r in result['rows']))
    # main(): exit codes, usage/internal errors, O_EXCL, GPU lock
    def fresh(name):
        target=pathlib.Path(tempfile.mkdtemp(dir=work));shutil.copytree(roots[name],target/'r');return target/'r'
    try:
        for name,want in (('pass',0),('tool_regression',1),('noninferiority_borderline',2),('vram_over_budget',3)):
            root=fresh(name);code=run_main(mod,root);path=root/'reports/fixture-main-retain-vs-bf128-comparison.json'
            check(f'main_exit_{name}',code==want and path.exists() and json.loads(path.read_text())['verdict']==SCENARIOS[name][1])
        root=fresh('pass')
        check('main_usage_missing_name',run_main(mod,root,['--candidate','main-retain'])==4)
        check('main_usage_bad_lengths',run_main(mod,root,['--name','fixture','--candidate','main-retain','--lengths','64,64'])==4)
        with patch.object(mod,'compare',side_effect=KeyError('boom')):code=run_main(mod,root)
        check('main_internal_error',code==4 and not (root/'reports/fixture-main-retain-vs-bf128-comparison.json').exists())
        check('main_first_write',run_main(mod,root)==0)
        path=root/'reports/fixture-main-retain-vs-bf128-comparison.json';before=path.read_bytes()
        check('main_existing_refused',run_main(mod,root)==4 and path.read_bytes()==before)
        root=fresh('pass');path=root/'reports/fixture-main-retain-vs-bf128-comparison.json';real=mod.compare
        def racing(*args,**kw):
            report=real(*args,**kw);path.write_text('racer');return report
        with patch.object(mod,'compare',side_effect=racing):code=run_main(mod,root)
        check('main_o_excl_race',code==4 and path.read_text()=='racer')
        root=fresh('pass');lock=root/'held.lock';held=bench.acquire_gpu_lock(lock)
        try:code=run_main(mod,root,lock=lock)
        finally:os.close(held)
        check('main_lock_busy',code==75 and not (root/'reports/fixture-main-retain-vs-bf128-comparison.json').exists())
    except Exception as error:bad.append('main_error_'+type(error).__name__)
    return bad


MUTANTS=(
    ('C01 bench gain bound without Bonferroni','out.append(stats_row(key[0],key[1],metric,ratios,gating,family))',
     'out.append(stats_row(key[0],key[1],metric,ratios,gating,1))'),
    ('C02 tool gain bound without Bonferroni',"task=stats_row('tool_cycle',None,'task_elapsed_s',task_ratios,True,family)",
     "task=stats_row('tool_cycle',None,'task_elapsed_s',task_ratios,True,1)"),
    ('C03 family size 1','family=gating_cases*candidates','family=1'),
    ('C04 one-sided alpha .05 instead of .025','ALPHA_ONE_SIDED=.025','ALPHA_ONE_SIDED=.05'),
    ('C05 population sd (ddof 0)','se=statistics.stdev(logs)/math.sqrt(PAIRS)','se=statistics.pstdev(logs)/math.sqrt(PAIRS)'),
    ('C06 arithmetic mean of ratios','mean=statistics.fmean(logs)','mean=math.log(statistics.fmean(ratios))'),
    ('C07 NONINFERIORITY=1/1.10','NONINFERIORITY=1/1.03','NONINFERIORITY=1/1.10'),
    ('C08 GAIN=1.0','GAIN=1.05','GAIN=1.0'),
    ('C09 GAIN=1.03','GAIN=1.05','GAIN=1.03'),
    ('C10 noninferiority ignores tool cycle','verdict_rows=out+[task]','verdict_rows=out'),
    ('C11 gain over any metric',"gain=any(r['gain_lower_bonferroni']>=GAIN for r in gating)",
     "gain=any((r.get('gain_lower_bonferroni') or r['lower_975'])>=GAIN for r in rows)"),
    ('C12 noninferiority any instead of all',"noninferior=all(r['lower_975']>=NONINFERIORITY for r in gating)",
     "noninferior=any(r['lower_975']>=NONINFERIORITY for r in gating)"),
    ('C13 regression uses lower bound',"regressions=[r for r in gating if r['upper_975']<NONINFERIORITY]",
     "regressions=[r for r in gating if r['lower_975']<NONINFERIORITY]"),
    ('C14 gain exclusion uses lower bound',"gain_excluded=all(r['upper_975']<GAIN for r in gating)",
     "gain_excluded=all(r['lower_975']<GAIN for r in gating)"),
    ('V6-01 decode gates','gating=metric==GATING_METRIC',"gating=metric in (GATING_METRIC,'decode_client_tps')"),
    ('V6-02 gain label always (compare-ab.py:129 bug)',"*(['gain_not_established'] if not gain else [])]","'gain_not_established']"),
    ('V6-03 t with df=5','T_DF=PAIRS-1','T_DF=PAIRS'),
    ('V6-04 report without O_EXCL','os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600','os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600'),
    ('V6-05 internal error exits 1 (FAIL)','EXIT_INTERNAL=4','EXIT_INTERNAL=1'),
    ('V6-06 argparse usage error exits 2 (INCONCLUSIVE)','    def error(self,message):raise UsageError(message)','    pass'),
    ('V6-07 VRAM budget ignored',"require(budget['gate']!='INVALID'","require(True or budget['gate']!='INVALID'"),
    ('V6-08 launcher env equality ignored','require(len(launcher_envs)<=1','require(True'),
    ('V6-09 launcher problems ignored',"\n               and launcher.get('problems')==[]",''),
    ('V6-10 matrix arm verdict ignored',"require(entry.get('arm_verdict')=='VALID'","require(True or entry.get('arm_verdict')=='VALID'"),
    ('V6-11 compare without GPU lock','try:lock=acquire_gpu_lock(lock_path)','try:lock=os.open(os.devnull,os.O_RDONLY)'),
)


def load_variant(source,label):
    module=types.ModuleType('compare_ab_'+label);module.__file__=str(HERE/'compare-ab.py')
    exec(compile(source,f'compare-ab[{label}]','exec'),module.__dict__)
    return module


class V6Statistics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();base=pathlib.Path(cls.tmp.name)
        (base/'roots').mkdir();(base/'work').mkdir()
        cls.roots=build_scenarios(base/'roots');cls.work=base/'work'

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_t_quantiles_known_vectors_and_closed_form(self):
        self.assertAlmostEqual(comparison.t_quantile(.975),T975,places=12)
        self.assertAlmostEqual(comparison.t_quantile(1-.025/18),6.569700477595786,places=8)  # review: t4 Bonferroni 6.57
        for p in (.5,.75,.95,.99,.999,1-.025/27):self.assertAlmostEqual(comparison.t_quantile(p),0. if p==.5 else t_closed(p),places=8)
        for p in (.4,1.,1.5,'x'):
            with self.assertRaises(ValueError):comparison.t_quantile(p)
        with self.assertRaises(ValueError):comparison.t_quantile(.975,df=5)
        with self.assertRaises(ValueError):comparison.summarize([1.1]*5,0)

    def test_battery_passes_on_v6_module(self):
        self.assertEqual(battery(comparison,self.roots,self.work),[])

    def test_noninferiority_and_gain_use_only_elapsed(self):
        result=compare(self.roots['decode_regression_not_gating'])
        gating={(r['case'],r['metric']) for r in result['rows'] if r['gating']}
        self.assertEqual({metric for _,metric in gating},{'elapsed_s'})
        self.assertTrue(all(r.get('gain_lower_bonferroni') is None for r in result['rows'] if not r['gating']))
        self.assertTrue(result['tool_task']['gating']);self.assertEqual(result['verdict'],'PASS')

    def test_missing_decode_or_ttft_is_reported_not_gating(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);fixture(root)
            edit(root,'fixture-0-bf128',lambda d:d['rows'][1].update(decode_client_tps=None,ttft_content_s=None))
            result=compare(root)
        self.assertEqual(result['verdict'],'PASS',result['failures'])
        self.assertEqual(sorted(result['report_metric_gaps']),['decode_client_tps_0_fresh_64','ttft_content_s_0_fresh_64'])

    def test_gain_label_only_when_gain_not_established(self):
        result=compare(self.roots['noninferiority_borderline'])
        self.assertTrue(result['whole_request_gain_at_least_5pct_bonferroni'])
        self.assertNotIn('gain_not_established',result['verdict_reasons'])

    def test_mutants_c01_c14_and_v6_are_killed(self):
        for label,old,new in MUTANTS:
            with self.subTest(mutant=label):
                self.assertEqual(COMPARE_SOURCE.count(old),1,'anchor must be unique: '+label)
                try:
                    module=load_variant(COMPARE_SOURCE.replace(old,new),label.split()[0])
                    failures=battery(module,self.roots,self.work)
                except Exception as error:failures=['load_error_'+type(error).__name__]
                MUTATION_RESULTS[label]={'killed':bool(failures),'by':failures[:4]}
                self.assertTrue(failures,'mutant survived: '+label)

    def test_verify_current_drift_names_launcher_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);manifest=fixture(root)
            current={key:manifest[key] for key in ('schema_version','profiles','lengths','output','pairs','arms','harness_source_files')}
            current['launcher_files']=dict(FAKE_LAUNCHER,**{'run-experiment.sh':'7'*64})
            with patch.object(comparison,'current_run_spec',return_value=current):
                result=comparison.compare(root,'fixture','main-retain','bf128')
            self.assertEqual(result['verdict'],'INVALID');self.assertEqual(result['drift_keys'],['launcher_files'])
            current['launcher_files']=dict(FAKE_LAUNCHER)
            with patch.object(comparison,'current_run_spec',return_value=current):
                self.assertEqual(comparison.compare(root,'fixture','main-retain','bf128')['verdict'],'PASS')

    def test_manifest_without_launcher_or_windows_reserve_is_invalid(self):
        for key in ('launcher_files','windows_reserve_bytes'):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as folder:
                root=pathlib.Path(folder);fixture(root)
                edit(root,'fixture-manifest',lambda d:d.pop(key))
                result=compare(root);self.assertEqual(result['verdict'],'INVALID')


# =============================================================================================
class V6Launcher(unittest.TestCase):
    def test_environment_filter_redacts_secrets(self):
        raw=b'\0'.join([b'PYTHONPATH=/p',b'QWEN36_API_KEY=supersecretvalue',b'HF_TOKEN=tok',b'INVOCATION_ID=abc',
                        b'LD_LIBRARY_PATH=/l',b'RANDOM_VAR=1',b'OMP_NUM_THREADS=16',b'PATH=/bin',b'noequals'])
        env=bench.launcher_environment(raw)
        self.assertEqual(env,{'LD_LIBRARY_PATH':'/l','OMP_NUM_THREADS':'16','PATH':'/bin','PYTHONPATH':'/p','QWEN36_API_KEY':'<redacted>'})
        self.assertNotIn('supersecret',json.dumps(env))

    def test_real_run_experiment_sh_satisfies_static_contract(self):
        lines=[line for line in (HERE/'run-experiment.sh').read_text().splitlines()
               if line.startswith(('root=','export ','unset '))]
        self.assertTrue(any(line.startswith('export PYTHONPATH=') for line in lines))
        out=subprocess.run(['bash','-c','\n'.join(lines)+'\nexec env -0'],capture_output=True,timeout=30,
                           env={'PATH':'/usr/bin:/bin','HOME':'/tmp','FREETOKEN_API_LOG_DIR':'/x'}).stdout
        env=bench.launcher_environment(out)
        root=bench.ROOT;cmd=[str(root.parent/'.venv/bin/python'),str(root/'experiment.py'),'bf128']
        self.assertEqual(bench.launcher_problems(env,cmd,'bf128'),[])
        for key,value,code in (('PYTHONPATH',None,'pythonpath'),('LD_PRELOAD','/x.so','ld_preload'),('CUDA_HOME','/other','cuda_home'),
                               ('QWEN36_FUSED_ATTN_GATE','1','feature_env_in_launcher'),('FREETOKEN_API_LOG_DIR','/x','api_log_dir')):
            broken=dict(env);broken.pop(key,None)
            if value is not None:broken[key]=value
            self.assertIn(code,bench.launcher_problems(broken,cmd,'bf128'))
        self.assertIn('cmdline',bench.launcher_problems(env,cmd[:2]+['other'],'bf128'))

    def test_launcher_receipt_from_proc_and_unit(self):
        with tempfile.TemporaryDirectory() as folder:
            base=pathlib.Path(folder);root=base/'opt';root.mkdir();proc=base/'proc';(proc/'77').mkdir(parents=True)
            for name in bench.LAUNCHER_FILES:(root/name).write_text(name)
            cuda=str(base/'cuda-13.4')
            env={'PYTHONPATH':str(root/'python'),'QWEN36_DATA_ROOT':str(root),'CUDA_HOME':cuda,
                 'LD_LIBRARY_PATH':f'{cuda}/lib:{cuda}/lib64:/usr/lib/wsl/lib:','QWEN36_API_KEY':'s3cret'}
            (proc/'77/environ').write_bytes(b'\0'.join(f'{k}={v}'.encode() for k,v in env.items())+b'\0')
            (proc/'77/cmdline').write_bytes(b'\0'.join([b'/x/.venv/bin/python',str(root/'experiment.py').encode(),b'bf128'])+b'\0')
            values={'MainPID':'77','InvocationID':'inv'}
            receipt=bench.launcher_receipt('bf128',root,show=values.get,proc=proc)
            self.assertEqual(receipt['files'],{n:hashlib.sha256(n.encode()).hexdigest() for n in bench.LAUNCHER_FILES})
            self.assertEqual(receipt['problems'],['feature_env_in_launcher'])  # QWEN36_API_KEY present at exec
            self.assertEqual(receipt['environment']['QWEN36_API_KEY'],'<redacted>');self.assertNotIn('s3cret',json.dumps(receipt))
            self.assertEqual(receipt['environment_sha256'],bench.canonical_hash(receipt['environment']))
            answers=iter(['77','inv','78','inv'])
            with self.assertRaisesRegex(RuntimeError,'changed'):bench.launcher_receipt('bf128',root,show=lambda p:next(answers),proc=proc)
            with self.assertRaisesRegex(RuntimeError,'no main process'):
                bench.launcher_receipt('bf128',root,show={'MainPID':'0','InvocationID':'inv'}.get,proc=proc)

    def test_matrix_check_launcher_first_arm_file_and_mismatches(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);(root/'reports').mkdir();spec={'launcher_files':dict(FAKE_LAUNCHER)}
            matrix.check_launcher(root,'m',PROFILES[0],spec,launcher_doc())
            path=root/'reports/m-launcher.json';self.assertEqual(path.stat().st_mode&0o777,0o600);before=path.read_bytes()
            matrix.check_launcher(root,'m',PROFILES[1],spec,launcher_doc())
            for label,doc,expected in (('env',launcher_doc(dict(LAUNCHER_ENV,OMP_NUM_THREADS='8')),'changed_between_arms'),
                                        ('files',dict(launcher_doc(),files=dict(FAKE_LAUNCHER,**{'start-experiment.sh':'0'*64})),'files_differ'),
                                        ('problems',launcher_doc(problems=['pythonpath']),'unexpected:pythonpath')):
                with self.subTest(label=label):
                    with self.assertRaisesRegex(matrix.LauncherIdentityError,expected):matrix.check_launcher(root,'m','x',spec,doc)
            self.assertEqual(path.read_bytes(),before)

    def test_current_run_spec_hashes_launchers_and_keeps_soak_identity_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);(root/'python/freetoken').mkdir(parents=True);(root/'config').mkdir()
            (root/'python/freetoken/a.py').write_text('x=1\n')
            for name in bench.SERVICE_FILES+bench.LAUNCHER_FILES:(root/name).write_text(name)
            (root/'config/bf128.json').write_text(json.dumps({'context':131072,'chunk':2048,'flags':[],'env':{}}))
            with patch('importlib.metadata.version', return_value='synthetic-cpu-test'):
                spec=bench.current_run_spec(root,['bf128'],bench.DEFAULT_LENGTHS,128,5)
        self.assertEqual(spec['launcher_files'],{n:hashlib.sha256(n.encode()).hexdigest() for n in bench.LAUNCHER_FILES})
        # soak.py IDENTITY_KEYS == validate_receipt identity == manifest arms[profile]
        self.assertEqual(set(spec['arms']['bf128']),{'source_sha256','service_source_files','checkpoint','checkpoint_revision',
                                                     'packages','configuration','native_libraries'})


# =============================================================================================
class V6MatrixArms(unittest.TestCase):
    def harness(self,root,spec,over_budget_at=None,launcher_bad_at=None,start_fail_at=None,drift_at=None):
        state={'inv':'','count':0};stops=[];calls=[];specs={'n':0}
        def fake_run(args,timeout=1200):
            calls.append(args)
            if args[0]=='bash':
                state['count']+=1;state['inv']=f'inv-{state["count"]}'
                (root/'active-run.json').write_text(json.dumps(receipt_for(spec,args[2],f'{state["count"]:032x}')))
                if state['count']==start_fail_at:raise RuntimeError('bash: exit 1')
            elif args[:3]==['systemctl','--user','stop']:stops.append(state['inv']);state['inv']=''
            elif str(args[1]).endswith('bench.py'):
                used=18*GIB+1 if state['count']==over_budget_at else 12*GIB  # synthetic: 3 GiB foreign before start
                (root/'reports'/f'{args[3]}.json').write_text(json.dumps({'rows':[{'case':'warmup','peaks':{'min_gpu_free_bytes':GIB}},
                    {'case':'fresh','peaks':{'min_gpu_free_bytes':GIB//4,'gpu_total_used_bytes':used},
                     'primer':{'peaks':{'min_gpu_free_bytes':GIB,'gpu_total_used_bytes':11*GIB}}}]}))
            elif str(args[1]).endswith('tool-task-bench.py'):
                (root/'reports'/f'{args[3]}-tool-task.json').write_text(json.dumps({'row':{'steps':[{'peaks':{'min_gpu_free_bytes':GIB}}]}}))
            return ''
        def fake_state(prop,unit='qwen36-experiment'):
            if prop=='InvocationID':return state['inv'] if unit=='qwen36-experiment' else ''
            if prop=='ActiveState':return 'active' if state['inv'] and unit=='qwen36-experiment' else 'inactive'
            return ''
        def fake_launcher(profile,root_):
            env=dict(LAUNCHER_ENV,OMP_NUM_THREADS='4') if state['count']==launcher_bad_at else LAUNCHER_ENV
            return dict(launcher_doc(env),invocation_id=state['inv'])
        def fake_spec(*args):
            specs['n']+=1
            return dict(spec,harness_source_files={'bench.py':'0'*64}) if drift_at and specs['n']==drift_at+1 else spec
        patches=[patch.object(matrix,'run',side_effect=fake_run),patch.object(matrix,'state',side_effect=fake_state),
                 patch.object(matrix,'build_spec',side_effect=fake_spec),patch.object(matrix,'cgroup_peak',return_value=None),
                 patch.object(matrix,'workspace_config_drift',return_value=False),
                 patch.object(bench,'gpu_memory_snapshot',return_value={'used_bytes':3*GIB,'free_bytes':13*GIB,'total_bytes':16*GIB}),
                 patch.object(bench,'launcher_receipt',side_effect=fake_launcher)]
        return patches,stops,calls

    def run_case(self,name,**kw):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        root=pathlib.Path(folder.name);(root/'reports').mkdir();spec=make_manifest()
        patches,stops,calls=self.harness(root,spec,**kw)
        for item in patches:item.start()
        try:rc=quiet(matrix.run_matrix,types.SimpleNamespace(name=name,output=128,pairs=5,tool_pilot='p'),PROFILES,bench.DEFAULT_LENGTHS,root)
        finally:
            for item in reversed(patches):item.stop()
        return rc,json.loads((root/'reports'/f'{name}-matrix.json').read_text()),stops,root,spec

    def test_vram_budget_marks_arm_invalid_and_matrix_exits_3(self):
        rc,rows,stops,root,spec=self.run_case('vram',over_budget_at=2)
        self.assertEqual((rc,len(rows)),(3,15));self.assertEqual(len(stops),15)
        bad=[r for r in rows if r['arm_verdict']=='INVALID']
        self.assertEqual(len(bad),1);self.assertEqual(bad[0]['schedule_position'],1)
        self.assertEqual((bad[0]['vram']['gate'],bad[0]['vram']['code'],bad[0]['invalid_codes']),
                         ('INVALID','process_vram_over_budget',['process_vram_over_budget']))
        self.assertEqual((bad[0]['vram']['process_peak_bytes'],bad[0]['vram']['budget_bytes']),(15*GIB+1,15*GIB))
        # A quarter GiB free on every arm is not a failure: only the footprint is budgeted.
        self.assertTrue(all(r['vram']['min_gpu_free_bytes']==GIB//4 for r in rows))
        self.assertEqual(rows[0]['vram']['foreign_gpu_used_bytes_pre_start'],3*GIB)
        self.assertTrue(all(r['vram']['gate']=='PASS' and r['status']=='complete' for r in rows if r is not bad[0]))
        self.assertEqual(json.loads((root/'reports/vram-manifest.json').read_text())['windows_reserve_bytes'],bench.WINDOWS_RESERVE_BYTES)
        rc,rows,_,_,_=self.run_case('allgood')
        self.assertEqual(rc,0);self.assertTrue(all(r['arm_verdict']=='VALID' for r in rows))

    def test_launcher_change_between_arms_fails_arm_not_resumable(self):
        rc,rows,stops,root,spec=self.run_case('launch',launcher_bad_at=3)
        self.assertEqual((rc,len(rows)),(1,3));self.assertEqual(stops,['inv-1','inv-2','inv-3'])
        self.assertEqual((rows[-1]['failed_step'],rows[-1]['launcher_error']),('launcher','launcher_environment_changed_between_arms'))
        self.assertTrue((root/'reports/launch-launcher.json').exists())
        with self.assertRaisesRegex(RuntimeError,'cannot be resumed'):
            matrix.check_start_or_resume(root,'launch',spec,rows,bench.DEFAULT_LENGTHS,128)

    def test_start_failure_stops_orphan_unit_and_drift_is_recorded(self):
        rc,rows,stops,_,_=self.run_case('orphan',start_fail_at=2)
        self.assertEqual((rc,len(rows),stops),(1,2,['inv-1','inv-2']))
        self.assertEqual((rows[-1]['failed_step'],rows[-1]['stopped_after_start_failure']),('start',True))
        rc,rows,stops,_,_=self.run_case('drift',drift_at=2)
        self.assertEqual((rc,len(rows)),(1,2));self.assertEqual(rows[-1]['failed_step'],'preflight')
        self.assertEqual(rows[-1]['drift_keys'],['harness_source_files']);self.assertEqual(stops,['inv-1'])

    def test_state_is_fail_closed_and_snapshot_parsing(self):
        with patch.object(matrix.subprocess,'run',return_value=types.SimpleNamespace(returncode=1,stdout='')):
            with self.assertRaises(RuntimeError):matrix.state('ActiveState')
            with self.assertRaises(RuntimeError):matrix.units_inactive()
        ok=lambda *a,**k:types.SimpleNamespace(returncode=0,stdout='3072, 13238, 16303\n')
        self.assertEqual(bench.gpu_memory_snapshot(ok),{'used_bytes':3072<<20,'free_bytes':13238<<20,'total_bytes':16303<<20})
        for fake in (lambda *a,**k:types.SimpleNamespace(returncode=9,stdout=''),lambda *a,**k:types.SimpleNamespace(returncode=0,stdout='N/A'),
                     MagicMock(side_effect=OSError('missing'))):
            self.assertIsNone(bench.gpu_memory_snapshot(fake))
        rows=[{'peaks':{'min_gpu_free_bytes':5}},{'primer':{'peaks':{'min_gpu_free_bytes':3}}},{'steps':[{'peaks':{'min_gpu_free_bytes':4}}]},
              {'peaks':{'min_gpu_free_bytes':float('nan')}},{'peaks':'x'}]
        self.assertEqual(bench.min_gpu_free(rows),3);self.assertIsNone(bench.min_gpu_free([{'peaks':{}}]))

    def test_workspace_config_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);(root/'config').mkdir();ws=root/'ws';ws.mkdir()
            (root/'config/a.json').write_text('{"x":1}')
            self.assertFalse(matrix.workspace_config_drift(root,'a',ws))
            (ws/'a.json').write_text('{"x":1}');self.assertFalse(matrix.workspace_config_drift(root,'a',ws))
            (ws/'a.json').write_text('{"x":2}');self.assertTrue(matrix.workspace_config_drift(root,'a',ws))
            (ws/'b.json').write_text('{}');self.assertTrue(matrix.workspace_config_drift(root,'b',ws))


# =============================================================================================
class V6StartExperiment(unittest.TestCase):
    """start-experiment.sh in a sandbox: stubs for systemctl/systemd-run; root/workspace rewritten."""
    def sandbox(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup);base=pathlib.Path(folder.name)
        native=base/'native';ws=base/'ws';stubs=base/'stubs'
        for d in (native/'config',ws,stubs):d.mkdir(parents=True)
        text=(HERE/'start-experiment.sh').read_text()
        self.assertEqual(text.count('root=/home/USER/qwen36-api/optimization\n'),1)
        self.assertEqual(text.count("workspace='/path/to/qwen36-workspace/outputs/qwen36-api/optimization'\n"),1)
        text=text.replace('root=/home/USER/qwen36-api/optimization\n',f'root={native}\n').replace(
            "workspace='/path/to/qwen36-workspace/outputs/qwen36-api/optimization'\n",f"workspace='{ws}'\n")
        (base/'start.sh').write_text(text)
        (stubs/'systemctl').write_text('#!/bin/bash\n[ "${STUB_ACTIVE:-}" = "$4" ] && exit 0\nexit 3\n')
        (stubs/'systemd-run').write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > {base}/systemd-run.args\n')
        for f in stubs.iterdir():f.chmod(0o700)
        def start(profile,**env):
            return subprocess.run(['bash',str(base/'start.sh'),profile],capture_output=True,text=True,timeout=30,
                                  env={'PATH':f'{stubs}:/usr/bin:/bin',**env})
        def args():
            path=base/'systemd-run.args'
            if not path.exists():return None
            out=path.read_text().splitlines();path.unlink();return out
        return native,ws,start,args

    def test_copies_only_the_started_profile_config(self):
        native,ws,start,args=self.sandbox()
        (ws/'bf128.json').write_text('{"chunk":2048}');(ws/'other.json').write_text('{}');(ws/'evil.sh').write_text('rm -rf /\n')
        result=start('bf128');self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(sorted(p.name for p in native.iterdir()),['config'])
        self.assertEqual(sorted(p.name for p in (native/'config').iterdir()),['bf128.json'])
        self.assertEqual((native/'config/bf128.json').read_text(),'{"chunk":2048}')
        self.assertEqual((native/'config/bf128.json').stat().st_mode&0o777,0o600)
        self.assertEqual(args()[-3:],['/bin/bash',f'{native}/run-experiment.sh','bf128'])
        (ws/'teacher-cuda-bitexact128-2048.json').write_text('{"teacher":1}')
        self.assertEqual(start('teacher-cuda-bitexact128-2048').returncode,0)
        self.assertEqual((native/'config/teacher-cuda-bitexact128-2048.json').read_text(),'{"teacher":1}')
        self.assertEqual(args()[-1],'teacher-cuda-bitexact128-2048')

    def test_names_native_only_profiles_and_refusals(self):
        native,ws,start,args=self.sandbox()
        for bad in ('../evil','Bad','a/b','-x','x y'):
            with self.subTest(name=bad):
                result=start(bad);self.assertEqual(result.returncode,64);self.assertIsNone(args())
        (native/'config/profile-final-main-retain2048.json').write_text('{}')
        self.assertEqual(start('profile-final-main-retain2048').returncode,0)
        self.assertEqual(args()[-2:],[f'{native}/profile.sh','profile-final-main-retain2048'])
        self.assertEqual(start('missing').returncode,66);self.assertIsNone(args())
        (ws/'real.json').write_text('{}');(ws/'link.json').symlink_to(ws/'real.json')
        self.assertEqual(start('link').returncode,66);self.assertFalse((native/'config/link.json').exists())
        (ws/'busy.json').write_text('{}')
        self.assertEqual(start('busy',STUB_ACTIVE='qwen36-experiment.service').returncode,73)
        self.assertEqual(start('busy',STUB_ACTIVE='qwen36-inference.service').returncode,73)
        self.assertFalse((native/'config/busy.json').exists());self.assertIsNone(args())

    def test_no_blanket_copy_and_qualify_flow_compatible(self):
        text=(HERE/'start-experiment.sh').read_text()
        self.assertNotIn('"$workspace/"*',text);self.assertNotIn('*.sh',text.replace('# ',''))
        qualify=(HERE/'qualify.py').read_text()
        ws_start=text.split("workspace='",1)[1].split("'",1)[0]
        ws_qualify=qualify.split("workspace=pathlib.Path('",1)[1].split("'",1)[0]
        self.assertEqual(ws_start,ws_qualify)
        self.assertIn("teacher='teacher-'+a.candidate",qualify);self.assertIn("start(teacher)",qualify)
        import re
        for candidate in ('cuda-bitexact128-2048','cuda-bitexact128','main-retain2048','bf128'):
            for name in (candidate,'teacher-'+candidate):self.assertTrue(re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,120}',name))
            self.assertTrue((HERE/f'{candidate}.json').exists())


# =============================================================================================
screen=load('screen-chunk')


class V6ChunkScreen(unittest.TestCase):
    def test_screen_receipts_scoped_stop_failure_and_refusals(self):
        manifest=make_manifest();state={'inv':'','count':0};stops=[]
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);(root/'reports').mkdir()
            def fake_run(args,timeout=1200):
                if args[0]=='bash':
                    state['count']+=1;state['inv']=f'inv-{state["count"]}'
                    (root/'active-run.json').write_text(json.dumps(receipt_for(manifest,args[2],f'{state["count"]:032x}')))
                elif args[:3]==['systemctl','--user','stop']:stops.append(state['inv']);state['inv']=''
                elif str(args[1]).endswith('check-experiment.py') and state['count']==2:raise RuntimeError('python: exit 1')
                return ''
            def fake_state(prop,unit='qwen36-experiment'):
                return state['inv'] if prop=='InvocationID' and unit=='qwen36-experiment' else ''
            def fake_generate(messages,maximum=128,seed=17,temperature=.6,ignore_eos=False,**kw):
                return {'status':200,'usage':{'prompt_tokens':3000,'completion_tokens':maximum,'total_tokens':3000+maximum},
                        'ttft_content_s':2.0,'elapsed_s':3.0,'decode_client_tps':50.,'finish_reason':'length',
                        'peaks':{'min_gpu_free_bytes':GIB,'gpu_total_used_bytes':12*GIB},'engine_metrics_status':'joined',
                        'engine':{'prefill_gpu_ms':1500.,'expert_slots':3175,'expert_counter_deltas':{'prefill_h2d_bytes':123,'decode_missing':7},
                                  'uid':1}},''
            spec=lambda root_,profiles,*a:{'arms':{p:manifest['arms'][p] for p in profiles},'launcher_files':dict(FAKE_LAUNCHER),
                                           'harness_source_files':dict(FAKE_HARNESS)}
            # screen-chunk imports the sys.modules `matrix`, not this file's load('matrix') copy.
            patches=[patch.object(screen.matrix,'run',side_effect=fake_run),patch.object(screen.matrix,'state',side_effect=fake_state),
                     patch.object(screen.matrix,'units_inactive',side_effect=lambda:not state['inv']),
                     patch.object(screen.matrix,'workspace_config_drift',return_value=False),
                     patch.object(screen.matrix,'cgroup_peak',return_value=None),
                     patch.object(screen,'current_run_spec',side_effect=spec),
                     patch.object(bench,'gpu_memory_snapshot',return_value={'used_bytes':3*GIB,'free_bytes':13*GIB,'total_bytes':16*GIB}),
                     patch.object(bench,'launcher_receipt',side_effect=lambda profile,root_:launcher_doc()),
                     patch.object(bench,'generate',side_effect=fake_generate),patch.object(bench,'document',return_value='x')]
            for item in patches:item.start()
            try:
                argv=['--name','scr','--profiles','bf128,main-retain,main-retain2048']
                rc=quiet(screen.main,argv,root=root,lock_path=root/'l',tokenizer=object())
                self.assertEqual(rc,1);self.assertEqual(stops,['inv-1','inv-2','inv-3'])
                docs={p:json.loads((root/'reports'/f'scr-{p}-chunk-screen.json').read_text()) for p in PROFILES}
                self.assertEqual([docs[p]['status'] for p in PROFILES],['measured','failed','measured'])
                self.assertEqual(docs['main-retain']['failed_step'],'readiness')
                first=docs['bf128'];fresh=[r for r in first['rows'] if r['case']=='fresh']
                self.assertEqual([r['length'] for r in fresh],[16384,65536,120000]);self.assertFalse(first['acceptance'])
                self.assertAlmostEqual(fresh[0]['prefill_engine_tps'],2000.);self.assertAlmostEqual(fresh[0]['prefill_client_tps'],1500.)
                self.assertEqual(fresh[0]['engine']['expert_counter_deltas']['prefill_h2d_bytes'],123);self.assertNotIn('uid',fresh[0]['engine'])
                self.assertEqual((first['vram_gate'],first['min_gpu_free_bytes']),('PASS',GIB))
                summary=json.loads((root/'reports/scr-chunk-screen.json').read_text())
                self.assertEqual(summary['results'][0]['fresh'][2]['prefill_h2d_bytes'],123)
                self.assertEqual((root/'reports/scr-chunk-screen.json').stat().st_mode&0o777,0o600)
                self.assertEqual(quiet(screen.main,argv,root=root,lock_path=root/'l',tokenizer=object()),2)
                held=bench.acquire_gpu_lock(root/'l')
                try:self.assertEqual(quiet(screen.main,['--name','scr2'],root=root,lock_path=root/'l',tokenizer=object()),75)
                finally:os.close(held)
                self.assertFalse(any(p.name.startswith('scr2') for p in (root/'reports').iterdir()))
            finally:
                for item in reversed(patches):item.stop()

    def test_screen_configs_differ_only_in_chunk_and_ratio(self):
        base=json.loads((HERE/'cuda-bitexact128-2048.json').read_text())
        retain=json.loads((HERE/'main-retain2048.json').read_text())
        self.assertEqual({k:v for k,v in base['env'].items() if k not in ('QWEN36_NVFP4_PREFILL','QWEN36_CUDA_NVFP4_LIBRARY')},retain['env'])
        self.assertEqual(base['env']['QWEN36_CUDA_NVFP4_LIBRARY'],'/home/USER/qwen36-api/optimization/private-libraries/libqwen_nvfp4_bitexact.so')
        for name,chunk,ratio in (('cuda-bitexact128-c2048',2048,.75),('cuda-bitexact128',4096,.73),('cuda-bitexact128-c6144',6144,.71),
                                 ('cuda-bitexact128-c8192',8192,.69),('cuda-bitexact128-c16384',16384,.63)):
            config=json.loads((HERE/f'{name}.json').read_text());bench.validate_speed_config(config)
            self.assertEqual((config['chunk'],config['memory_ratio']),(chunk,ratio))
            self.assertEqual({k:v for k,v in config.items() if k not in ('chunk','memory_ratio')},
                             {k:v for k,v in base.items() if k not in ('chunk','memory_ratio')})


# =============================================================================================
logit_ref=load('compare-logit-reference')


class V6LogitReference(unittest.TestCase):
    def test_old_vs_new_reference_report(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);(root/'reports').mkdir();old=root/'reports/logit-reference-pre-r1';new=root/'reports/logit-reference'
            old.mkdir();new.mkdir()
            np.save(old/'11-0.npy',np.array([1.,2.,3.],dtype=np.float32));np.save(new/'11-0.npy',np.array([1.,2.,3.],dtype=np.float32))
            np.save(old/'11-8.npy',np.array([3.,1.,0.],dtype=np.float32));np.save(new/'11-8.npy',np.array([1.,3.,0.],dtype=np.float32))
            np.save(old/'12-0.npy',np.zeros(3,dtype=np.float32))
            rows=[{'profile':'teacher-bf128','run_id':'a','case':'c1','nll':[1.,2.]},{'profile':'teacher-main','run_id':'z','case':'c1','nll':[9.]},
                  {'profile':'teacher-bf128','run_id':'b','case':'c1','nll':[1.,2.5]}]
            (root/'reports/teacher.jsonl').write_text('\n'.join(map(json.dumps,rows))+'\n')
            self.assertEqual(quiet(logit_ref.main,['--name','ref-r1'],root=root),0)
            doc=json.loads((root/'reports/ref-r1.json').read_text())
            self.assertEqual((doc['logits']['compared'],doc['logits']['only_old'],doc['logits']['bit_identical']),(2,1,1))
            self.assertEqual((doc['logits']['max_abs'],doc['logits']['top1_agreement']),(2.,.5))
            self.assertEqual((doc['nll']['old_run_id'],doc['nll']['new_run_id'],doc['nll']['tokens'],doc['nll']['max_token_abs_delta']),('a','b',2,.5))
            self.assertAlmostEqual(doc['nll']['relative_perplexity_change'],math.exp(.25)-1);self.assertFalse(doc['gate'])
            self.assertEqual(quiet(logit_ref.main,['--name','ref-r1'],root=root),2)


def cuda_state():
    torch=sys.modules.get('torch')
    return {'torch_imported':torch is not None,'cuda_initialized':bool(torch is not None and torch.cuda.is_initialized()),
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}


def main():
    receipt=None
    if '--receipt' in sys.argv:
        index=sys.argv.index('--receipt');receipt=pathlib.Path(sys.argv[index+1]);del sys.argv[index:index+2]
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    ids=[test.id() for group in suite for test in group]  # before run(): the suite releases tests
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    sys.stdout.write(stream.getvalue())
    cuda=cuda_state()
    doc={'status':'PASS' if result.wasSuccessful() and not cuda['cuda_initialized'] else 'FAIL',
         'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
         'v4_adapted_tests':sum('.V4Adapted.' in i for i in ids),'v5_tests':sum('.V5' in i for i in ids),
         'v6_new_tests':sum('.V6' in i for i in ids),
         'mutants':len(MUTANTS),'mutants_killed':sum(v['killed'] for v in MUTATION_RESULTS.values()),'mutation_results':MUTATION_RESULTS,
         'test_ids':[i.split('.',1)[1] for i in ids],'model_started':False,'http_contacted':False,**cuda,
         'python':sys.version.split()[0],
         'sources_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                           for name in (*bench.HARNESS_FILES,*bench.LAUNCHER_FILES,'test-paired-contract-cpu.py','qualify.py')}}
    print(json.dumps({k:v for k,v in doc.items() if k not in ('test_ids','mutation_results')}))
    if receipt:receipt.write_text(json.dumps(doc,indent=2)+'\n')
    raise SystemExit(0 if doc['status']=='PASS' else 1)


if __name__=='__main__':main()
