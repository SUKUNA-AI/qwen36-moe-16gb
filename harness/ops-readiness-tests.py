"""CPU-only readiness checks. No secret file, sockets, SSH or GPU is accessed."""
import importlib.util,json,pathlib,tempfile,types
from unittest.mock import patch

BASE=pathlib.Path(__file__).resolve().parents[3]
path=BASE/'outputs/qwen36-api/readiness.py'
spec=importlib.util.spec_from_file_location('readiness',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
results=[]
def check(name,condition):
    assert condition,name
    results.append({'name':name,'pass':True})
good={'status':'ok','model':'qwen36-35b','maintenance':'serving','instance_id':'generation-a'}
route={'ready':True,'tunnel':{'ready':True,'instance_id':'generation-a'},'https':{'ready':True}}
served_ok={'served_context':131072,'profile_context':131072,'stats_instance_id':'generation-a'}
with tempfile.TemporaryDirectory() as temp:
    m.ROOT=pathlib.Path(temp)
    # Served-profile lookups are mocked: no /v1/stats request leaves this fixture.
    with patch.object(m,'load_key',return_value='synthetic-fixture-only'),patch.object(m,'served_profile',return_value=dict(served_ok)):
        with patch.object(m,'backend_health',return_value=dict(good)),patch.object(m,'edge_health',return_value=json.loads(json.dumps(route))):
            check('full route ready',m.collect(True)['ready'])
        with patch.object(m,'backend_health',return_value={**good,'model':'wrong-model'}):
            check('wrong model fails readiness',not m.collect()['ready'])
        with patch.object(m,'backend_health',return_value={**good,'status':'loading'}):
            check('loading is not ready',not m.collect()['ready'])
        for failed in ('tunnel','https'):
            bad=json.loads(json.dumps(route));bad[failed]['ready']=False;bad['ready']=False
            with patch.object(m,'backend_health',return_value=dict(good)),patch.object(m,'edge_health',return_value=bad):
                check(f'{failed} down fails complete readiness',not m.collect(True)['ready'])
        bad=json.loads(json.dumps(route));bad['tunnel']['instance_id']='generation-b'
        with patch.object(m,'backend_health',return_value=dict(good)),patch.object(m,'edge_health',return_value=bad):
            result=m.collect(True);check('tunnel generation mismatch',not result['ready'] and result['route']['tunnel']['error_type']=='generation_mismatch')
        with patch.object(m,'backend_health',side_effect=ConnectionError),patch.object(m,'edge_health',return_value=json.loads(json.dumps(route))) as edge:
            result=m.collect(True);check('route still inspected while backend down',edge.call_count==1 and not result['ready'])
        for local,remote in ((None,'generation-a'),('generation-a',None),('',''),(None,None)):
            bad=json.loads(json.dumps(route));bad['tunnel']['instance_id']=remote
            backend={**good,'instance_id':local} if local is not None else {k:v for k,v in good.items() if k!='instance_id'}
            with patch.object(m,'backend_health',return_value=backend),patch.object(m,'edge_health',return_value=bad):
                result=m.collect(True)
                check(f'missing instance identity is not a route match {local!r}/{remote!r}',not result['ready'] and result['route']['tunnel']['error_type']=='generation_unverified')
        (m.ROOT/'config').mkdir();(m.ROOT/'config/selected-profile').write_text('main128\n')
        with patch.object(m,'backend_health',return_value=dict(good)):
            result=m.collect()
            check('served context equal to selected profile is verified',result['ready'] and result['profile']=='main128' and result['profile_verified'] is True and result['served_context']==131072)
        for change in ({'served_context':262144},{'profile_context':None},{'stats_instance_id':'generation-b'},{'stats_instance_id':None}):
            with patch.object(m,'backend_health',return_value=dict(good)),patch.object(m,'served_profile',return_value={**served_ok,**change}):
                result=m.collect()
                check(f'served profile mismatch is not verified {sorted(change)}',result['profile_verified'] is False and result['ready'])
        with patch.object(m,'backend_health',return_value=dict(good)),patch.object(m,'served_profile',side_effect=ConnectionError('synthetic-secret-marker')):
            result=m.collect();check('stats failure leaves profile unverified and sanitized',result['profile_verified'] is False and result['profile_verify_error_type']=='ConnectionError' and 'synthetic-secret-marker' not in json.dumps(result))
        with patch.object(m,'backend_health',return_value={**good,'status':'loading'}),patch.object(m,'served_profile',side_effect=AssertionError('must not be called')) as served:
            result=m.collect();check('unready backend skips stats request',served.call_count==0 and result['profile_verified'] is False)
    with patch.object(m,'load_key',side_effect=FileNotFoundError):
        check('missing credential handled without traceback',m.collect()['error_type']=='FileNotFoundError')
    reply=types.SimpleNamespace(returncode=0,stdout=json.dumps(route),stderr='')
    with patch.object(m.subprocess,'run',return_value=reply) as run:
        result=m.edge_health('synthetic-fixture-only')
        argv=run.call_args.args[0]
        check('SSH retains strict checking and token only on stdin',result['ready'] and 'StrictHostKeyChecking=yes' in argv and 'synthetic-fixture-only' not in ' '.join(argv) and run.call_args.kwargs['input']=='synthetic-fixture-only')
    with patch.object(m.subprocess,'run',return_value=types.SimpleNamespace(returncode=1,stdout='synthetic-secret-marker',stderr='synthetic-secret-marker')):
        result=m.edge_health('synthetic-fixture-only');check('SSH stderr and stdout excluded from failure report','synthetic-secret-marker' not in json.dumps(result) and not result['ready'])
    class Response:
        status=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):return json.dumps({'status':'error','message':'synthetic-sensitive-trace','instance_id':'x'}).encode()
    with patch.object(m.OPENER,'open',return_value=Response()):
        result=m.backend_health('synthetic-fixture-only');check('backend exception detail redacted','synthetic-sensitive-trace' not in json.dumps(result))
    (m.ROOT/'profiles.json').write_text(json.dumps({'schema':1,'default':'main128','profiles':{'main128':{'context':131072},'long256':{'context':262144}}}))
    class Stats(Response):
        def read(self,n):return json.dumps({'instance_id':'generation-a','model':{'id':'qwen36-35b','ctx':131072},'private':'synthetic-sensitive-trace'}).encode()
    with patch.object(m.OPENER,'open',return_value=Stats()) as opened:
        served=m.served_profile('synthetic-fixture-only','main128')
        request=opened.call_args.args[0]
        check('served profile reads loopback stats with profiles.json context',served=={'served_context':131072,'profile_context':131072,'stats_instance_id':'generation-a'}
              and request.full_url=='http://127.0.0.1:1919/v1/stats' and 'synthetic-sensitive-trace' not in json.dumps(served))
        check('unknown selected profile has no expected context',m.served_profile('synthetic-fixture-only','legacy64')['profile_context'] is None)
report={'result':'PASS','tests':results,'real_network':'NOT_RUN','real_credentials':'NOT_READ'}
(BASE/'outputs/qwen36-api/optimization/ops-readiness-tests.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
