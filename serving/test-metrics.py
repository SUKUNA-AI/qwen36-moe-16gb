"""Privacy, observed token accounting, rejection/cancellation and log-retention tests."""
import asyncio,json,os,pathlib,tempfile,types
os.environ['QWEN36_API_KEY']='a'*43
from guard import APIGuard
from request_metrics import CURRENT,NumericLog,RequestMetrics

class Resources:
    def snapshot(self): return {'service_cgroup_memory_bytes':100,'gpu_total_used_bytes':200}

async def main():
    canary='PRIVATE_CANARY_MUST_NEVER_APPEAR_54801'
    with tempfile.TemporaryDirectory(prefix='qwen36-metrics-test-') as temp:
        active=asyncio.Event()
        async def app(scope,receive,send):
            await receive()
            record=CURRENT.get()
            record.acknowledge(types.SimpleNamespace(prompt_tokens_delta=17,completion_tokens_delta=1,cached_tokens=4))
            record.semantic(types.SimpleNamespace())
            if scope['path']=='/v1/chat/completions':
                active.set()
                await send({'type':'http.response.start','status':200,'headers':[]})
                await asyncio.sleep(60)
            else:
                await send({'type':'http.response.start','status':200,'headers':[]})
                await send({'type':'http.response.body','body':canary.encode()})
        measured=RequestMetrics(APIGuard(app),log_root=temp,resources=Resources(),monitor=False)
        async def call(path,auth=None,body=b'',disconnect=None):
            first=True;events=[]
            async def receive():
                nonlocal first
                if first:first=False;return {'type':'http.request','body':body}
                if disconnect:await disconnect.wait()
                else:await asyncio.sleep(60)
                return {'type':'http.disconnect'}
            async def send(message):events.append(message)
            scope={'type':'http','path':path,'method':'POST' if path=='/v1/chat/completions' else 'GET','headers':[]}
            if auth:scope['headers'].append((b'authorization',auth))
            await measured(scope,receive,send)
            return events
        token=b'Bearer '+b'a'*43
        await call('/v1/models')
        await call('/'+canary,token)
        await call('/v1/models',token)
        disconnect=asyncio.Event()
        body=json.dumps({'model':'qwen36-35b','messages':[{'role':'user','content':canary}],'stream':True,'max_tokens':16}).encode()
        running=asyncio.create_task(call('/v1/chat/completions',token,body,disconnect))
        await active.wait()
        await call('/v1/chat/completions',token,body)
        disconnect.set();await running
        await measured.aclose()
        raw=(pathlib.Path(temp)/'requests.jsonl').read_text()
        assert canary not in raw and 'a'*43 not in raw and 'messages' not in raw
        rows=[json.loads(line) for line in raw.splitlines()]
        assert any(x['status']==401 for x in rows)
        assert any(x['status']==429 for x in rows)
        cancelled=next(x for x in rows if x['outcome']=='cancelled')
        assert cancelled['prompt_tokens']==17 and cancelled['completion_tokens']==1 and not cancelled['counts_final']
        assert cancelled['ttft_generated_s'] is not None and cancelled['ttft_content_s'] is None
        assert CURRENT.get() is None
        assert (pathlib.Path(temp)/'requests.jsonl').stat().st_mode&0o777==0o600
        tiny=NumericLog(pathlib.Path(temp)/'rotation',max_bytes=80,backups=2)
        for i in range(30):tiny.write('requests.jsonl',{'index':i,'test':'numeric only'})
        assert len(list(tiny.root.iterdir()))==3
        print(json.dumps({'privacy_canary_absent':True,'key_absent':True,'partial_cancel_counts':True,'auth_and_overload_logged':True,'context_reset':True,'mode_0600':True,'rotation_bounded':True,'result':'PASS'}))
asyncio.run(main())
