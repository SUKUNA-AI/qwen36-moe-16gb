"""Real HTTP overload, slow body and cancellation checks against the isolated app."""
import asyncio,json,pathlib,time
import httpx
from bench import ROOT,headers

async def main():
    rows=[];auth=headers();base='http://127.0.0.1:1919'
    async with httpx.AsyncClient(timeout=30) as client:
        r=await client.post(base+'/v1/chat/completions',headers=auth,content='{"model":"qwen36-35b","messages":[{"role":"user","content":"x"}],"temperature":1e999}')
        rows.append({'case':'overflow-number','pass':r.status_code==400,'status':r.status_code})
        r=await client.post(base+'/v1/chat/completions',headers=auth,content=b'x'*(2*1024*1024+1))
        rows.append({'case':'body-limit','pass':r.status_code==413,'status':r.status_code})
        # Eight bodies occupy the admission budget; the ninth is refused immediately.
        sockets=[]
        for i in range(8):
            reader,writer=await asyncio.open_connection('127.0.0.1',1919)
            header=('POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: '+auth['Authorization']+'\r\nContent-Length: 300\r\nContent-Type: application/json\r\n\r\n{')
            writer.write(header.encode());await writer.drain();sockets.append((reader,writer))
        await asyncio.sleep(.5)
        started=time.monotonic();r=await client.get(base+'/v1/models',headers=auth)
        rows.append({'case':'eight-inflight','pass':r.status_code==503,'status':r.status_code,'latency_s':time.monotonic()-started})
        first=await asyncio.wait_for(sockets[0][0].read(1024),18)
        rows.append({'case':'slow-body-deadline','pass':b'408' in first.split(b'\r\n',1)[0]})
        for reader,writer in sockets:writer.close();await writer.wait_closed()
        await asyncio.sleep(1)
        payload={'model':'qwen36-35b','messages':[{'role':'user','content':'Продолжай подробно объяснять алгоритмы сортировки, не завершай ответ.'}],
                 'max_tokens':8192,'temperature':.6,'seed':21,'ignore_eos':True,'chat_template_kwargs':{'enable_thinking':False},'stream':True}
        async with client.stream('POST',base+'/v1/chat/completions',headers=auth,json=payload) as stream:
            async for line in stream.aiter_lines():
                if line.startswith('data: ') and 'content' in line:break
            r=await client.post(base+'/v1/chat/completions',headers=auth,json=payload)
            rows.append({'case':'no-inference-queue','pass':r.status_code==429,'status':r.status_code})
        started=time.monotonic();ready=False
        for _ in range(100):
            r=await client.get(base+'/v1/stats',headers=auth)
            doc=r.json()
            # Native stats are nested differently across versions; the followup
            # generation below is the authoritative slot-release check.
            await asyncio.sleep(.1)
            small={**payload,'stream':False,'max_tokens':32,'ignore_eos':False,'messages':[{'role':'user','content':'Сколько будет 2+2?'}]}
            r=await client.post(base+'/v1/chat/completions',headers=auth,json=small)
            if r.status_code==200:ready=True;break
            if r.status_code!=429:break
        rows.append({'case':'sse-cancel-releases-slot','pass':ready,'recovery_s':time.monotonic()-started})
        payload['stream']=False
        task=asyncio.create_task(client.post(base+'/v1/chat/completions',headers=auth,json=payload))
        await asyncio.sleep(2);task.cancel()
        try:await task
        except asyncio.CancelledError:pass
        started=time.monotonic()
        for _ in range(30):
            await asyncio.sleep(.25)
            r=await client.post(base+'/v1/chat/completions',headers=auth,json=small)
            if r.status_code!=429:break
        rows.append({'case':'nonstream-cancel-releases-slot','pass':r.status_code==200,'status':r.status_code,'recovery_s':time.monotonic()-started})
    (ROOT/'reports/resilience.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows));assert all(row['pass'] for row in rows)

if __name__=='__main__':asyncio.run(main())
