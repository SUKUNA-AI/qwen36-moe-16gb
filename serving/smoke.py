"""Authenticated synthetic acceptance checks. No secret is printed or serialized."""
import argparse, json, pathlib, time
import httpx

p=argparse.ArgumentParser()
p.add_argument('--base',default='http://127.0.0.1:1919')
p.add_argument('--out',required=True)
p.add_argument('--operational',action='store_true')
a=p.parse_args()
key=pathlib.Path('/home/USER/qwen36-api/secrets/api.env').read_text().split('=',1)[1].strip()
c=httpx.Client(base_url=a.base,headers={'Authorization':'Bearer '+key},timeout=900,trust_env=False)
local=httpx.Client(base_url='http://127.0.0.1:1919',headers={'Authorization':'Bearer '+key},timeout=10,trust_env=False)
results=[]
def record(name, **data):
    row={'test':name,**data};results.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
def chat(messages,**kw):
    payload={'model':'qwen36-35b','messages':messages,'max_tokens':256,'temperature':0,'chat_template_kwargs':{'enable_thinking':False},**kw}
    r=c.post('/v1/chat/completions',json=payload);r.raise_for_status();return r.json()
try:
    for auth in [None,'Bearer invalid-key']:
        headers={} if auth is None else {'Authorization':auth}
        r=httpx.get(a.base+'/v1/models',headers=headers,timeout=15,trust_env=False)
        assert r.status_code==401,r.status_code
    record('missing_wrong_auth',status=401)
    r=c.get('/v1/models');r.raise_for_status();models=r.json()
    assert any(m['id']=='qwen36-35b' for m in models['data'])
    record('models',model='qwen36-35b',context=models['data'][0].get('context_length'))
    for path in ['/','/docs','/openapi.json','/v1/requests','/v1/responses','/metrics','/v1/cache/status']:
        if a.base.startswith('http://127.') and path=='/v1/cache/status':continue
        assert c.get(path).status_code==404,path
    assert c.post('/v1/models',json={}).status_code==404
    assert c.get('/v1/chat/completions').status_code==404
    record('closed_routes',status=404)
    ru=chat([{'role':'user','content':'Ответь двумя предложениями по-русски: зачем проверять API?'}])
    content=ru['choices'][0]['message']['content']
    assert any('а'<=x.lower()<='я' for x in content)
    record('russian_nonstream',usage=ru['usage'],finish=ru['choices'][0]['finish_reason'],answer=content)
    code=chat([{'role':'user','content':'Напиши только Python-функцию add(a, b), которая возвращает сумму. Без запуска кода.'}],max_tokens=128)
    text=code['choices'][0]['message']['content'];assert 'def add' in text and 'return' in text
    record('code',usage=code['usage'],answer=text)
    tools=[{'type':'function','function':{'name':'fake_temperature','description':'Return a fixed synthetic temperature for a city; no network or real device access.','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city'],'additionalProperties':False}}}]
    messages=[{'role':'user','content':'Вызови fake_temperature для города Пермь. После результата инструмента сообщи температуру на русском.'}]
    first=chat(messages,tools=tools,tool_choice={'type':'function','function':{'name':'fake_temperature'}},max_tokens=512)
    msg=first['choices'][0]['message'];calls=msg.get('tool_calls');assert calls,'No tool call'
    call=calls[0];assert call['id'] and call['type']=='function' and call['function']['name']=='fake_temperature'
    arguments=json.loads(call['function']['arguments']);assert isinstance(arguments.get('city'),str)
    messages += [msg,{'role':'tool','tool_call_id':call['id'],'content':json.dumps({'temperature_c':7,'synthetic':True},ensure_ascii=False)}]
    final=chat(messages,tools=tools,tool_choice='none',max_tokens=256)
    answer=final['choices'][0]['message']['content'];assert '7' in answer
    record('tool_cycle',call_id=call['id'],arguments=arguments,first_usage=first['usage'],final_usage=final['usage'],answer=answer)
    payload={'model':'qwen36-35b','messages':[{'role':'user','content':'Напиши длинный связный текст на русском о тестировании программ, не менее 2000 слов.'}],'max_tokens':4096,'stream':True,'stream_options':{'include_usage':True},'temperature':0.6,'chat_template_kwargs':{'enable_thinking':False}}
    start=time.perf_counter();events=0;first_time=None
    with c.stream('POST','/v1/chat/completions',json=payload) as r:
        r.raise_for_status();assert r.headers['content-type'].startswith('text/event-stream')
        for line in r.iter_lines():
            if not line.startswith('data: ') or line=='data: [DONE]':continue
            event=json.loads(line[6:]);assert 'error' not in event,event
            text=''.join(x.get('delta',{}).get('content') or '' for x in event.get('choices',[]))
            if text:
                events+=1
                if first_time is None:first_time=time.perf_counter()
                if events==3:
                    if a.operational:
                        busy=c.post('/v1/chat/completions',json={'model':'qwen36-35b','messages':[{'role':'user','content':'Ответь: да.'}],'max_tokens':8})
                        assert busy.status_code==429,busy.status_code
                        record('bounded_overload',status=429,retry_after=busy.headers.get('retry-after'))
                    break
    assert events>=3
    closed=time.perf_counter();deadline=closed+15;active=None
    while time.perf_counter()<deadline:
        stats=local.get('/v1/stats');stats.raise_for_status();active=stats.json().get('requests',{}).get('active')
        if active==0:break
        time.sleep(.1)
    assert active==0,('Generation failed to cancel',active)
    record('sse_and_cancel',content_events=events,ttft_content_s=first_time-start,cancel_ack_s=time.perf_counter()-closed,active_after=active)
    recovered=chat([{'role':'user','content':'Ответь одним словом: готово.'}],max_tokens=32)
    assert recovered['choices'][0]['message']['content']
    record('slot_recovered',usage=recovered['usage'])
    record('body_limit',status=c.post('/v1/chat/completions',content=b' '*((2*1024*1024)+1)).status_code)
    assert results[-1]['status']==413
    pathlib.Path(a.out).write_text(json.dumps({'base':a.base,'results':results,'passed':True},ensure_ascii=False,indent=2),encoding='utf-8')
finally:
    c.close();local.close()
