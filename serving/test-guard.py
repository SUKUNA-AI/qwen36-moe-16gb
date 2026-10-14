import asyncio,importlib.util,json,os
os.environ['QWEN36_API_KEY']='a'*43
spec=importlib.util.spec_from_file_location('guard',os.path.join(os.path.dirname(__file__),'guard.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

async def main():
    active=asyncio.Event();cancelled=asyncio.Event()
    async def app(scope,receive,send):
        await receive()
        if scope['path']=='/v1/chat/completions':
            active.set()
            try: await asyncio.sleep(60)
            except asyncio.CancelledError: cancelled.set();raise
        await send({'type':'http.response.start','status':200,'headers':[]})
        await send({'type':'http.response.body','body':b'{}'})
    guarded=module.APIGuard(app)
    async def call(method,path,auth=None,body=b'',headers=None,disconnect=None):
        events=[];first=True
        async def receive():
            nonlocal first
            if first:first=False;return {'type':'http.request','body':body}
            if disconnect:await disconnect.wait()
            else:await asyncio.sleep(60)
            return {'type':'http.disconnect'}
        async def send(event):events.append(event)
        scope={'type':'http','method':method,'path':path,'headers':list(headers or [])}
        if auth:scope['headers'].append((b'authorization',auth))
        await guarded(scope,receive,send)
        return next((e['status'] for e in events if e['type']=='http.response.start'),None)
    token=b'Bearer '+b'a'*43
    assert await call('GET','/v1/models')==401
    assert await call('GET','/v1/models',b'Bearer incorrect')==401
    assert await call('GET','/v1/models',token)==200
    assert await call('GET','/docs',token)==404
    assert await call('POST','/v1/cache/rebuild',token)==404
    assert await call('POST','/v1/chat/completions',token,headers=[(b'content-length',b'3000000')])==413
    assert await call('POST','/v1/chat/completions',token,body=b'not json')==400
    body=json.dumps({'model':'qwen36-35b','messages':[{'role':'user','content':'test'}],'max_tokens':16}).encode()
    disconnect=asyncio.Event()
    running=asyncio.create_task(call('POST','/v1/chat/completions',token,body,disconnect=disconnect))
    await active.wait()
    assert await call('POST','/v1/chat/completions',token,body)==429
    disconnect.set();await running
    assert cancelled.is_set() and not guarded.busy
    print(json.dumps({'auth_missing':401,'auth_wrong':401,'auth_valid':200,'routes_closed':True,'body_limit':413,'invalid_json':400,'concurrent_rejected':429,'disconnect_cancels_app':True,'result':'PASS'}))
asyncio.run(main())
