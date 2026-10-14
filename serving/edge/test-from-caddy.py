"""Run on edge. Read key through SSH stdin; never place it in argv or an edge file."""
import json,ssl,sys,time,urllib.error,urllib.request
key=sys.stdin.read().split('=',1)[1].strip()
assert len(key)>=43
base=sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:1919'
def request(path,body=None,auth=key):
    h={} if auth is None else {'Authorization':'Bearer '+auth}
    if body is not None:h['Content-Type']='application/json';body=json.dumps(body).encode()
    req=urllib.request.Request(base+path,data=body,headers=h)
    try:
        with urllib.request.urlopen(req,timeout=900,context=ssl.create_default_context()) as r:return r.status,json.load(r)
    except urllib.error.HTTPError as e:return e.code,None
for auth in [None,'invalid-key']:
    status,_=request('/v1/models',auth=auth);assert status==401,status
    status,_=request('/v1/chat/completions',{'model':'qwen36-35b','messages':[{'role':'user','content':'Synthetic unauthorized test'}],'max_tokens':8},auth=auth)
    assert status==401,status
status,m=request('/v1/models');assert status==200 and m['data'][0]['id']=='qwen36-35b'
start=time.perf_counter()
status,out=request('/v1/chat/completions',{'model':'qwen36-35b','messages':[{'role':'user','content':'Ответь по-русски одним предложением: что проверяет этот запрос?'}],'max_tokens':64,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}})
assert status==200 and out['choices'][0]['message']['content']
print(json.dumps({'source':'Caddy host edge (home LAN)','base':base,'missing_auth_models_and_chat':401,'wrong_auth_models_and_chat':401,'models':200,'generation':200,'model':out['model'],'usage':out['usage'],'elapsed_s':time.perf_counter()-start,'answer':out['choices'][0]['message']['content'],'tls_verified':base.startswith('https://')},ensure_ascii=False))
