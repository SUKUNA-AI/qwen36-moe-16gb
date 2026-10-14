import json,pathlib,time,urllib.error,urllib.request,sys
key=pathlib.Path('/home/USER/qwen36-api/secrets/api.env').read_text().split('=',1)[1].strip()
deadline=time.monotonic()+300;before=None
while time.monotonic()<deadline:
    try:
        req=urllib.request.Request('http://127.0.0.1:1919/health',headers={'Authorization':'Bearer '+key})
        with urllib.request.urlopen(req,timeout=5) as r:doc=json.load(r)
        current=(doc.get('status'),doc.get('phase'))
        if current!=before:print(json.dumps({'status':current[0],'phase':current[1]},ensure_ascii=False),flush=True);before=current
        if doc.get('status')=='ok' and doc.get('model')=='qwen36-35b':
            print(json.dumps({'ready':True,'instance_id':doc.get('instance_id')},ensure_ascii=False),flush=True);sys.exit(0)
        if doc.get('status')=='error':
            print('Worker failed; bounded service recovery may reload it',flush=True)
    except (OSError,ValueError):pass
    time.sleep(2)
print('Readiness timeout 300s',flush=True);sys.exit(1)
