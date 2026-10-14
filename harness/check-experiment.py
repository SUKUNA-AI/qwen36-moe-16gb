import json,pathlib,requests,sys,time,subprocess
root=pathlib.Path('/home/USER/qwen36-api/optimization')
until=time.monotonic()+int(sys.argv[1] if len(sys.argv)>1 else 1)
while True:
    try:
        key=(root/'config/experiment.env').read_text().strip().split('=',1)[1]
        response=requests.get('http://127.0.0.1:1919/health',headers={'Authorization':'Bearer '+key},timeout=3)
        if response.status_code==200 and response.json().get('status') in ('ok','ready'):
            print(json.dumps({'ready':True,'health':response.json()}));break
    except (OSError,requests.RequestException): pass
    active=subprocess.run(['systemctl','--user','is-active','--quiet','qwen36-experiment'],capture_output=True).returncode==0
    if not active:print(json.dumps({'ready':False,'reason':'experiment_exited'}));sys.exit(1)
    if time.monotonic()>=until: print(json.dumps({'ready':False}));sys.exit(1)
    time.sleep(1)
