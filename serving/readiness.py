"""Authenticated readiness with independent backend, tunnel and HTTPS results."""
import argparse
import json
import pathlib
import shlex
import ssl
import subprocess
import sys
import urllib.request

ROOT = pathlib.Path('/home/USER/qwen36-api')
MODEL = 'qwen36-35b'

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def direct_opener():
    # A readiness credential belongs only to these exact endpoints. Do not
    # follow Location or let inherited proxy variables choose another receiver.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()))


OPENER = direct_opener()

REMOTE = '''import json,ssl,sys,urllib.request
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,request,fp,code,msg,headers,newurl):return None
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
key=sys.stdin.read().strip()
out={"source":"Caddy home LAN; not an external test"}
for name,url in [("tunnel","http://127.0.0.1:1919/health"),("https","https://llm.example.com/v1/models")]:
 try:
  with opener.open(urllib.request.Request(url,headers={"Authorization":"Bearer "+key}),timeout=6) as r:
   if r.status!=200:raise ValueError("unexpected readiness status")
   raw=r.read(1048577)
   if len(raw)>1048576:raise ValueError("oversized readiness response")
   d=json.loads(raw)
   ready=(d.get("status")=="ok" and d.get("model")=="qwen36-35b" and d.get("maintenance","serving")=="serving") if name=="tunnel" else any(x.get("id")=="qwen36-35b" for x in d.get("data",[]))
   out[name]={"http":r.status,"ready":ready}
   if name=="tunnel":out[name]["instance_id"]=d.get("instance_id")
 except Exception as e:out[name]={"ready":False,"error_type":type(e).__name__}
print(json.dumps(out))
'''

def load_key():
    for line in (ROOT/'secrets/api.env').read_text().splitlines():
        name, sep, value = line.partition('=')
        if sep and name.strip() == 'QWEN36_API_KEY':
            value = value.strip()
            if value and not any(ord(c) < 33 or ord(c) > 126 for c in value):
                return value
    raise ValueError('API credential is unavailable')

def backend_health(key):
    request = urllib.request.Request('http://127.0.0.1:1919/health',
                                     headers={'Authorization':'Bearer '+key})
    with OPENER.open(request, timeout=5) as response:
        if response.status != 200:
            raise ValueError('Unexpected readiness status')
        raw = response.read(1048577)
        if len(raw) > 1048576:
            raise ValueError('Oversized readiness response')
        data = json.loads(raw)
    # Engine error strings can contain model paths; retain only public state.
    allowed = ('status','phase','progress','model','instance_id','uptime_s','maintenance','version')
    health = {name:data[name] for name in allowed if name in data}
    if data.get('status') == 'error':
        health['error_type'] = 'engine_error'
    return health

def served_profile(key, selected):
    """Context served by the running process (/v1/stats) versus the selected profile."""
    request = urllib.request.Request('http://127.0.0.1:1919/v1/stats',
                                     headers={'Authorization':'Bearer '+key})
    with OPENER.open(request, timeout=5) as response:
        if response.status != 200:
            raise ValueError('Unexpected stats status')
        raw = response.read(1048577)
        if len(raw) > 1048576:
            raise ValueError('Oversized stats response')
        stats = json.loads(raw)
    card = stats.get('model') if isinstance(stats, dict) and isinstance(stats.get('model'), dict) else {}
    profiles = json.loads((ROOT/'profiles.json').read_text()).get('profiles', {})
    expected = profiles.get(selected, {}).get('context') if isinstance(profiles, dict) else None
    served = card.get('ctx')
    instance = stats.get('instance_id') if isinstance(stats, dict) else None
    return {'served_context':served if type(served) is int else None,
            'profile_context':expected if type(expected) is int else None,
            'stats_instance_id':instance if type(instance) is str else None}

def edge_health(key):
    try:
        result = subprocess.run(
            ['/mnt/c/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes',
             '-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=5',
             'EDGE_SSH_HOST','python3 -c '+shlex.quote(REMOTE)],
            input=key, text=True, capture_output=True, timeout=22)
        if result.returncode != 0:
            raise RuntimeError('SSH readiness probe failed')
        route = json.loads(result.stdout)
        if not isinstance(route, dict):
            raise ValueError('Invalid readiness response')
        route['ready'] = all(isinstance(route.get(name), dict) and route[name].get('ready') is True
                             for name in ('tunnel','https'))
        return route
    except Exception as exc:
        return {'ready':False,'error_type':type(exc).__name__,
                'source':'Caddy home LAN; not an external test'}

def collect(include_edge=False):
    try:
        key = load_key()
    except Exception as exc:
        return {'status':'unavailable','ready':False,'backend_ready':False,
                'error_type':type(exc).__name__}
    try:
        health = backend_health(key)
    except Exception as exc:
        health = {'status':'unavailable','error_type':type(exc).__name__}
    health['backend_ready'] = (health.get('status') == 'ok' and health.get('model') == MODEL
                               and health.get('maintenance','serving') == 'serving')
    health['ready'] = health['backend_ready']
    if include_edge:
        route = edge_health(key)
        local_id = health.get('instance_id')
        remote_id = route.get('tunnel', {}).get('instance_id')
        if local_id and remote_id and local_id != remote_id:
            route['tunnel'].update(ready=False,error_type='generation_mismatch')
            route['ready'] = False
        elif route['ready'] and not (type(local_id) is str and local_id and type(remote_id) is str and remote_id):
            # The route is attributed to this backend only by an equal, non-empty
            # instance identity on both sides; a missing identity is not a match.
            route['tunnel'].update(ready=False,error_type='generation_unverified')
            route['ready'] = False
        health['route'] = route
        health['ready'] = health['ready'] and route['ready']
    try:
        selected = ROOT/'config/selected-profile'
        health['profile'] = selected.read_text().strip() if selected.exists() else 'legacy64'
    except Exception as exc:
        health['profile_error_type'] = type(exc).__name__
    # The selection file alone does not prove what the running process serves.
    # profile_verified: served ctx equals the selected profile context, for the
    # same non-empty instance. Informational; it does not change 'ready'.
    health['profile_verified'] = False
    if health['backend_ready'] and 'profile' in health:
        try:
            served = served_profile(key, health['profile'])
            health['served_context'] = served['served_context']
            local_id = health.get('instance_id')
            health['profile_verified'] = (served['profile_context'] is not None
                and served['served_context'] == served['profile_context']
                and type(local_id) is str and bool(local_id) and served['stats_instance_id'] == local_id)
        except Exception as exc:
            health['profile_verify_error_type'] = type(exc).__name__
    return health

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edge', action='store_true')
    args = parser.parse_args()
    result = collect(args.edge)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['ready'] else 1

if __name__ == '__main__':
    sys.exit(main())
