"""Live ops acceptance helpers for qwen36. Default is PLAN_ONLY: no key, no network, no processes.

Run inside WSL as USER with the service venv, only after a released profile is
manually started and `qwen36 status` is ready. Every live mode needs --live.

  identity        Loopback /health + /v1/stats + unit state + selected-profile file.
                  Proves model id, non-empty equal instance ids, served context of the
                  requested profile (stats.model.ctx) and that the experiment is not active.
  overload        Through the existing SSH alias `EDGE_SSH_HOST` (key only on SSH stdin):
                  early ingress 429 for an unauthenticated burst, 413 for a >2 MiB body,
                  inference-busy 429 (+Retry-After, JSON, no SSE) while an own SSE stream
                  is active, slot release after closing that stream and after a closed
                  nonstream connection. Release is a FAIL criterion: <= 5 s after the
                  close AND before the own request (max_tokens 8192, ignore_eos) could
                  finish naturally (>= 27 s even at 300 tok/s), so natural completion
                  can never pass as a cancel. Never more than one own generation at a
                  time; the nonstream case is skipped (FAIL) if the SSE release failed.
  keepalive-gone  Read-only: for every private keepalive receipt of the given owner
                  generations, the exact Linux helper identity is no longer alive
                  (inspect-keepalive.verify expect=gone). No signals.

Reports: new 0600 JSON (O_EXCL) in /home/USER/qwen36-api/optimization/reports.
Never stored: API key, Authorization, prompts, answers, response bodies, error strings.
This script never starts/stops units, sends signals or touches Caddy/DNS/firewall.
"""
import argparse
import datetime
import importlib.util
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import urllib.request
import uuid

ROOT = pathlib.Path('/home/USER/qwen36-api')
REPORTS = ROOT / 'optimization' / 'reports'
KEEPALIVE_ROOT = ROOT / 'run' / 'keepalive'
LOCAL = 'http://127.0.0.1:1919'
MODEL = 'qwen36-35b'
CONTEXTS = {'baseline64': 65536, 'main128': 131072, 'long256': 262144}
UNITS = ('qwen36-inference.service', 'qwen36-experiment.service')
GENERATION = re.compile(r'[a-f0-9]{32}')
SSH = '/mnt/c/Windows/System32/OpenSSH/ssh.exe'


class Refused(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def direct_opener():
    # The credential is for 127.0.0.1:1919 only: no inherited proxy, no redirects.
    import ssl
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
                                       urllib.request.HTTPSHandler(context=ssl.create_default_context()))


OPENER = direct_opener()


def load_key():
    spec = importlib.util.spec_from_file_location('ops_live_readiness', ROOT / 'readiness.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_key()


def local_json(key, path, timeout=8):
    request = urllib.request.Request(LOCAL + path, headers={'Authorization': 'Bearer ' + key})
    with OPENER.open(request, timeout=timeout) as response:
        if response.status != 200:
            raise Refused('unexpected_status')
        raw = response.read(1048577)
    if len(raw) > 1048576:
        raise Refused('oversized_response')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise Refused('response_schema')
    return value


def unit_states(run=subprocess.run):
    names = ('LoadState', 'ActiveState', 'SubState', 'MainPID', 'NRestarts', 'InvocationID')
    states = {}
    for unit in UNITS:
        done = run(['systemctl', '--user', 'show', unit, *['--property=' + n for n in names]],
                   text=True, capture_output=True, timeout=10)
        if done.returncode != 0:
            raise Refused('systemctl_show_failed')
        values = {}
        for line in done.stdout.splitlines():
            if '=' not in line:
                continue
            name, value = line.split('=', 1)
            if name in values:
                raise Refused('duplicate_systemctl_field')
            values[name] = value
        if any(name not in values for name in ('LoadState', 'ActiveState', 'SubState')):
            raise Refused('incomplete_systemctl_state')
        states[unit] = values
    return states


def read_selected_profile():
    path = ROOT / 'config' / 'selected-profile'
    if not path.exists():
        return None
    with path.open('rb') as handle:
        raw = handle.read(65)
    return raw.decode('ascii').strip()


def identity(key, profile, expect_instance=None, different_from=None,
             json_get=local_json, states=unit_states, selected=read_selected_profile):
    """Fail-closed identity of the running service for the acceptance profile."""
    if profile not in CONTEXTS:
        raise Refused('unknown_profile')
    rows = []

    def check(name, condition, **facts):
        rows.append({'check': name, 'pass': bool(condition), **facts})

    health = json_get(key, '/health')
    stats = json_get(key, '/v1/stats')
    units = states()
    inference, experiment = units[UNITS[0]], units[UNITS[1]]
    instance = health.get('instance_id')
    card = stats.get('model') if isinstance(stats.get('model'), dict) else {}
    requests = stats.get('requests') if isinstance(stats.get('requests'), dict) else {}
    check('health_ok', health.get('status') == 'ok' and health.get('maintenance', 'serving') == 'serving')
    check('health_model_id', health.get('model') == MODEL)
    check('instance_id_present', isinstance(instance, str) and len(instance) > 0)
    check('stats_same_instance', isinstance(instance, str) and stats.get('instance_id') == instance)
    check('stats_model_id', card.get('id') == MODEL)
    check('served_context_matches_profile', card.get('ctx') == CONTEXTS[profile],
          expected_ctx=CONTEXTS[profile], served_ctx=card.get('ctx') if type(card.get('ctx')) is int else None)
    chosen = selected()
    check('selected_profile_file_matches', chosen == profile, selected_profile=chosen if chosen in CONTEXTS else None)
    check('inference_unit_active', inference.get('ActiveState') == 'active')
    check('experiment_not_active', experiment.get('ActiveState') in ('inactive', 'failed'))
    if expect_instance is not None:
        check('instance_equals_expected', instance == expect_instance)
    if different_from is not None:
        check('instance_differs_from_previous', isinstance(instance, str) and instance != different_from)
    active = requests.get('active')
    return {'mode': 'identity', 'profile': profile, 'rows': rows,
            'instance_id': instance if isinstance(instance, str) else None,
            'requests_active': active if type(active) is int else None,
            'inference_unit': {k: inference.get(k) for k in ('ActiveState', 'SubState', 'MainPID', 'NRestarts', 'InvocationID')},
            'experiment_unit': {k: experiment.get(k) for k in ('LoadState', 'ActiveState', 'SubState')},
            'pass': all(row['pass'] for row in rows)}


# Executed on the Caddy host `EDGE_SSH_HOST` with `python3 -c`; key arrives on stdin.
# Stdlib only. Never prints request/response bodies, headers or the key.
REMOTE = r'''
import json,select,socket,ssl,sys,time,urllib.error,urllib.request
HTTPS='https://llm.example.com'
TUNNEL='http://127.0.0.1:1919'
MODEL='qwen36-35b'
MAX_BODY=2*1024*1024
# Client-close slot release is a FAIL criterion, not a guideline: the confirming
# 200 must arrive <= RELEASE_DEADLINE s after the own connection was closed.
RELEASE_DEADLINE=5.0
# The own request cannot finish by itself inside that window: guard maximum
# output budget with ignore_eos. Even at DECODE_CEILING_TPS (well above the
# measured 72-131 tok/s) natural completion needs >= NATURAL_FLOOR_S from the
# request start, so a release observed earlier is attributable to the cancel.
OWN_MAX_TOKENS=8192
DECODE_CEILING_TPS=300.0
NATURAL_FLOOR_S=OWN_MAX_TOKENS/DECODE_CEILING_TPS
NONSTREAM_HOLD_S=4.0
FIRST_CHUNK_DEADLINE=90.0
EARLY_WAIT=1.5
LONG_PAYLOAD={'model':MODEL,'messages':[{'role':'user','content':'Напиши подробный нумерованный список из 300 пунктов об алгоритмах сортировки.'}],
              'max_tokens':OWN_MAX_TOKENS,'ignore_eos':True,'temperature':0,'seed':29,'chat_template_kwargs':{'enable_thinking':False}}
clock=time.monotonic
pause=time.sleep
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,newurl):return None
CONTEXT=ssl.create_default_context()
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect(),urllib.request.HTTPSHandler(context=CONTEXT))
def call(key,base,path,body=None,auth=True,timeout=30):
    headers={'Authorization':'Bearer '+key} if auth else {}
    if body is not None:headers['Content-Type']='application/json'
    request=urllib.request.Request(base+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
    try:return OPENER.open(request,timeout=timeout)
    except urllib.error.HTTPError as error:return error
def content_type(response):
    return (response.headers.get('Content-Type') or '').split(';',1)[0].strip().lower()
def error_message(response):
    if content_type(response)!='application/json':return None
    raw=response.read(65537)
    if len(raw)>65536:return None
    try:value=json.loads(raw)
    except ValueError:return None
    error=value.get('error') if isinstance(value,dict) else None
    return error.get('message') if isinstance(error,dict) and isinstance(error.get('message'),str) else None
def split_base(base):
    scheme,rest=base.split('://',1)
    host,_,port=rest.partition(':')
    return scheme,host,int(port) if port else (443 if scheme=='https' else 80)
def raw_exchange(key,base,body_size,payload=None,hold_seconds=0.0,timeout=30.0):
    """Own raw HTTP/1.1 POST: either an oversized body streamed until the server answers,
    or a small JSON body whose connection is closed after hold_seconds (client cancel)."""
    scheme,host,port=split_base(base)
    sock=socket.create_connection((host,port),timeout=15)
    try:
        if scheme=='https':sock=CONTEXT.wrap_socket(sock,server_hostname=host)
        body=json.dumps(payload).encode() if payload is not None else None
        length=len(body) if body is not None else body_size
        head=('POST /v1/chat/completions HTTP/1.1\r\nHost: '+(host if port in (80,443) else host+':'+str(port))+
              '\r\nAuthorization: Bearer '+key+'\r\nContent-Type: application/json\r\nContent-Length: '+str(length)+
              '\r\nConnection: close\r\n\r\n').encode()
        sock.sendall(head)
        if body is not None:
            sock.sendall(body)
            pause(hold_seconds)
            return {'sent':len(body),'closed_by_client':True}
        # Wait briefly for an early answer to the declared Content-Length before
        # streaming the body: a server that rejects on the header and closes then
        # has no unread bytes, so its reply is not lost to a TCP reset.
        sent=0;response=b'';chunk=b'x'*16384;deadline=clock()+timeout;early=clock()+EARLY_WAIT
        send_error=None;recv_error=None
        while clock()<deadline and b'\r\n' not in response:
            pending=scheme=='https' and sock.pending()>0
            may_send=sent<body_size and send_error is None and clock()>=early
            readable,writable,_=select.select([sock],[sock] if may_send else [],[],1 if may_send else 0.2)
            if readable or pending:
                try:data=sock.recv(4096)
                except (ConnectionResetError,ssl.SSLError,OSError) as error:
                    recv_error=type(error).__name__;break
                if not data:break
                response+=data;continue
            if writable:
                try:sent+=sock.send(chunk[:min(len(chunk),body_size-sent)])
                except (BrokenPipeError,ConnectionResetError,ssl.SSLError) as error:send_error=type(error).__name__
        parts=response.split(b'\r\n',1)[0].split()
        status=int(parts[1]) if len(parts)>=2 and parts[1].isdigit() else None
        return {'status':status,'body_bytes_declared':body_size,'body_bytes_sent':sent,
                'send_error_type':send_error,'recv_error_type':recv_error}
    finally:
        try:sock.close()
        except OSError:pass
def open_stream(key,base,payload,timeout=120):
    response=call(key,base,'/v1/chat/completions',{**payload,'stream':True},timeout=timeout)
    if response.status!=200 or content_type(response)!='text/event-stream':
        response.close();raise ValueError('own_stream_not_started')
    started=clock()
    while clock()-started<FIRST_CHUNK_DEADLINE:
        line=response.readline(65537)
        if not line or len(line)>65536:break
        if line.startswith(b'data: ') and line.strip()!=b'data: [DONE]':
            try:event=json.loads(line[6:])
            except ValueError:continue
            if any((choice.get('delta') or {}).get('content') for choice in event.get('choices',[])):
                return response,clock()-started
    response.close();raise ValueError('own_stream_without_content')
def close_stream(response):
    try:
        import socket as s
        response.fp.raw._sock.shutdown(s.SHUT_RDWR)
    except (OSError,AttributeError):pass
    try:response.close()
    except Exception:pass
def busy_response(key,base,stream):
    started=clock()
    small={'model':MODEL,'messages':[{'role':'user','content':'Ответь одним словом: да.'}],'max_tokens':8,'temperature':0,'seed':5,
           'stream':stream,'chat_template_kwargs':{'enable_thinking':False}}
    with call(key,base,'/v1/chat/completions',small,timeout=30) as response:
        status=response.status;retry=response.headers.get('Retry-After');ctype=content_type(response)
        message=error_message(response) if status==429 else None
    return {'status':status,'expected_status':429,'retry_after_present':retry is not None,
            'json_error':message is not None,'sse_started':ctype=='text/event-stream','elapsed_s':round(clock()-started,3),
            'pass':status==429 and retry is not None and message is not None and ctype!='text/event-stream'}
def wait_slot_released(key,base,own_started):
    """Poll until a small request is admitted. PASS only when the confirming 200
    arrives <= RELEASE_DEADLINE after the client close AND before the own request
    could have completed naturally (own_started + NATURAL_FLOOR_S); both times
    are taken at the confirming response, which is conservative."""
    started=clock();attempts=0;last=None;rate_limited=0
    small={'model':MODEL,'messages':[{'role':'user','content':'Сколько будет 2+2? Ответь числом.'}],'max_tokens':16,'temperature':0,'seed':7,
           'chat_template_kwargs':{'enable_thinking':False}}
    facts={'release_deadline_s':RELEASE_DEADLINE,'natural_completion_floor_s':round(NATURAL_FLOOR_S,3)}
    while clock()-started<RELEASE_DEADLINE:
        attempts+=1
        with call(key,base,'/v1/chat/completions',small,timeout=60) as response:
            last=response.status
            if last==200:
                now=clock()
                raw=response.read(1048577)
                data=json.loads(raw) if len(raw)<=1048576 else {}
                release_s=now-started;own_age=now-own_started
                attributable=release_s<=RELEASE_DEADLINE and own_age<NATURAL_FLOOR_S
                return {**facts,'status':200,'model_alias_valid':data.get('model')==MODEL,'release_s':round(release_s,3),
                        'own_request_age_s':round(own_age,3),'release_attributable_to_cancel':attributable,
                        'attempts':attempts,'rate_limited_polls':rate_limited,'pass':data.get('model')==MODEL and attributable}
            if last==429 and error_message(response)=='Request rate exceeded':rate_limited+=1
        if last!=429:break
        pause(0.25)
    return {**facts,'status':last,'release_s':round(clock()-started,3),'release_attributable_to_cancel':False,
            'attempts':attempts,'rate_limited_polls':rate_limited,'pass':False}
def burst(base,count=40):
    statuses={};ingress_messages=0
    for _ in range(count):
        with call('',base,'/v1/models',auth=False,timeout=10) as response:
            statuses[response.status]=statuses.get(response.status,0)+1
            if response.status==429 and error_message(response)=='Request rate exceeded':ingress_messages+=1
    pause(3.0)
    allowed=set(statuses)<={401,429}
    return {'statuses':{str(k):v for k,v in sorted(statuses.items())},'ingress_rate_messages':ingress_messages,
            'pass':allowed and statuses.get(429,0)>=1 and ingress_messages>=1 and statuses.get(401,0)>=1}
def collect(key):
    rows=[]
    def checked(base,case,fn):
        row={'base':base,'case':case,'pass':False}
        try:
            facts=fn()
            row.update(facts)
            row['pass']=facts.get('pass') is True
        except Exception as error:
            row.update(error_type=type(error).__name__,transport_or_schema_failure=True)
        rows.append(row);return row
    checked(TUNNEL,'unauthenticated-burst-ingress-429',lambda:burst(TUNNEL))
    for base in (TUNNEL,HTTPS):
        def oversized(base=base):
            facts=raw_exchange(key,base,MAX_BODY+1)
            facts['expected_status']=413
            facts['pass']=facts['status']==413
            return facts
        checked(base,'body-over-2MiB-413',oversized)
    stream=None;sse_started=clock()
    try:
        holder={}
        def start():
            response,ttft=open_stream(key,HTTPS,LONG_PAYLOAD);holder['response']=response
            return {'pass':True,'first_content_s':round(ttft,3),'own_max_tokens':OWN_MAX_TOKENS,'ignore_eos':True}
        started=checked(HTTPS,'own-sse-stream-started',start)
        stream=holder.get('response')
        if started['pass']:
            checked(HTTPS,'busy-429-nonstream',lambda:busy_response(key,HTTPS,False))
            checked(HTTPS,'busy-429-stream-before-sse',lambda:busy_response(key,HTTPS,True))
    finally:
        if stream is not None:close_stream(stream)
    sse_release=checked(HTTPS,'sse-client-close-releases-slot',lambda:wait_slot_released(key,HTTPS,sse_started))
    def nonstream_cancel():
        # Own nonstream request on a raw connection; the client closes it after
        # NONSTREAM_HOLD_S. Busy must be observed while it is active, then released.
        facts={'own_max_tokens':OWN_MAX_TOKENS,'ignore_eos':True};holder={}
        def hold():
            try:holder['facts']=raw_exchange(key,HTTPS,0,payload={**LONG_PAYLOAD,'stream':False},hold_seconds=NONSTREAM_HOLD_S)
            except Exception as error:holder['error_type']=type(error).__name__
        import threading
        own_started=clock()
        worker=threading.Thread(target=hold,daemon=True);worker.start()
        pause(2.0)
        busy=busy_response(key,HTTPS,False)
        worker.join(30)
        facts['busy_while_own_nonstream_active']=busy['pass']
        facts['own_request_error_type']=holder.get('error_type')
        released=wait_slot_released(key,HTTPS,own_started)
        facts.update({name:released.get(name) for name in ('release_s','own_request_age_s','release_deadline_s',
                      'natural_completion_floor_s','release_attributable_to_cancel','rate_limited_polls')})
        facts['release_status']=released.get('status')
        facts['pass']=busy['pass'] and released['pass'] and 'facts' in holder and not worker.is_alive()
        return facts
    if sse_release['pass']:
        checked(HTTPS,'nonstream-client-close-releases-slot',nonstream_cancel)
    else:
        # The slot may still be held by the own SSE generation (cancel not
        # proven). Do not stack a second long generation; busy/release here
        # would not be attributable to the nonstream close.
        rows.append({'base':HTTPS,'case':'nonstream-client-close-releases-slot','pass':False,
                     'skipped':'previous_release_failed_slot_may_be_held'})
    return {'source':'Caddy edge on home LAN; NOT an external end-to-end test','external_e2e':'NOT_RUN',
            'redirects_followed':False,'rows':rows,'pass':all(row['pass'] for row in rows)}
'''


def remote_overload(key, run=subprocess.run):
    command = 'python3 -c ' + shlex.quote(REMOTE + '\nprint(json.dumps(collect(sys.stdin.read().strip())))')
    result = run([SSH, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=5', 'EDGE_SSH_HOST', command],
                 input=key, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise Refused('caddy_probe_process_failed')
    report = json.loads(result.stdout)
    if not isinstance(report, dict) or type(report.get('pass')) is not bool or not isinstance(report.get('rows'), list):
        raise Refused('invalid_remote_report')
    return report


def overload(key, profile, json_get=local_json, states=unit_states, selected=read_selected_profile, remote=remote_overload):
    before = identity(key, profile, json_get=json_get, states=states, selected=selected)
    if not before['pass']:
        return {'mode': 'overload', 'pass': False, 'scope': 'IDENTITY_PRECONDITION_FAILED', 'identity_before': before}
    if before['requests_active'] != 0:
        return {'mode': 'overload', 'pass': False, 'scope': 'SERVICE_NOT_IDLE', 'identity_before': before}
    try:
        report = remote(key)
    except Exception as error:
        report = {'pass': False, 'rows': [], 'error_type': type(error).__name__, 'scope': 'SSH_OR_REMOTE_CHECK_FAILURE',
                  'external_e2e': 'NOT_RUN'}
    after = identity(key, profile, expect_instance=before['instance_id'], json_get=json_get, states=states, selected=selected)
    slot_idle = after['requests_active'] == 0
    report.update(mode='overload', identity_before=before, identity_after=after, slot_idle_after=slot_idle,
                  same_instance_after=after['pass'])
    report['pass'] = bool(report.get('pass')) and after['pass'] and slot_idle
    return report


def load_inspector():
    path = pathlib.Path(__file__).with_name('inspect-keepalive.py')
    spec = importlib.util.spec_from_file_location('ops_live_inspector', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def keepalive_gone(generations, wait_seconds=45, root=KEEPALIVE_ROOT, inspector=None):
    inspector = inspector or load_inspector()
    rows = []
    if not generations:
        raise Refused('no_generations')
    for generation in generations:
        if not GENERATION.fullmatch(generation):
            raise Refused('invalid_generation')
        directory = root / generation
        try:
            inspector.private(root, True)
            inspector.private(directory, True)
            paths = sorted(directory.glob('*.json'))
        except (OSError, ValueError) as error:
            rows.append({'generation': generation, 'pass': False, 'error_type': type(error).__name__})
            continue
        if not paths or len(paths) > 64:
            rows.append({'generation': generation, 'pass': False, 'error_type': 'missing_or_too_many_receipts',
                         'receipts': len(paths)})
            continue
        for path in paths:
            row = {'generation': generation, 'pass': False}
            try:
                inspector.private(path, False)
                with path.open('rb') as handle:
                    raw = handle.read(16385)
                if len(raw) > 16384:
                    raise ValueError('receipt_size_limit')
                record = json.loads(raw)
                lease = record.get('lease_id')
                if not isinstance(lease, str) or not GENERATION.fullmatch(lease):
                    raise ValueError('invalid_lease')
                result = inspector.verify(generation, lease, 'gone', wait_seconds, root)
                row.update(lease_id=lease, linux_pid=result.get('linux_pid'), exact_identity_alive=result.get('exact_identity_alive'),
                           exit_reason=record.get('reason') if record.get('state') == 'exited' else None,
                           receipt_state=record.get('state') if record.get('state') in ('running', 'exited') else None,
                           started_utc=record.get('started_utc'), finished_utc=record.get('finished_utc'))
                row['pass'] = result.get('pass') is True
            except (OSError, ValueError, KeyError, TypeError) as error:
                row['error_type'] = type(error).__name__
            rows.append(row)
    return {'mode': 'keepalive-gone', 'wait_seconds': wait_seconds, 'rows': rows, 'signals_sent': 0,
            'read_only': True, 'pass': bool(rows) and all(row['pass'] for row in rows)}


def generations_from_ops_report(path):
    with open(path, 'rb') as handle:
        raw = handle.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise Refused('ops_report_too_large')
    report = json.loads(raw.decode('utf-8-sig'))
    found = []
    for item in report.get('FinalProcesses') or []:
        value = item.get('Generation') if isinstance(item, dict) else None
        if isinstance(value, str) and GENERATION.fullmatch(value) and value not in found:
            found.append(value)
    if not found:
        raise Refused('no_generations_in_ops_report')
    return found


def write_report(report, mode, report_dir=REPORTS, name=None):
    stamp = datetime.datetime.now(datetime.timezone.utc)
    name = name or f'ops-live-{mode}-{stamp.strftime("%Y%m%dT%H%M%SZ")}-{uuid.uuid4().hex[:8]}'
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', name):
        raise Refused('invalid_report_name')
    report_dir = pathlib.Path(report_dir)
    report_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = report_dir / (name + '.json')
    report = {**report, 'recorded_utc': stamp.isoformat(), 'external_e2e': 'NOT_RUN'}
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    return target


def plan():
    return {'result': 'PLAN_ONLY', 'live': 'NOT_RUN', 'modes': {
        'identity': 'loopback /health + /v1/stats; ctx == profile; instance equal & non-empty; selected-profile; units',
        'overload': 'edge via SSH stdin key: burst 429, 413 >2MiB (tunnel+HTTPS), busy 429 nonstream/stream, SSE and nonstream client-close slot release <=5 s and before natural completion (max_tokens 8192 + ignore_eos)',
        'keepalive-gone': 'read-only exact helper identities gone for given generations (no signals)'},
        'service_actions': False, 'signals': 0, 'external_e2e': 'NOT_RUN'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', nargs='?', choices=('identity', 'overload', 'keepalive-gone'))
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--profile', choices=sorted(CONTEXTS))
    parser.add_argument('--expect-instance')
    parser.add_argument('--different-from')
    parser.add_argument('--generation', action='append', default=[])
    parser.add_argument('--from-ops-report')
    parser.add_argument('--wait-seconds', type=int, default=45)
    parser.add_argument('--report-name')
    args = parser.parse_args(argv)
    if not args.live or args.mode is None:
        print(json.dumps(plan()))
        return 0
    if sys.platform != 'linux' or os.geteuid() == 0:
        raise SystemExit('Run inside WSL as the ordinary service user')
    if args.mode in ('identity', 'overload') and not args.profile:
        raise SystemExit('--profile is required')
    if not 0 <= args.wait_seconds <= 60:
        raise SystemExit('--wait-seconds must be 0..60')
    try:
        if args.mode == 'keepalive-gone':
            generations = list(args.generation)
            if args.from_ops_report:
                generations += [g for g in generations_from_ops_report(args.from_ops_report) if g not in generations]
            report = keepalive_gone(generations, args.wait_seconds)
        else:
            key = load_key()
            if args.mode == 'identity':
                report = identity(key, args.profile, args.expect_instance, args.different_from)
            else:
                report = overload(key, args.profile)
    except Exception as error:
        report = {'mode': args.mode, 'pass': False, 'error_type': type(error).__name__,
                  'error_code': str(error) if isinstance(error, Refused) else None}
    target = write_report(report, args.mode, name=args.report_name)
    print(json.dumps({**report, 'report': str(target)}))
    return 0 if report.get('pass') is True else 1


if __name__ == '__main__':
    raise SystemExit(main())
