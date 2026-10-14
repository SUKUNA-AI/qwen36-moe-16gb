"""Scoped Linux worker fault acceptance. Default is plan-only: no secrets/processes/API.

Run inside WSL using the existing service venv, with an exclusive live-test slot.
--live --case kill: idle GPU worker SIGKILL, bounded automatic recovery.
--live --case stall: active synthetic request, SIGSTOP, >300s stall detector/recovery.
No service start/stop, pkill, numeric-PID os.kill, or WSL operations are performed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import pathlib
import select
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

ROOT = pathlib.Path('/home/USER/qwen36-api')
UNIT = 'qwen36-inference.service'
CGROUP_ROOT = pathlib.Path('/sys/fs/cgroup')
BASE = 'http://127.0.0.1:1919'


class Refused(RuntimeError): pass


def proc_stat(pid):
    raw = (pathlib.Path('/proc')/str(pid)/'stat').read_text()
    # comm can contain spaces or parentheses; field 3 follows the final ') '.
    end = raw.rfind(') ')
    if end < 0: raise Refused('invalid_proc_stat')
    rest = raw[end+2:].split()
    return {'state':rest[0], 'ppid':int(rest[1]), 'starttime':int(rest[19])}


def service():
    names = ('ControlGroup','MainPID','ActiveState','NRestarts','InvocationID')
    done = subprocess.run(['systemctl','--user','show',UNIT,*['--property='+n for n in names]],
                          text=True,capture_output=True,timeout=10,check=True)
    values = dict(line.split('=',1) for line in done.stdout.splitlines() if '=' in line)
    if not all(name in values for name in names): raise Refused('incomplete_service_identity')
    values['MainPID'], values['NRestarts'] = int(values['MainPID']), int(values['NRestarts'])
    cg = values['ControlGroup']
    path = (CGROUP_ROOT/cg.lstrip('/')).resolve()
    if not cg.startswith(f'/user.slice/user-{os.getuid()}.slice/') or '..' in pathlib.PurePosixPath(cg).parts:
        raise Refused('unexpected_user_cgroup')
    if not path.is_relative_to(CGROUP_ROOT) or path.name != UNIT:
        raise Refused('unexpected_service_cgroup')
    if not values['InvocationID']: raise Refused('missing_service_invocation')
    return values


def member(pid, expected):
    lines = (pathlib.Path('/proc')/str(pid)/'cgroup').read_text().splitlines()
    return '0::'+expected in lines


def current_pids(snapshot):
    path = (CGROUP_ROOT/snapshot['ControlGroup'].lstrip('/')/'cgroup.procs').resolve()
    if not path.is_relative_to(CGROUP_ROOT): raise Refused('cgroup_path_escape')
    return [int(value) for value in path.read_text().split()]


def descendant_of(pid, main):
    for _ in range(16):
        if pid == main: return True
        if pid <= 1: return False
        pid = proc_stat(pid)['ppid']
    return False


def select_worker(snapshot, requested=None):
    candidates = []
    btime = next(int(line.split()[1]) for line in pathlib.Path('/proc/stat').read_text().splitlines()
                 if line.startswith('btime '))
    hz = os.sysconf('SC_CLK_TCK')
    executable = pathlib.Path(sys.executable).resolve()
    # PID candidates originate ONLY from this exact systemd cgroup. The two
    # corresponding placement files confirm a model role, never supply PIDs.
    for pid in current_pids(snapshot):
        if requested is not None and pid != requested: continue
        if pid == snapshot['MainPID']: continue
        proc = pathlib.Path('/proc')/str(pid)
        try:
            info = proc_stat(pid)
            if proc.stat().st_uid != os.getuid() or not member(pid,snapshot['ControlGroup']): continue
            if (proc/'exe').resolve() != executable: continue
            args = (proc/'cmdline').read_bytes().split(b'\0')
            if not any(arg.startswith(b'from multiprocessing.spawn import spawn_main') for arg in args): continue
            if not descendant_of(pid,snapshot['MainPID']): continue
            began = btime + info['starttime']/hz
            placement = ROOT/'reports'/f'placement-{pid}.json'
            cache = ROOT/'reports'/f'gpu-cache-{pid}.json'
            if placement.stat().st_mtime < began or cache.stat().st_mtime < began: continue
            p, c = json.loads(placement.read_text()), json.loads(cache.read_text())
            if p.get('pid') != pid or c.get('pid') != pid: continue
            if p.get('host_bank_bytes',0) <= 0 or c.get('gpu_bank_bytes',0) <= 0: continue
            if not any(str(device).startswith('cuda') for device in c.get('gpu_bank_devices',[])): continue
            candidates.append({'pid':pid,'starttime':info['starttime']})
        except (OSError,ValueError,KeyError): continue
    if len(candidates) != 1: raise Refused('model_worker_is_not_unique')
    return candidates[0]


def open_verified_pidfd(identity, snapshot):
    if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
        raise Refused('pidfd_api_unavailable')
    pid = identity['pid']
    before = proc_stat(pid)
    if before['starttime'] != identity['starttime']: raise Refused('pid_reused_before_open')
    fd = os.pidfd_open(pid,0)
    try:
        after = proc_stat(pid)
        if after['starttime'] != before['starttime'] or not member(pid,snapshot['ControlGroup']):
            raise Refused('pid_identity_changed_during_open')
        now = service()
        if any(now[name] != snapshot[name] for name in ('ControlGroup','MainPID','InvocationID')):
            raise Refused('service_generation_changed')
        signal.pidfd_send_signal(fd,0,None,0)
        return fd
    except BaseException:
        os.close(fd)
        raise


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,newurl):
        return None


# The Bearer key belongs only to BASE: ignore inherited proxy variables and
# never follow a Location header (same policy as readiness.py).
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())


def request(key,path,body=None,timeout=8):
    payload = None if body is None else json.dumps(body).encode()
    headers = {'Authorization':'Bearer '+key}
    if body is not None: headers['Content-Type']='application/json'
    return OPENER.open(urllib.request.Request(BASE+path,data=payload,headers=headers),timeout=timeout)


def json_request(key,path,body=None,timeout=8):
    with request(key,path,body,timeout) as response:
        raw = response.read(1048577)
        if len(raw)>1048576: raise Refused('oversized_response')
        return json.loads(raw)


def health(key):
    result = json_request(key,'/health')
    if result.get('status') != 'ok' or result.get('model') != 'qwen36-35b':
        raise Refused('model_not_ready')
    if result.get('maintenance','serving') != 'serving' or not result.get('instance_id'):
        raise Refused('engine_not_serving')
    return result


def active_count(key):
    result = json_request(key,'/v1/stats')
    value = result.get('requests',{}).get('active')
    if type(value) is not int: raise Refused('unknown_stats_active_schema')
    return value


def load_key():
    spec = importlib.util.spec_from_file_location('fault_readiness',ROOT/'readiness.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.load_key()


class ActiveStream:
    def __init__(self,key):
        self.key=key;self.started=threading.Event();self.done=threading.Event()
        self.response=None;self.error=None;self.bytes=0;self.request_id=None
        self.thread=threading.Thread(target=self.run,daemon=True,name='qwen36-owned-synthetic-stream')
    def run(self):
        body={'model':'qwen36-35b','messages':[{'role':'user','content':'Write a long numbered explanation of sorting algorithms with examples.'}],
              'max_tokens':8192,'ignore_eos':True,'stream':True,'temperature':0,'seed':17,
              'chat_template_kwargs':{'enable_thinking':False}}
        try:
            with request(self.key,'/v1/chat/completions',body,timeout=850) as response:
                self.response=response; self.request_id=response.headers.get('x-qwen36-request-id')
                for line in response:
                    self.bytes+=len(line)
                    if line.startswith(b'data: ') and line.strip()!=b'data: [DONE]':
                        try:
                            doc=json.loads(line[6:])
                            if any(choice.get('delta',{}).get('content') for choice in doc.get('choices',[])):
                                self.started.set()
                        except ValueError: pass
        except Exception as exc:self.error=type(exc).__name__
        finally:self.done.set()
    def begin(self):
        self.thread.start()
        if not self.started.wait(60) or self.done.is_set(): raise Refused('synthetic_stream_not_active')
    def close(self):
        if self.response is not None:
            # Socket shutdown wakes a blocked reader without waiting on its IO lock.
            try:
                import socket
                self.response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
            except (OSError,AttributeError): pass
        self.thread.join(5)


def arm_resume_guard(pidfd, seconds):
    """Independent child resumes ONLY this pidfd if tester exits or times out."""
    descriptors=set();guard=None;returned=False
    try:
        read_fd,write_fd=os.pipe();descriptors.update((read_fd,write_fd))
        ready_read,ready_write=os.pipe();descriptors.update((ready_read,ready_write))
        # No Python fork body after starting the HTTP reader thread. A clean
        # interpreter receives only the explicit pidfd and two pipe descriptors.
        guard=subprocess.Popen([sys.executable,str(pathlib.Path(__file__).resolve()),
            '--resume-guard',str(pidfd),str(read_fd),str(ready_write),str(seconds)],
            pass_fds=(pidfd,read_fd,ready_write),stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        os.close(read_fd);descriptors.remove(read_fd)
        os.close(ready_write);descriptors.remove(ready_write)
        if not select.select([ready_read],[],[],10)[0] or os.read(ready_read,1)!=b'R':
            raise Refused('resume_guard_failed_to_arm')
        returned=True;descriptors.remove(write_fd)
        return guard,write_fd
    finally:
        # On Popen/handshake failure every locally owned pipe is closed. Closing
        # the control writer wakes an already running guard before any SIGSTOP.
        for descriptor in descriptors:
            try:os.close(descriptor)
            except OSError:pass
        if not returned and guard is not None:
            try:guard.wait(timeout=10)
            except subprocess.TimeoutExpired:pass


def resume_guard(args):
    pidfd,control_fd,ready_fd,seconds=args
    os.write(ready_fd,b'R');os.close(ready_fd)
    try:
        select.select([control_fd],[],[],seconds)
        try:signal.pidfd_send_signal(pidfd,signal.SIGCONT,None,0)
        except ProcessLookupError:pass
    finally:
        os.close(pidfd);os.close(control_fd)
    return 0


def pidfd_exited(fd):
    return bool(select.select([fd],[],[],0)[0])


def confirm_stall_event(since):
    for suffix in ('','.1','.2','.3','.4','.5'):
        path=ROOT/'logs'/('state.jsonl'+suffix)
        try:lines=path.read_text().splitlines()
        except OSError:continue
        for line in reversed(lines):
            try:row=json.loads(line)
            except ValueError:continue
            if not isinstance(row,dict) or row.get('event')!='engine_stall_shutdown':continue
            try:stamp=dt.datetime.fromisoformat(row['ts'])
            except (KeyError,ValueError,TypeError):continue
            if stamp.tzinfo is not None and stamp>=since and row.get('deadline_s')==300:return True
    return False


def recovery(key,before,old_instance,timeout):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        try:
            now=service();ready=health(key)
            if ready['instance_id']!=old_instance and now['InvocationID']!=before['InvocationID']:
                if now['NRestarts']<=before['NRestarts']:raise Refused('restart_not_automatic_or_counter_reset')
                reply=json_request(key,'/v1/chat/completions',{'model':'qwen36-35b',
                    'messages':[{'role':'user','content':'Return the number two.'}], 'max_tokens':64,
                    'seed':17,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}},timeout=60)
                if reply.get('model')!='qwen36-35b' or not reply.get('choices'):raise Refused('recovery_generation_invalid')
                if not reply['choices'][0].get('message',{}).get('content'):raise Refused('recovery_generation_empty')
                return {'instance_id':ready['instance_id'],'invocation_id':now['InvocationID'],
                        'nrestarts':now['NRestarts'],'generation_http':200,'usage':reply.get('usage')}
        except (OSError,ValueError,Refused,subprocess.SubprocessError):pass
        time.sleep(2)
    raise Refused('automatic_recovery_deadline')


def cleanup_fault(fd,guard,stream,was_stopped):
    """Independent cleanup steps: one failing action cannot skip the others."""
    result={'errors':[]}
    def attempt(name,fn):
        try:return fn()
        except BaseException as error:
            result['errors'].append({'step':name,'error_type':type(error).__name__})
    if was_stopped and fd is not None:
        def resume():
            try:signal.pidfd_send_signal(fd,signal.SIGCONT,None,0)
            except ProcessLookupError:pass
        attempt('resume_original_pidfd',resume)
    if guard is not None:
        attempt('close_guard_control',lambda:os.close(guard[1]))
        attempt('wait_guard',lambda:guard[0].wait(timeout=10))
    if fd is not None:attempt('close_pidfd',lambda:os.close(fd))
    if stream is not None:
        attempt('close_synthetic_stream',stream.close)
        result['synthetic_stream_closed']=stream.done.is_set()
        if not result['synthetic_stream_closed']:
            result['errors'].append({'step':'close_synthetic_stream','error_type':'ReaderStillRunning'})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--case',choices=('kill','stall'),default='kill')
    parser.add_argument('--worker-pid',type=int,help='Optional disambiguation; must still pass every cgroup/role/starttime check')
    parser.add_argument('--recovery-timeout',type=int,default=480)
    parser.add_argument('--report',type=pathlib.Path)
    parser.add_argument('--resume-guard',nargs=4,type=int,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.resume_guard is not None:return resume_guard(args.resume_guard)
    if not args.live:
        print(json.dumps({'result':'PLAN_ONLY','live':'NOT_RUN','case':args.case,
            'scope':UNIT,'signals':'verified pidfd only','stall_deadline_seconds':300,
            'fail_open_resume_guard_seconds':380,'service_actions':False,'external_e2e':'NOT_RUN'}))
        return 0
    if sys.platform!='linux' or os.geteuid()==0:raise Refused('requires_ordinary_service_user_inside_wsl')
    if not 60<=args.recovery_timeout<=1200:raise Refused('invalid_recovery_timeout')
    stamp=dt.datetime.now(dt.timezone.utc)
    report=args.report or ROOT/'reports'/f'worker-fault-{args.case}-{stamp.strftime("%Y%m%dT%H%M%S")}-{uuid.uuid4().hex[:8]}.json'
    if report.exists():raise Refused('report_already_exists')
    result={'schema':1,'case':args.case,'started_utc':stamp.isoformat(),'result':'FAIL',
            'scope':UNIT,'cold_boot':'NOT_RUN','external_e2e':'NOT_RUN','caddy_generation':'NOT_RUN'}
    fd=None;guard=None;stream=None;was_stopped=False;timer=time.monotonic()
    def interrupted(signum,frame):raise Refused('tester_interrupted')
    old_handlers={sig:signal.signal(sig,interrupted) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        key=load_key();before=service();ready=health(key)
        if before['ActiveState']!='active' or active_count(key)!=0:raise Refused('service_not_idle')
        identity=select_worker(before,args.worker_pid)
        fd=open_verified_pidfd(identity,before)
        result.update(worker=identity,before_instance_id=ready['instance_id'],before_invocation_id=before['InvocationID'],
                      before_nrestarts=before['NRestarts'])
        if args.case=='stall':
            stream=ActiveStream(key);stream.begin()
            if active_count(key)!=1 or stream.done.is_set():raise Refused('owned_stream_no_longer_active')
            guard=arm_resume_guard(fd,380)
        # Revalidate unit identity and /proc starttime immediately before signal.
        now=service()
        if any(now[k]!=before[k] for k in ('ControlGroup','MainPID','InvocationID')):raise Refused('owner_changed_before_signal')
        if proc_stat(identity['pid'])['starttime']!=identity['starttime'] or not member(identity['pid'],before['ControlGroup']):
            raise Refused('target_changed_before_signal')
        sent=time.monotonic();sent_utc=dt.datetime.now(dt.timezone.utc)
        if args.case=='stall':
            if guard[0].poll() is not None:raise Refused('resume_guard_died_before_stop')
            signal.pidfd_send_signal(fd,signal.SIGSTOP,None,0);was_stopped=True
            if proc_stat(identity['pid'])['state'] not in ('T','t'):
                time.sleep(.1)
                if proc_stat(identity['pid'])['state'] not in ('T','t'):raise Refused('worker_not_stopped')
            while not pidfd_exited(fd) and time.monotonic()-sent<370:time.sleep(2)
            waited=time.monotonic()-sent
            result['stopped_until_worker_exit_seconds']=waited
            if not pidfd_exited(fd):raise Refused('stall_detector_did_not_remove_worker_before_resume_deadline')
            if waited<300:raise Refused('worker_exited_before_expected_300_second_stall')
            if not confirm_stall_event(sent_utc):raise Refused('missing_current_300s_stall_shutdown_event')
            result['stall_shutdown_event']=True
        else:
            signal.pidfd_send_signal(fd,signal.SIGKILL,None,0)
            result['signal']='SIGKILL'
        recovered=recovery(key,before,ready['instance_id'],args.recovery_timeout)
        result.update(after=recovered,result='PASS')
    except BaseException as exc:
        result['error_code']=str(exc) if isinstance(exc,Refused) else type(exc).__name__
    finally:
        cleaned=cleanup_fault(fd,guard,stream,was_stopped)
        result['cleanup']=cleaned
        if cleaned['errors']:result.update(result='FAIL',cleanup_error=True)
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
        result['elapsed_seconds']=time.monotonic()-timer
        out=os.open(report,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(out,'w') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result));return 0 if result['result']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
