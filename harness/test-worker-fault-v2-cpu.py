"""CPU-only mock checks for fault selection; never open a real pidfd or send signals."""
import ast
import importlib.util
import json
import pathlib
import types
from unittest.mock import Mock,patch

path=pathlib.Path(__file__).with_name('test-worker-fault-v2.py')
spec=importlib.util.spec_from_file_location('fault_test_module',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]
def check(name,condition):
    assert condition,name
    checks.append(name)

raw='27 (worker ) label) '+ ' '.join(['S','5']+['0']*17+['98765']+['0']*4)
with patch.object(pathlib.Path,'read_text',return_value=raw):
    check('proc_stat_starttime_and_ppid',m.proc_stat(27)==dict(state='S',ppid=5,starttime=98765))
snapshot=dict(ControlGroup='/user.slice/user-1000.slice/user@1000.service/app.slice/qwen36-inference.service',MainPID=4,InvocationID='new')
identity=dict(pid=27,starttime=100)
with patch.object(m.os,'pidfd_open',return_value=77,create=True), patch.object(m.signal,'pidfd_send_signal',create=True) as sent, patch.object(m.os,'close') as closed, patch.object(m,'proc_stat',return_value=dict(starttime=100)), patch.object(m,'member',return_value=True), patch.object(m,'service',return_value=snapshot):
    check('verified_target_returns_pidfd',m.open_verified_pidfd(identity,snapshot)==77)
    check('open_only_checks_signal_zero',sent.call_args.args[1]==0)
    check('verified_handle_retained',closed.call_count==0)
with patch.object(m.os,'pidfd_open',return_value=77,create=True), patch.object(m.signal,'pidfd_send_signal',create=True) as sent, patch.object(m.os,'close') as closed, patch.object(m,'proc_stat',side_effect=[dict(starttime=100),dict(starttime=101)]), patch.object(m,'member',return_value=True):
    try:m.open_verified_pidfd(identity,snapshot)
    except m.Refused:pass
    else:raise AssertionError('PID reuse accepted')
    check('pid_reuse_closes_handle_without_signal',closed.call_args.args==(77,) and sent.call_count==0)
with patch.object(m.os,'pidfd_open',return_value=77,create=True), patch.object(m.signal,'pidfd_send_signal',create=True) as sent, patch.object(m.os,'close') as closed, patch.object(m,'proc_stat',return_value=dict(starttime=100)), patch.object(m,'member',return_value=True), patch.object(m,'service',return_value={**snapshot,'InvocationID':'different'}):
    try:m.open_verified_pidfd(identity,snapshot)
    except m.Refused:pass
    else:raise AssertionError('Changed invocation accepted')
    check('new_service_generation_closes_without_signal',closed.call_count==1 and sent.call_count==0)
with patch.object(m,'json_request',return_value={'requests':{'active':1}}):
    check('pinned_nested_stats_schema',m.active_count('synthetic')==1)
tree=ast.parse(path.read_text())
check('no_numeric_os_kill',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
    isinstance(n.func.value,ast.Name) and n.func.value.id=='os' and n.func.attr=='kill' for n in ast.walk(tree)))

# Even an error resuming or waiting for the guard must not skip fd and stream
# cleanup. Every signal/process operation below is replaced by a mock.
stream=types.SimpleNamespace(close=Mock(),done=types.SimpleNamespace(is_set=lambda:True))
guard=types.SimpleNamespace(wait=Mock(side_effect=m.subprocess.TimeoutExpired('synthetic',10)))
with patch.object(m.signal,'SIGCONT',18,create=True),patch.object(m.signal,'pidfd_send_signal',side_effect=PermissionError('synthetic'),create=True),patch.object(m.os,'close') as close:
    cleaned=m.cleanup_fault(77,(guard,88),stream,True)
    check('failed_resume_and_guard_wait_do_not_skip_descriptor_close',close.call_args_list[0].args==(88,) and close.call_args_list[1].args==(77,))
    check('failed_cleanup_still_closes_stream',stream.close.call_count==1 and cleaned['synthetic_stream_closed'])
    check('cleanup_errors_have_sanitized_steps_only',cleaned['errors']==[
        {'step':'resume_original_pidfd','error_type':'PermissionError'},
        {'step':'wait_guard','error_type':'TimeoutExpired'}])
with patch.object(m.signal,'SIGCONT',18,create=True),patch.object(m.signal,'pidfd_send_signal',side_effect=ProcessLookupError,create=True),patch.object(m.os,'close'):
    check('dead_pidfd_resume_is_expected',not m.cleanup_fault(77,None,None,True)['errors'])
with patch.object(m.os,'pipe',side_effect=[(1,2),(3,4)]),patch.object(m.subprocess,'Popen',side_effect=OSError('synthetic')),patch.object(m.os,'close') as close:
    try:m.arm_resume_guard(77,380)
    except OSError:pass
    else:raise AssertionError('Spawn failure unexpectedly accepted')
    check('guard_spawn_failure_closes_all_local_pipes',{c.args[0] for c in close.call_args_list}=={1,2,3,4})
guard=types.SimpleNamespace(wait=Mock())
with patch.object(m.os,'pipe',side_effect=[(1,2),(3,4)]),patch.object(m.subprocess,'Popen',return_value=guard),patch.object(m.select,'select',return_value=([],[],[])),patch.object(m.os,'close') as close:
    try:m.arm_resume_guard(77,380)
    except m.Refused:pass
    else:raise AssertionError('Unarmed guard accepted')
    check('guard_timeout_closes_all_local_pipes',{c.args[0] for c in close.call_args_list}=={1,2,3,4})
    check('guard_timeout_waits_for_owned_child',guard.wait.call_count==1)
with patch.object(m.os,'pipe',side_effect=[(1,2),(3,4)]),patch.object(m.subprocess,'Popen',return_value=guard),patch.object(m.select,'select',return_value=([3],[],[])),patch.object(m.os,'read',return_value=b'R'),patch.object(m.os,'close') as close:
    check('armed_guard_retains_only_parent_control_writer',m.arm_resume_guard(77,380)==(guard,2) and {c.args[0] for c in close.call_args_list}=={1,3,4})
# F-04: the Bearer request ignores inherited proxies and never follows redirects.
import email.message,io,os,socket,urllib.error,urllib.request,urllib.response
class MemoryTransport(urllib.request.BaseHandler):
    handler_order=150
    def __init__(self,code,location=None):self.code=code;self.location=location;self.calls=[]
    def http_open(self,request):
        self.calls.append((request.host,request.has_proxy(),request.get_header('Authorization')))
        headers=email.message.Message()
        if self.location:headers['Location']=self.location
        response=urllib.response.addinfourl(io.BytesIO(b'{"status":"ok"}'),headers,request.full_url,self.code);response.msg='synthetic'
        return response
    https_open=http_open
proxies={k:'http://third-party.invalid:8888' for k in ('HTTP_PROXY','http_proxy','HTTPS_PROXY','https_proxy','ALL_PROXY','all_proxy')}
proxies.update(NO_PROXY='',no_proxy='')
build=urllib.request.build_opener
with patch.dict(os.environ,proxies),patch.object(socket,'create_connection',side_effect=AssertionError('no real sockets')):
    for code,location in ((200,None),(302,'https://third-party.invalid/collect')):
        transport=MemoryTransport(code,location);passed=[]
        # Fresh module instance: its own OPENER is built from exactly the handlers
        # the module passes, plus the in-memory transport (no socket, no proxy).
        def mocked(*handlers):passed.extend(handlers);return build(*handlers,transport)
        with patch.object(urllib.request,'build_opener',side_effect=mocked):
            fresh=importlib.util.module_from_spec(spec);spec.loader.exec_module(fresh)
        try:status=fresh.request('SYNTHETIC-NOT-A-CREDENTIAL','/health').status
        except urllib.error.HTTPError as error:status=error.code
        check('bearer_request_direct_no_proxy_'+str(code),transport.calls==[('127.0.0.1:1919',False,'Bearer SYNTHETIC-NOT-A-CREDENTIAL')] and status==code)
check('module_opener_has_empty_proxy_map_and_no_redirect',
    any(isinstance(h,urllib.request.ProxyHandler) and h.proxies=={} for h in passed) and
    any(isinstance(h,fresh.NoRedirect) for h in passed) and
    not any(isinstance(n,ast.Attribute) and n.attr=='urlopen' for n in ast.walk(tree)))
print(json.dumps(dict(result='PASS',count=len(checks),checks=checks,live='NOT_RUN')))
