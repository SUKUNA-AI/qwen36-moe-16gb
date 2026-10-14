"""Opt-in, own synthetic Linux helpers only. Never touches production state/unit/GPU."""
import argparse,datetime,hashlib,importlib.util,json,os,pathlib,select,signal,stat,subprocess,sys,tempfile,time,uuid

SOURCE=pathlib.Path(__file__).resolve().parents[1]/'keepalive.py'
spec=importlib.util.spec_from_file_location('keepalive_inspector',pathlib.Path(__file__).with_name('inspect-keepalive.py'))
inspector=importlib.util.module_from_spec(spec);spec.loader.exec_module(inspector)

def atomic_state(path,state):
    temp=path.with_suffix('.new');temp.write_text(json.dumps(state));temp.chmod(0o600);temp.replace(path)

def starttime(pid):
    raw=pathlib.Path(f'/proc/{pid}/stat').read_text()
    return int(raw[raw.rfind(')')+2:].split()[19])

def case(directory,name):
    generation=uuid.uuid4().hex;lease=uuid.uuid4().hex
    state_path=directory/(name+'-owner.json');records=directory/(name+'-records')
    owner={'Generation':generation,'KeepaliveLeaseId':lease,'SupervisorPid':424242,
        'SupervisorStarted':'synthetic-window-owner','LifetimeJobArmed':True,'HeartbeatSequence':1}
    atomic_state(state_path,owner)
    command=[sys.executable,str(SOURCE),'--state',str(state_path),'--record-root',str(records),
        '--generation',generation,'--lease-id',lease,'--owner-pid','424242','--owner-started','synthetic-window-owner']
    process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,close_fds=True)
    fd=None;record=None;record_path=None
    try:
        fd=os.pidfd_open(process.pid)
        expected_start=starttime(process.pid)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            found=list((records/generation).glob('*.json')) if (records/generation).exists() else []
            if found:
                if len(found)!=1:raise RuntimeError('ambiguous_synthetic_receipt')
                record_path=found[0];record=json.loads(record_path.read_text());break
            if process.poll() is not None:raise RuntimeError('helper_exited_before_own_receipt')
            time.sleep(.05)
        if record is None:raise RuntimeError('own_receipt_timeout')
        if record['linux_pid']!=process.pid or record['linux_starttime_ticks']!=expected_start or record['linux_uid']!=os.getuid():
            raise RuntimeError('own_identity_mismatch')
        if stat.S_IMODE(record_path.stat().st_mode)!=0o600 or stat.S_IMODE(record_path.parent.stat().st_mode)!=0o700:
            raise RuntimeError('private_receipt_permissions')
        if not inspector.verify(generation,lease,root=records)['pass']:raise RuntimeError('inspector_did_not_find_owned_helper')
        time.sleep(1.1)  # Let helper observe the initial lease before mutation.
        if name=='stale':
            owner['HeartbeatSequence']=2;atomic_state(state_path,owner)
            time.sleep(1.2)
            if process.poll() is not None:raise RuntimeError('live_heartbeat_did_not_keep_helper_alive')
            expected='heartbeat_stale';timeout=40
        elif name=='new_owner':
            owner['Generation']=uuid.uuid4().hex;atomic_state(state_path,owner);expected='owner_changed';timeout=8
        elif name=='new_lease':
            owner['KeepaliveLeaseId']=uuid.uuid4().hex;atomic_state(state_path,owner);expected='lease_replaced';timeout=8
        elif name=='removed':
            state_path.unlink();expected='owner_state_removed';timeout=8
        else:raise RuntimeError('unsupported_fixture_case')
        started=time.monotonic()
        code=process.wait(timeout=timeout)
        elapsed=time.monotonic()-started
        finished=json.loads(record_path.read_text())
        if code!=0 or finished['state']!='exited' or finished['reason']!=expected:raise RuntimeError('unexpected_helper_exit')
        poll=select.poll();poll.register(fd,select.POLLIN)
        if not poll.poll(0):raise RuntimeError('exact_helper_identity_not_exited')
        if not inspector.verify(generation,lease,'gone',root=records)['pass']:raise RuntimeError('inspector_did_not_confirm_exit')
        if name=='stale' and (finished['last_sequence']!=2 or elapsed<28):raise RuntimeError('stale_boundary_not_observed')
        return {'case':name,'pass':True,'exit_reason':expected,'elapsed_after_mutation_s':elapsed,
            'pid_starttime_matched':True,'pidfd_confirms_exit':True,'private_record_mode_0600':True,'readonly_inspector_before_after':True,
            'new_owner_state_preserved':not state_path.exists() if name=='removed' else json.loads(state_path.read_text())==owner}
    finally:
        # Only the retained pidfd of this Popen child may be signalled. No scan,
        # numeric-PID kill, pkill, systemctl, WSL terminate, or production paths.
        if process.poll() is None and fd is not None:
            try:signal.pidfd_send_signal(fd,signal.SIGTERM,None,0)
            except ProcessLookupError:pass
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:signal.pidfd_send_signal(fd,signal.SIGKILL,None,0)
                except ProcessLookupError:pass
                process.wait(timeout=5)
        if fd is not None:os.close(fd)
        if process.stdout:process.stdout.close()
        if process.stderr:process.stderr.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--own-helpers',action='store_true');parser.add_argument('--report',type=pathlib.Path)
    parser.add_argument('--case',choices=('stale','new_owner','new_lease','removed'))
    args=parser.parse_args()
    if not args.own_helpers:
        print(json.dumps({'result':'PLAN_ONLY','model_gpu':'NOT_RUN','production_state':'NOT_TOUCHED'}));return 0
    if sys.platform!='linux' or os.getuid()==0 or not hasattr(os,'pidfd_open'):raise SystemExit('Requires ordinary Linux owner and pidfd')
    report={'schema':1,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'REAL_OWN_LINUX_HELPERS_WITH_SYNTHETIC_OWNER_STATE','model_gpu':'NOT_RUN',
        'production_state':'NOT_TOUCHED','systemd':'NOT_USED','windows_job_integration':'NOT_RUN',
        'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'cases':[],'pass':False}
    try:
        with tempfile.TemporaryDirectory(prefix='qwen36-own-keepalive-') as temp:
            directory=pathlib.Path(temp);directory.chmod(0o700)
            for name in ([args.case] if args.case else ('stale','new_owner','new_lease','removed')):report['cases'].append(case(directory,name))
        report['pass']=all(row['pass'] for row in report['cases'])
    except Exception as error:report['error_type']=type(error).__name__
    if args.report:
        fd=os.open(args.report,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as handle:json.dump(report,handle,indent=2);handle.write('\n')
    print(json.dumps(report));return 0 if report['pass'] else 1

if __name__=='__main__':raise SystemExit(main())
