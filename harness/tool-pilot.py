"""Managed V5 tool-fixture pilot on ONE profile (no matrix).

Holds the GPU-exclusive flock, refuses if any model unit is active, starts only
the named experiment profile, validates its receipt against the current native
source/config, runs `tool-task-bench.py --pilot` N times on the loaded model and
stops only the InvocationID it started. Writes reports/NAME-tool-pilot.json (from
tool-task-bench) and reports/NAME-tool-pilot-runner.json (this runner).
"""
import argparse,hashlib,json,os,pathlib,subprocess,sys,time
import bench
from bench import DEFAULT_LENGTHS,GpuLockBusy,acquire_gpu_lock,current_run_spec,validate_receipt
import matrix

EXIT_GPU_LOCK_BUSY=matrix.EXIT_GPU_LOCK_BUSY
EXIT_REFUSED=matrix.EXIT_REFUSED


def run_pilot(root,profile,repeats,name):
    pilot_path=root/'reports'/f'{name}-tool-pilot.json';runner_path=root/'reports'/f'{name}-tool-pilot-runner.json'
    if pilot_path.exists() or runner_path.exists():raise RuntimeError('Pilot receipts already exist; use a new --name')
    if not matrix.units_inactive():raise RuntimeError('A model service is active')
    arm=current_run_spec(root,[profile],DEFAULT_LENGTHS,128,1)['arms'][profile]
    row={'name':name,'profile':profile,'repeats':repeats,'harness_version':bench.HARNESS_VERSION,
         'runner_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    invocation=None;step='start';interrupted=None
    try:
        started=time.monotonic()
        matrix.run(['bash',str(root/'start-experiment.sh'),profile],60)
        invocation=matrix.state('InvocationID')
        if not invocation:raise RuntimeError('Started experiment has no invocation ID')
        step='readiness'
        matrix.run([matrix.PY,str(root/'check-experiment.py'),'300'],310)
        step='receipt'
        receipt=json.loads((root/'active-run.json').read_text());validate_receipt(receipt,profile,arm)
        row.update(load_ready_s=time.monotonic()-started,run_id=receipt['run_id'])
        step='pilot'
        result=subprocess.run([matrix.PY,str(root/'tool-task-bench.py'),'--pilot','--name',name,'--repeats',str(repeats),
                               '--expect-profile',profile],capture_output=True,text=True,timeout=300*repeats+60)
        row['pilot_exit_code']=result.returncode
        doc=json.loads(pilot_path.read_text())
        row.update(pilot_status=doc.get('status'),passes=doc.get('passes'),failure_code_counts=doc.get('failure_code_counts'),
                   pilot_report_sha256=hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
                   pilot_run_id_matches=doc.get('run',{}).get('run_id')==receipt['run_id'])
        row['status']='PASS' if doc.get('status')=='PASS' and result.returncode==0 and row['pilot_run_id_matches'] else 'FAIL'
    except (RuntimeError,subprocess.TimeoutExpired,ValueError,OSError,KeyError,TypeError) as error:
        row.update(status='FAIL',failed_step=step,error_type=type(error).__name__)
    except (KeyboardInterrupt,SystemExit) as error:
        row.update(status='FAIL',failed_step=step,error_type=type(error).__name__,interrupted=True);interrupted=error
    finally:
        if invocation and matrix.state('InvocationID')==invocation:
            matrix.run(['systemctl','--user','stop','qwen36-experiment'],60)
            row['stopped_own_invocation']=True
    fd=os.open(runner_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as target:target.write(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    if interrupted is not None:raise interrupted
    return 0 if row['status']=='PASS' else 1


def main(argv=None,root=None,lock_path=None):
    p=argparse.ArgumentParser();p.add_argument('--profile',default='bf128')
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--name',required=True)
    a=p.parse_args(argv)
    if not matrix.NAME_PATTERN.fullmatch(a.name) or not matrix.NAME_PATTERN.fullmatch(a.profile) or not 1<=a.repeats<=10:
        p.error('invalid --name/--profile/--repeats')
    try:lock=acquire_gpu_lock(lock_path)
    except GpuLockBusy as error:
        print(json.dumps({'result':'refused','reason':'gpu_lock_busy','message':str(error)}),file=sys.stderr,flush=True)
        return EXIT_GPU_LOCK_BUSY
    try:
        matrix.install_signal_handlers()
        try:return run_pilot(bench.ROOT if root is None else root,a.profile,a.repeats,a.name)
        except (RuntimeError,ValueError,OSError,KeyError) as error:
            print(json.dumps({'result':'refused','error_type':type(error).__name__,'message':str(error)}),file=sys.stderr,flush=True)
            return EXIT_REFUSED
    finally:os.close(lock)


if __name__=='__main__':raise SystemExit(main())
