"""Serial restart A/B runner (paired harness V6).

Immutable V6 manifest, GPU-exclusive flock held for the whole matrix, scoped unit
ownership (stop only the InvocationID this process started), a mandatory passing
tool-fixture pilot, and no resume of V5/legacy manifests or of a matrix whose
workload already failed.

V6: the manifest also freezes the launcher files (start-experiment.sh,
run-experiment.sh, check-experiment.py) and the VRAM floor; before every arm the full
identity (runtime, configs, harness, launchers, pilot) is re-hashed and the workspace
copy of the profile config must equal the native one (start-experiment.sh refreshes
exactly that file). After readiness the serving process's launcher identity (file
hashes + filtered start environment from /proc/<MainPID>/environ) must match the
manifest, the static run-experiment.sh contract and the first arm's environment
(NAME-launcher.json, written once with O_EXCL). Every arm row records the foreign VRAM
use before start (nvidia-smi, no CUDA context), the service-sampled peak used and minimum
free VRAM; a process footprint above total - 1 GiB (bench.vram_budget, the owner's Windows
reserve) makes the arm INVALID (code process_vram_over_budget) and the matrix exits 3 after
the remaining arms. Free VRAM is reported, not gated.
Exit codes: 0 all arms complete and valid; 1 an arm failed (stops the matrix);
2 refused before any arm; 3 complete but some arm INVALID (VRAM budget); 75 lock busy.
--pairs N (N >= 3, default 5): N != 5 is a screen. Its manifest carries screen:true, and on completion
NAME-screen.json lists every pair and the medians per profile/case/metric; compare-ab still takes only 5 pairs.
"""
import argparse,hashlib,json,os,pathlib,re,signal,subprocess,sys,time
import bench
from bench import (DEFAULT_LENGTHS,HARNESS_VERSION,BENCH_FIXTURE_VERSION,WINDOWS_RESERVE_BYTES,GpuLockBusy,acquire_gpu_lock,
                   current_run_spec,parse_lengths,validate_receipt)
import tool_fixture_v5 as tool_fixture

PY=os.environ.get('QWEN36_GPU_PYTHON','/home/USER/qwen36-api/.venv/bin/python')
NAME_PATTERN=re.compile(r'[a-z0-9][a-z0-9.-]{0,60}')
PAIRS=5
MIN_PAIRS=3
SCREEN_METRICS=('elapsed_s','ttft_content_s','decode_client_tps')
MIN_PILOT_REPEATS=3
# Failures before any request was served may be retried under the same manifest.
RESUMABLE_FAILED_STEPS=('preflight','start','readiness')
EXIT_GPU_LOCK_BUSY=75
EXIT_REFUSED=2
EXIT_ARM_INVALID=3


class LauncherIdentityError(RuntimeError):
    """The serving process was not started by the frozen launcher/environment."""


def run(args,timeout=1200):
    result=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if args[:3]==['systemctl','--user','stop'] and result.returncode==5:return ''
    if result.returncode:raise RuntimeError(f'{pathlib.Path(args[0]).name}: exit {result.returncode}')
    return result.stdout


def make_schedule(profiles,pairs):
    result=[]
    for repeat in range(pairs):
        shift=(repeat//2)%len(profiles) if len(profiles)>2 else 0
        order=profiles[shift:]+profiles[:shift]
        if repeat%2:order=order[::-1]
        for position,profile in enumerate(order):
            result.append({'repeat':repeat,'profile':profile,'pair_position':position,'schedule_position':len(result)})
    return result


def freeze_manifest(path,spec,has_results=False):
    if path.exists():
        existing=json.loads(path.read_text())
        if existing.get('harness_version')!=spec.get('harness_version'):
            raise RuntimeError('Existing manifest belongs to another harness version (V5 or older); '
                               'V6 never resumes it - use a new --name')
        if existing!=spec:raise RuntimeError('Existing matrix manifest differs; use a new --name')
        return
    if has_results:raise RuntimeError('Old results have no frozen manifest; use a new --name')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as target:json.dump(spec,target,indent=2)


def state(prop,unit='qwen36-experiment'):
    """systemctl --user show; fail-closed (V6): a failed query raises instead of reading as inactive."""
    result=subprocess.run(['systemctl','--user','show',unit,'--property='+prop,'--value'],
                          capture_output=True,text=True,timeout=10)
    if result.returncode:raise RuntimeError(f'systemctl show {unit} {prop}: exit {result.returncode}')
    return result.stdout.strip()


def units_inactive():
    return all(state('ActiveState',unit) in ('','inactive','failed') for unit in ('qwen36-inference','qwen36-experiment'))


def cgroup_peak(invocation):
    """Read only this unit's current invocation; never reset or probe another cgroup."""
    if not invocation or state('InvocationID')!=invocation:return None
    control=state('ControlGroup');base=pathlib.Path('/sys/fs/cgroup').resolve()
    if not control.startswith('/') or pathlib.PurePosixPath(control).name!='qwen36-experiment.service':return None
    path=(base/control.lstrip('/')).resolve()
    if not path.is_relative_to(base):return None
    try:value=(path/'memory.peak').read_text().strip()
    except OSError:return None
    return int(value) if value.isdigit() and state('InvocationID')==invocation else None


def validate_tool_pilot(root,pilot_name,profiles,arms,harness):
    """The V5 tool fixture must have passed on this exact runtime/harness before the matrix loads."""
    if not isinstance(pilot_name,str) or not NAME_PATTERN.fullmatch(pilot_name):
        raise RuntimeError('tool_pilot_invalid_name')
    path=root/'reports'/f'{pilot_name}-tool-pilot.json'
    try:raw=path.read_bytes();doc=json.loads(raw)
    except FileNotFoundError:raise RuntimeError('tool_pilot_missing: run tool-pilot.py first') from None
    except ValueError:raise RuntimeError('tool_pilot_unreadable') from None
    rows=doc.get('rows') if isinstance(doc.get('rows'),list) else []
    profile=doc.get('profile')
    if doc.get('mode')!='pilot' or doc.get('schema_version')!=2 or doc.get('name')!=pilot_name:
        raise RuntimeError('tool_pilot_wrong_schema')
    if doc.get('status')!='PASS' or type(doc.get('repeats')) is not int or doc['repeats']<MIN_PILOT_REPEATS \
            or len(rows)!=doc['repeats'] or not all(row.get('correct') is True and row.get('failure_codes')==[] for row in rows):
        raise RuntimeError('tool_pilot_not_passing')
    if doc.get('fixture')!=tool_fixture.describe():raise RuntimeError('tool_pilot_fixture_mismatch')
    sources=doc.get('sources',{})
    if doc.get('source_sha256')!=harness.get('tool-task-bench.py') or sources.get('tool-task-bench.py')!=harness.get('tool-task-bench.py') \
            or sources.get('tool_fixture_v5.py')!=harness.get('tool_fixture_v5.py') or sources.get('bench.py')!=harness.get('bench.py'):
        raise RuntimeError('tool_pilot_harness_mismatch')
    if profile not in profiles:raise RuntimeError('tool_pilot_profile_not_in_matrix')
    try:validate_receipt(doc.get('run'),profile,arms[profile])
    except (ValueError,KeyError,TypeError):raise RuntimeError('tool_pilot_runtime_mismatch') from None
    return {'name':pilot_name,'report_sha256':hashlib.sha256(raw).hexdigest(),'profile':profile,
            'repeats':doc['repeats'],'run_id':doc['run']['run_id']}


def build_spec(root,profiles,lengths,output,pairs,pilot_name):
    spec=current_run_spec(root,profiles,lengths,output,pairs)
    spec['schedule']=make_schedule(profiles,pairs)
    spec['harness_version']=HARNESS_VERSION
    spec['bench_fixture_version']=BENCH_FIXTURE_VERSION
    spec['tool_fixture']=tool_fixture.describe()
    spec['tool_pilot']=validate_tool_pilot(root,pilot_name,profiles,spec['arms'],spec['harness_source_files'])
    spec['windows_reserve_bytes']=WINDOWS_RESERVE_BYTES
    if pairs!=PAIRS:spec['screen']=True   # absent for 5 pairs: release manifests stay unchanged
    return spec


def screen_summary(root,name,rows):
    """Every pair and the median per profile/case/metric of a screen matrix (not a gate)."""
    import statistics
    values={}
    for arm in rows:
        if arm.get('status')!='complete':continue
        doc=json.loads((root/'reports'/arm['report']).read_text())
        for row in doc.get('rows',[]):
            if row.get('status')!=200 or row.get('case') in (None,'warmup','warmup_sustained'):continue
            for metric in SCREEN_METRICS:
                if type(row.get(metric)) in (int,float):
                    key=(arm['profile'],f"{row['case']}_{row.get('length')}",metric)
                    values.setdefault(key,{})[arm['repeat']]=row[metric]
    out={}
    for (profile,case,metric),by_repeat in sorted(values.items()):
        pairs=[by_repeat[r] for r in sorted(by_repeat)]
        out.setdefault(profile,{}).setdefault(case,{})[metric]={'pairs':pairs,'repeats':sorted(by_repeat),
                                                                'median':statistics.median(pairs)}
    return {'kind':'matrix-screen','screen':True,'gate':False,'name':name,'profiles':out,
            'note':'medians and every pair as measured; the release decision is compare-ab over five pairs'}


def workspace_config_drift(root,profile,workspace=None):
    """start-experiment.sh refreshes config/PROFILE.json from the workspace copy; they must already be equal."""
    source=pathlib.Path(bench.WORKSPACE_OPT if workspace is None else workspace)/f'{profile}.json'
    if not source.exists():return False
    try:return source.read_bytes()!=(root/'config'/f'{profile}.json').read_bytes()
    except OSError:return True


def check_launcher(root,name,profile,spec,launcher):
    """Launcher identity of the serving process vs manifest, static contract and the first arm (O_EXCL file)."""
    if launcher.get('files')!=spec['launcher_files']:raise LauncherIdentityError('launcher_files_differ_from_manifest')
    if launcher.get('problems'):raise LauncherIdentityError('launcher_environment_unexpected:'+','.join(launcher['problems']))
    path=root/'reports'/f'{name}-launcher.json'
    record={'name':name,'harness_version':HARNESS_VERSION,'first_profile':profile,'files':launcher['files'],
            'environment':launcher['environment'],'environment_sha256':launcher['environment_sha256']}
    try:
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as target:json.dump(record,target,indent=2)
        return
    except FileExistsError:pass
    first=json.loads(path.read_text())
    if first.get('environment_sha256')!=launcher['environment_sha256'] or first.get('files')!=launcher['files']:
        raise LauncherIdentityError('launcher_environment_changed_between_arms')


def arm_vram(root,name,plan,pre_start):
    """Owner budget (bench.vram_budget) over every service-sampled request of the arm, incl. tool steps."""
    bench_path,task_path=arm_files(root,name,plan);rows=[]
    for path,key in ((bench_path,'rows'),(task_path,'row')):
        try:doc=json.loads(path.read_text())
        except (OSError,ValueError):continue
        value=doc.get(key)
        rows+=value if isinstance(value,list) else [value]
    return {**bench.vram_budget(rows,pre_start),
            'pre_start_gpu':pre_start,
            'foreign_gpu_used_bytes_pre_start':pre_start.get('used_bytes') if isinstance(pre_start,dict) else None,
            'scope':'service samples during requests (incl. warmup, primers, tool steps); pre-start = nvidia-smi with no model unit active'}


def arm_files(root,name,plan):
    stem=f'{name}-{plan["repeat"]}-{plan["profile"]}'
    return root/'reports'/f'{stem}.json',root/'reports'/f'{stem}-tool-task.json'


def check_start_or_resume(root,name,spec,rows,lengths,output):
    """Return the completed (repeat, profile) set; refuse unsafe resumes."""
    completed=set()
    for row in rows:
        if row.get('status')=='failed' and row.get('failed_step') not in RESUMABLE_FAILED_STEPS:
            raise RuntimeError('Matrix has an arm that failed after the workload started; it cannot be resumed. '
                               'Keep its receipts and use a new --name')
    done={(r.get('repeat'),r.get('profile')) for r in rows if r.get('status')=='complete'}
    for plan in spec['schedule']:
        repeat,profile=plan['repeat'],plan['profile'];bench_path,task_path=arm_files(root,name,plan)
        if (repeat,profile) not in done:
            if bench_path.exists() or task_path.exists():
                raise RuntimeError('Unaccounted arm receipt exists under this --name; refusing to overwrite it')
            continue
        old=json.loads(bench_path.read_text())
        validate_receipt(old['run'],profile,spec['arms'][profile])
        benchmark=old.get('benchmark',{})
        if benchmark.get('output')!=output or benchmark.get('lengths')!=list(lengths) \
                or benchmark.get('fixture_version')!=BENCH_FIXTURE_VERSION or benchmark.get('harness_version')!=HARNESS_VERSION \
                or benchmark.get('source_sha256')!=spec['harness_source_files']['bench.py']:
            raise RuntimeError('Completed benchmark differs from the V6 manifest')
        task=json.loads(task_path.read_text())
        validate_receipt(task['run'],profile,spec['arms'][profile])
        if task.get('status')!='complete' or task.get('mode')!='matrix' or task.get('repeat')!=repeat \
                or task['run']['run_id']!=old['run']['run_id'] or task.get('fixture')!=spec['tool_fixture'] \
                or task.get('source_sha256')!=spec['harness_source_files']['tool-task-bench.py']:
            raise RuntimeError('Completed tool task differs from matrix arm')
        completed.add((repeat,profile))
    return completed


def terminate_cleanly(signum,_frame):
    """SIGTERM/SIGHUP become SystemExit so `finally` stops only our own unit."""
    raise SystemExit(128+signum)


def install_signal_handlers():
    for name in ('SIGTERM','SIGHUP'):signal.signal(getattr(signal,name),terminate_cleanly)


def tool_failure_codes(path):
    try:doc=json.loads(path.read_text())
    except (OSError,ValueError):return None
    codes=doc.get('failure_codes')
    return [code for code in codes if code in tool_fixture.FAILURE_CODES] if isinstance(codes,list) else None


def run_matrix(a,profiles,lengths,root):
    if not units_inactive():raise RuntimeError('A model service is active')
    spec=build_spec(root,profiles,lengths,a.output,a.pairs,a.tool_pilot)
    drifted=[profile for profile in profiles if workspace_config_drift(root,profile)]
    if drifted:raise RuntimeError('Workspace profile config differs from native config/ (run sync-v5 first): '+','.join(drifted))
    report=root/'reports'/f'{a.name}-matrix.json';manifest=root/'reports'/f'{a.name}-manifest.json'
    rows=json.loads(report.read_text()) if report.exists() else []
    if not manifest.exists() and not rows:
        # A fresh name must not collide with any earlier receipt.
        for plan in spec['schedule']:
            if any(path.exists() for path in arm_files(root,a.name,plan)):
                raise RuntimeError('Receipts already exist under this --name without a manifest; use a new --name')
    freeze_manifest(manifest,spec,has_results=bool(rows))
    completed=check_start_or_resume(root,a.name,spec,rows,lengths,a.output)
    invalid_arms=[r for r in rows if r.get('status')=='complete' and r.get('arm_verdict')=='INVALID']
    for plan in spec['schedule']:
        repeat,profile=plan['repeat'],plan['profile']
        if (repeat,profile) in completed:continue
        name=f'{a.name}-{repeat}-{profile}'
        row=dict(plan,report=name+'.json',harness_version=HARNESS_VERSION);invocation=None;step='preflight';interrupted=None
        print(json.dumps({'starting':row}),flush=True)
        try:
            current=build_spec(root,profiles,lengths,a.output,a.pairs,a.tool_pilot)
            if current!=spec:
                row['drift_keys']=sorted(key for key in set(spec)|set(current) if spec.get(key)!=current.get(key))
                raise RuntimeError('Runtime, profiles, harness, launchers or pilot changed during matrix')
            if workspace_config_drift(root,profile):raise RuntimeError('Workspace profile config differs from native config/')
            if not units_inactive():raise RuntimeError('Another model service became active')
            pre_start=bench.gpu_memory_snapshot()
            started=time.monotonic();step='start'
            run(['bash',str(root/'start-experiment.sh'),profile],60)
            invocation=state('InvocationID')
            if not invocation:raise RuntimeError('Started experiment has no invocation ID')
            step='readiness'
            run([PY,str(root/'check-experiment.py'),'300'],310)
            step='receipt'
            receipt=json.loads((root/'active-run.json').read_text());validate_receipt(receipt,profile,spec['arms'][profile])
            row.update(load_ready_s=time.monotonic()-started,run_id=receipt['run_id'])
            step='launcher'
            launcher=bench.launcher_receipt(profile,root)
            row['launcher']={key:launcher[key] for key in ('files','environment_sha256','problems','main_pid','invocation_id')}
            if launcher['invocation_id']!=invocation:raise LauncherIdentityError('launcher_invocation_mismatch')
            check_launcher(root,a.name,profile,spec,launcher)
            row['startup_cgroup_memory_peak_bytes']=cgroup_peak(invocation)
            row['startup_peak_scope']='unit lifetime through loaded-model readiness, before benchmark warmup; null means unavailable'
            step='bench'
            run([PY,str(root/'bench.py'),'--name',name,'--lengths',','.join(map(str,lengths)),
                 '--repeats','1','--repeat-start',str(repeat),'--output',str(a.output)])
            step='tool_task'
            run([PY,str(root/'tool-task-bench.py'),'--name',name,'--repeat',str(repeat)],300)
            row['tool_task_report']=name+'-tool-task.json'
            row['whole_run_cgroup_memory_peak_bytes']=cgroup_peak(invocation)
            row['vram']=arm_vram(root,a.name,plan,pre_start)
            row['arm_verdict']='INVALID' if row['vram']['gate']=='INVALID' else 'VALID'
            if row['arm_verdict']=='INVALID':row['invalid_codes']=[row['vram']['code']]
            row['status']='complete'
        except (RuntimeError,subprocess.TimeoutExpired,ValueError,OSError,KeyError,TypeError) as error:
            row.update(status='failed',failed_step=step,error_type=type(error).__name__)
            if isinstance(error,LauncherIdentityError):row['launcher_error']=str(error)[:200]
            if step=='tool_task':row['tool_failure_codes']=tool_failure_codes(arm_files(root,a.name,plan)[1])
        except (KeyboardInterrupt,SystemExit) as error:
            # Recorded with its real step: resumable only if no request was served yet.
            row.update(status='failed',failed_step=step,error_type=type(error).__name__,interrupted=True);interrupted=error
        finally:
            try:
                if invocation is None and step=='start':
                    # start-experiment.sh may have started the unit and failed afterwards: under the
                    # GPU lock with both units verified inactive just before, that unit is ours.
                    orphan=state('InvocationID')
                    if orphan:invocation=orphan;row['stopped_after_start_failure']=True
                if invocation and state('InvocationID')==invocation:
                    run(['systemctl','--user','stop','qwen36-experiment'],60)
            except (RuntimeError,subprocess.TimeoutExpired,OSError) as error:
                row.update(stop_error=type(error).__name__)
                if row.get('status')=='complete':row.update(status='failed',failed_step='stop')
            rows.append(row);report.write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
        if interrupted is not None:raise interrupted
        if row['status']!='complete':return 1
        if row.get('arm_verdict')=='INVALID':invalid_arms.append(row)
    if spec.get('screen'):
        fd=os.open(root/'reports'/f'{a.name}-screen.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as target:json.dump(screen_summary(root,a.name,rows),target,indent=2)
    print(json.dumps({'result':'complete','name':a.name,'arms':len(spec['schedule']),'screen':bool(spec.get('screen')),
                      'invalid_arms':[{'repeat':r['repeat'],'profile':r['profile'],'codes':r.get('invalid_codes')} for r in invalid_arms]}),flush=True)
    return EXIT_ARM_INVALID if invalid_arms else 0


def main(argv=None,root=None,lock_path=None):
    p=argparse.ArgumentParser();p.add_argument('--profiles',required=True)
    p.add_argument('--pairs',type=int,default=PAIRS);p.add_argument('--name',required=True)
    p.add_argument('--lengths',default=','.join(map(str,DEFAULT_LENGTHS)));p.add_argument('--output',type=int,default=128)
    p.add_argument('--tool-pilot',required=True,help='NAME of a PASS reports/NAME-tool-pilot.json from tool-pilot.py')
    a=p.parse_args(argv);profiles=a.profiles.split(',');lengths=parse_lengths(a.lengths)
    if len(profiles)<2 or len(set(profiles))!=len(profiles) or a.pairs<MIN_PAIRS or not 1<=a.output<=8192:
        raise ValueError('Invalid matrix dimensions: >=2 distinct profiles, >=3 pairs (5 for compare-ab)')
    if not NAME_PATTERN.fullmatch(a.name):raise ValueError('Invalid --name')
    try:lock=acquire_gpu_lock(lock_path)
    except GpuLockBusy as error:
        print(json.dumps({'result':'refused','reason':'gpu_lock_busy','message':str(error)}),file=sys.stderr,flush=True)
        return EXIT_GPU_LOCK_BUSY
    try:
        install_signal_handlers()
        print(json.dumps({'gpu_lock':'acquired','pid':os.getpid()}),flush=True)
        try:return run_matrix(a,profiles,lengths,bench.ROOT if root is None else root)
        except (RuntimeError,ValueError,OSError,KeyError,TypeError) as error:
            # Pre-arm refusals (manifest/pilot/resume/stale receipts): nothing was started.
            print(json.dumps({'result':'refused','error_type':type(error).__name__,'message':str(error)}),file=sys.stderr,flush=True)
            return EXIT_REFUSED
    finally:os.close(lock)


if __name__=='__main__':raise SystemExit(main())
