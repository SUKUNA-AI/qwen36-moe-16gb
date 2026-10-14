"""Synthetic, owner-local benchmark (paired harness V6).

No keys, prompt bodies or generations in reports. V5 semantics (unchanged in V6): the
fixed synthetic-history case is `history_continuation`; its actual cached-token count
is measured and labelled (`cache_hit_observed`), and zero cached tokens is a
legitimate observed miss, never reported as a hit.

V6 additions: launcher identity (sha256 of start-experiment.sh, run-experiment.sh,
check-experiment.py + the filtered start environment of the serving process, read from
/proc/<MainPID>/environ; secret-like names are redacted) in manifests (`launcher_files`)
and in every bench report (`launcher`); the service-sampled minimum GPU free memory per
arm against the owner's hard floor (0.5 GiB).
"""
import argparse, hashlib, json, math, os, pathlib, random, statistics, time, re
import requests
from transformers import AutoTokenizer

ROOT=pathlib.Path('/home/USER/qwen36-api/optimization')
DEFAULT_LENGTHS=(64,4096,16384,65536,120000)
SERVICE_FILES=('experiment.py','serve_guarded.py','guard.py','ingress.py','request_metrics.py')
PACKAGES=('torch','triton','flashinfer-python','sglang-kernel','transformers')
CHECKPOINT='nvidia/Qwen3.6-35B-A3B-NVFP4'
CHECKPOINT_REVISION='1355db6a052410cfd62085d94b58866fd0f2c3c5'
SOURCE_SUFFIXES=('.py','.cu','.h','.cuh','.cpp')
HARNESS_VERSION=6
# Benchmark prompts/fixtures are unchanged since V5.
BENCH_FIXTURE_VERSION=5
# Every file whose behaviour defines a V6 matrix/pilot result; hashed into manifests.
HARNESS_FILES=('bench.py','matrix.py','compare-ab.py','tool-task-bench.py','tool_fixture_v5.py','tool-pilot.py')
GPU_LOCK=pathlib.Path('/home/USER/qwen36-api/run/gpu-exclusive.lock')
# Launchers that define every arm's process environment (V6: hashed into manifest + reports).
LAUNCHER_FILES=('start-experiment.sh','run-experiment.sh','check-experiment.py')
EXPERIMENT_UNIT='qwen36-experiment'
# Workspace copy that start-experiment.sh copies the started profile's config from.
WORKSPACE_OPT=pathlib.Path('/path/to/qwen36-workspace/outputs/qwen36-api/optimization')
# Launcher environment recorded per arm: these names plus these prefixes; values of
# secret-like names are replaced by '<redacted>'.
LAUNCHER_ENV_KEYS=('PATH','CC','CXX','CUDAHOSTCXX','NVCC_CCBIN','MAX_JOBS','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
                   'LOG_LEVEL','VIRTUAL_ENV','HOME','LANG','LC_ALL')
LAUNCHER_ENV_PREFIXES=('QWEN36_','FREETOKEN_','SGLANG_','FLA_','TORCH','TRITON_','PYTORCH_','CUDA_','NCCL_','NVIDIA_',
                       'OMP_','MKL_','LD_','LIBRARY_PATH','PYTHON')
LAUNCHER_ENV_SECRET=re.compile('KEY|TOKEN|SECRET|PASSW|CREDENTIAL|AUTH|COOKIE',re.I)
# Owner decision 2026-10-13: VRAM hard floor for every profile; foreign usage is recorded, not refused.
GIB=1<<30
MIB=1<<20
VRAM_FLOOR_BYTES=GIB//2
NVIDIA_SMI='/usr/lib/wsl/lib/nvidia-smi'


class GpuLockBusy(RuntimeError):
    """Another process holds the GPU-exclusive lock; nothing was started."""


def acquire_gpu_lock(path=None):
    """Non-blocking exclusive flock. Returns the open fd; the lock lives until it is closed.

    The fd is non-inheritable (PEP 446), so model units and benchmark children do not
    keep the lock after this process exits. The lock file content is never modified.
    """
    import fcntl
    path=pathlib.Path(GPU_LOCK if path is None else path)
    path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    fd=os.open(path,os.O_RDWR|os.O_CREAT,0o600)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise GpuLockBusy(f'GPU-exclusive lock {path} is held by another process; refusing to start any GPU work') from None
    except OSError:
        os.close(fd);raise
    return fd


def cache_requirement(case):
    """Fresh inputs must not hit; every history/edit task records an observed value only."""
    return 'must_be_zero' if case=='fresh' else 'observed_not_required'


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def valid_sha(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def parse_lengths(value):
    lengths=tuple(int(item) for item in value.split(',')) if isinstance(value,str) else tuple(value)
    if not lengths or len(set(lengths))!=len(lengths) or any(type(x) is not int or x<=0 for x in lengths):
        raise ValueError('lengths must be distinct positive integers')
    return lengths


def validate_speed_config(config):
    if not isinstance(config,dict) or not {'context','chunk','flags','env'}<=set(config):
        raise ValueError('Incomplete speed profile')
    if type(config['context']) is not int or type(config['chunk']) is not int or config['context']<1 or config['chunk']<1:
        raise ValueError('Invalid context/chunk')
    if not isinstance(config['flags'],list) or not all(isinstance(v,str) for v in config['flags']) or not isinstance(config['env'],dict):
        raise ValueError('Invalid flags/env')
    forbidden=('QWEN36_TEACHER_FIXTURE','QWEN36_LOGIT_REFERENCE','QWEN36_NSYS_FORWARDS','QWEN36_KV_CALIBRATE')
    if any(config['env'].get(key) not in (None,'','0',0) for key in forbidden):
        raise ValueError('Teacher/profiler/calibration is forbidden in a speed matrix')
    if any('profiler' in flag.lower() or 'profile' in flag.lower() for flag in config['flags']):
        raise ValueError('Profiler CLI is forbidden in a speed matrix')


def validate_receipt(run,profile,expected=None):
    if not isinstance(run,dict) or run.get('profile')!=profile or not re.fullmatch('[0-9a-f]{32}',str(run.get('run_id',''))):
        raise ValueError('Wrong profile/run ID in source receipt')
    if run.get('checkpoint')!=CHECKPOINT or run.get('checkpoint_revision')!=CHECKPOINT_REVISION:
        raise ValueError('Wrong checkpoint receipt')
    files=run.get('source_files',{})
    if not isinstance(files,dict) or not files or not all(isinstance(name,str) and not pathlib.PurePosixPath(name).is_absolute()
            and '..' not in pathlib.PurePosixPath(name).parts and pathlib.PurePosixPath(name).suffix in SOURCE_SUFFIXES
            and valid_sha(value) for name,value in files.items()):
        raise ValueError('Invalid source file schema')
    # Matches experiment.py's exact digest representation.
    source_hash=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    if run.get('source_sha256')!=source_hash:
        raise ValueError('Source aggregate digest mismatch')
    service=run.get('service_source_files',{})
    if not isinstance(service,dict) or set(service)!=set(SERVICE_FILES) or not all(valid_sha(value) for value in service.values()):
        raise ValueError('Missing/invalid service hashes')
    packages=run.get('packages',{})
    if not isinstance(packages,dict) or set(packages)!=set(PACKAGES) or not all(isinstance(v,str) and v for v in packages.values()):
        raise ValueError('Missing/invalid package receipt')
    config=run.get('configuration');validate_speed_config(config)
    declared={key:value for key,value in config['env'].items() if key.endswith('_LIBRARY')}
    libraries=run.get('native_libraries',{})
    if not isinstance(libraries,dict) or set(libraries)!=set(declared):raise ValueError('Declared native library missing from receipt')
    if not all(isinstance(libraries[key],dict) and libraries[key].get('path')==path
               and valid_sha(libraries[key].get('sha256')) for key,path in declared.items()):
        raise ValueError('Invalid native library receipt')
    identity={key:run[key] for key in ('source_sha256','service_source_files','checkpoint','checkpoint_revision','packages')}
    identity.update(configuration=config,native_libraries=libraries)
    if expected is not None and identity!=expected:raise ValueError('Receipt differs from frozen matrix manifest')
    return identity


def current_run_spec(root,profiles,lengths,output,pairs):
    import importlib.metadata
    source=root/'python/freetoken'
    files={str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest()
           for path in sorted(source.rglob('*')) if path.is_file() and path.suffix in SOURCE_SUFFIXES}
    if not files:raise ValueError('Native source inventory is empty')
    common={'source_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
            'service_source_files':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SERVICE_FILES},
            'checkpoint':CHECKPOINT,'checkpoint_revision':CHECKPOINT_REVISION,
            'packages':{name:importlib.metadata.version(name) for name in PACKAGES}}
    arms={}
    for profile in profiles:
        if not re.fullmatch('[a-z0-9][a-z0-9-]{0,70}',profile):raise ValueError('Invalid profile name')
        config=json.loads((root/'config'/f'{profile}.json').read_text());validate_speed_config(config)
        libraries={key:{'path':value,'sha256':hashlib.sha256(pathlib.Path(value).read_bytes()).hexdigest()}
                   for key,value in config['env'].items() if key.endswith('_LIBRARY')}
        arms[profile]=dict(common,configuration=config,native_libraries=libraries)
    harness={name:hashlib.sha256((pathlib.Path(__file__).parent/name).read_bytes()).hexdigest() for name in HARNESS_FILES}
    # arms[profile] keeps exactly validate_receipt's identity (soak.py IDENTITY_KEYS); the
    # launcher identity is a separate top-level key.
    return {'schema_version':1,'profiles':profiles,'lengths':list(lengths),'output':output,'pairs':pairs,
            'arms':arms,'harness_source_files':harness,'launcher_files':launcher_files(root)}


def launcher_files(root=None):
    root=pathlib.Path(ROOT if root is None else root)
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in LAUNCHER_FILES}


def launcher_environment(raw):
    """Filtered start environment (bytes of /proc/PID/environ) -> sorted dict; secrets redacted."""
    env={}
    for item in raw.split(b'\0'):
        if b'=' not in item:continue
        key,value=item.split(b'=',1);key=key.decode('utf-8','replace')
        if key not in LAUNCHER_ENV_KEYS and not key.startswith(LAUNCHER_ENV_PREFIXES):continue
        env[key]='<redacted>' if LAUNCHER_ENV_SECRET.search(key) else value.decode('utf-8','replace')
    return dict(sorted(env.items()))


def launcher_problems(env,cmdline,profile,root=None):
    """Static launcher contract of run-experiment.sh (V6); [] means the frozen launcher environment."""
    root=pathlib.Path(ROOT if root is None else root);cuda=str(root.parent/'cuda-13.4');problems=[]
    if env.get('PYTHONPATH')!=str(root/'python'):problems.append('pythonpath')
    if env.get('QWEN36_DATA_ROOT')!=str(root):problems.append('data_root')
    if env.get('CUDA_HOME')!=cuda:problems.append('cuda_home')
    if not str(env.get('LD_LIBRARY_PATH','')).startswith(f'{cuda}/lib:{cuda}/lib64:/usr/lib/wsl/lib'):problems.append('ld_library_path')
    if 'LD_PRELOAD' in env:problems.append('ld_preload')
    if 'FREETOKEN_API_LOG_DIR' in env:problems.append('api_log_dir')
    if any(key.startswith('QWEN36_') and key!='QWEN36_DATA_ROOT' for key in env):problems.append('feature_env_in_launcher')
    if len(cmdline)!=3 or not cmdline[0].endswith('/.venv/bin/python') or cmdline[1]!=str(root/'experiment.py') or cmdline[2]!=profile:
        problems.append('cmdline')
    return problems


def launcher_receipt(profile,root=None,show=None,proc=pathlib.Path('/proc')):
    """Launcher identity of the RUNNING experiment unit: file hashes + MainPID start environment.

    show(prop) -> value of `systemctl --user show qwen36-experiment`; injectable for tests.
    Raises RuntimeError/OSError when the identity cannot be established.
    """
    root=pathlib.Path(ROOT if root is None else root)
    if show is None:
        import subprocess
        def show(prop):
            try:result=subprocess.run(['systemctl','--user','show',EXPERIMENT_UNIT,'--property='+prop,'--value'],
                                      capture_output=True,text=True,timeout=10)
            except subprocess.TimeoutExpired:raise RuntimeError('systemctl show timed out') from None
            if result.returncode:raise RuntimeError('systemctl show failed')
            return result.stdout.strip()
    pid,invocation=show('MainPID'),show('InvocationID')
    if not pid.isdigit() or int(pid)<=1 or not invocation:raise RuntimeError('Experiment unit has no main process')
    raw=(proc/pid/'environ').read_bytes()
    cmdline=[part.decode('utf-8','replace') for part in (proc/pid/'cmdline').read_bytes().split(b'\0') if part]
    if show('MainPID')!=pid or show('InvocationID')!=invocation:raise RuntimeError('Experiment process changed while reading its launcher identity')
    env=launcher_environment(raw)
    return {'schema_version':1,'files':launcher_files(root),'environment':env,'environment_sha256':canonical_hash(env),
            'cmdline':cmdline,'main_pid':int(pid),'invocation_id':invocation,
            'problems':launcher_problems(env,cmdline,profile,root)}


def min_gpu_free(rows):
    """Smallest service-sampled GPU free bytes over rows, their primers and tool steps; None if never sampled."""
    values=[]
    def visit(item):
        if not isinstance(item,dict):return
        peaks=item.get('peaks');value=peaks.get('min_gpu_free_bytes') if isinstance(peaks,dict) else None
        if type(value) in (int,float) and math.isfinite(value) and value>=0:values.append(value)
        visit(item.get('primer'))
        for step in item.get('steps') if isinstance(item.get('steps'),list) else ():visit(step)
    for row in rows if isinstance(rows,list) else ():visit(row)
    return min(values) if values else None


def gpu_memory_snapshot(run=None):
    """nvidia-smi used/free/total bytes of GPU 0 (no CUDA context is created); None if unavailable."""
    import subprocess
    run=run or subprocess.run
    try:
        result=run([NVIDIA_SMI,'--id=0','--query-gpu=memory.used,memory.free,memory.total','--format=csv,noheader,nounits'],
                   capture_output=True,text=True,timeout=20)
        if result.returncode:return None
        used,free,total=(int(value.strip())*MIB for value in result.stdout.strip().splitlines()[0].split(','))
    except (OSError,ValueError,IndexError,subprocess.TimeoutExpired):return None
    return {'used_bytes':used,'free_bytes':free,'total_bytes':total}


def run_receipt():
    path=ROOT/'active-run.json'
    if not path.exists():return {'source_receipt':'not_recorded_in_earlier_run'}
    return json.loads(path.read_text())

def headers():
    key=(ROOT/'config/experiment.env').read_text().strip().split('=',1)[1]
    return {'Authorization':'Bearer '+key}

def generate(messages, maximum=128, seed=17, temperature=.6, ignore_eos=False, tools=None, tool_choice=None):
    payload={'model':'qwen36-35b','messages':messages,'max_tokens':maximum,
             'seed':seed,'temperature':temperature,'top_p':.95 if temperature else 1,
             'chat_template_kwargs':{'enable_thinking':False},'ignore_eos':ignore_eos,
             'stream':True,'stream_options':{'include_usage':True}}
    if tools is not None:payload['tools']=tools
    if tool_choice is not None:payload['tool_choice']=tool_choice
    start=time.perf_counter(); first=last=first_reasoning=None; text=[]; reasoning=[]; usage={}; finish=None
    with requests.post('http://127.0.0.1:1919/v1/chat/completions',json=payload,
                       headers=headers(),stream=True,timeout=(5,900)) as response:
        if response.status_code != 200:
            return {'status':response.status_code,'error_code':'http_request_failed'},''
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b'data: ') or line==b'data: [DONE]': continue
            event=json.loads(line[6:])
            if event.get('usage'): usage=event['usage']
            for choice in event.get('choices',[]):
                delta=choice.get('delta',{})
                if delta.get('content'):
                    now=time.perf_counter()
                    if first is None:first=now
                    last=now;text.append(delta['content'])
                if delta.get('reasoning_content'):
                    if first_reasoning is None:first_reasoning=time.perf_counter()
                    reasoning.append(delta['reasoning_content'])
                if choice.get('finish_reason'):finish=choice['finish_reason']
    elapsed=time.perf_counter()-start
    output=''.join(text)
    completion=usage.get('completion_tokens',0)
    result={'status':200,'usage':usage,'ttft_content_s':None if first is None else first-start,
            'elapsed_s':elapsed,'decode_client_tps':(completion-1)/(last-first) if first and last>first else None,
            'finish_reason':finish,'content_sha256':hashlib.sha256(output.encode()).hexdigest(),
            'reasoning_chars':sum(map(len,reasoning)), 'sampling':{'seed':seed,'temperature':temperature,
            'top_p':payload['top_p'],'thinking':False,'max_tokens':maximum,'ignore_eos':ignore_eos},
            'request_id':response.headers.get('x-qwen36-request-id')}
    result['messages_sha256']=hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    result['request_fixture_sha256']=canonical_hash({'messages':messages,'tools':tools,'tool_choice':tool_choice})
    result['ttft_reasoning_s']=None if first_reasoning is None else first_reasoning-start
    attach_metrics(result)
    return result,output


def attach_metrics(result):
    """Bounded join by serving generation and UID after client timing finishes."""
    rid=result.get('request_id')
    if not rid:return
    def recent_rows(directory,name):
        # A rotation may occur between HTTP completion and the engine flush.
        for suffix in ('','.1','.2','.3','.4','.5'):
            path=ROOT/directory/(name+suffix)
            try:lines=path.read_text(encoding='utf-8').splitlines()
            except (OSError,UnicodeError):continue
            for line in reversed(lines):
                try:row=json.loads(line)
                except (ValueError,TypeError):continue
                if isinstance(row,dict):yield row
    record=None
    for attempt in range(60):
        if record is None:
            record=next((row for row in recent_rows('logs','requests.jsonl')
                         if row.get('request_id')==rid),None)
        if record is not None:
            for key in ('cached_prompt_tokens','ttft_generated_s','ttft_tool_s','peaks','counter_deltas','decode_ack_tps'):
                if key in record:result[key]=record[key]
            cached=record.get('cached_prompt_tokens')
            if type(cached) is int and cached>=0:result['cache_hit_observed']=cached>0
            uid,run_id=record.get('engine_uid'),record.get('run_id')
            result['serving_run_id']=run_id
            if uid is None:
                result['engine_metrics_status']='no_engine_uid';return
            if not isinstance(run_id,str) or not run_id:
                # Historical logs cannot establish which engine owned a UID.
                result['engine_metrics_status']='missing_run_id';return
            row=next((row for row in recent_rows('reports','engine.jsonl')
                      if row.get('uid')==uid and row.get('run_id')==run_id),None)
            if row is not None:
                result['engine']={key:value for key,value in row.items() if key not in ('uid','pid')}
                result['engine_metrics_status']='joined';return
        time.sleep(.05)
    result['engine_metrics_status']='engine_row_timeout' if record is not None else 'request_row_timeout'

def sustained_warmup(tokenizer,seconds=None):
    """Opt-in sustained warm-up (QWEN36_BENCH_WARMUP_S seconds, default 0 = off): repeated 16K-token prefill + 128 forced
    tokens on a dedicated synthetic document (variant 271828+i, never a measured fixture) until the budget is spent,
    so clocks/temperature, the expert cache and the prefill/decode paths are in steady state before measurement."""
    seconds=float(os.environ.get('QWEN36_BENCH_WARMUP_S','0') if seconds is None else seconds)
    rows=[]
    if seconds<=0:return rows
    started=time.monotonic();i=0
    while i==0 or time.monotonic()-started<seconds:
        body=document(tokenizer,16384,271828+i)
        result,_=generate([{'role':'system','content':f'Benchmark sustained warmup {i}.'},
                           {'role':'user','content':body+'\nКратко перечисли типы записей.'}],128,23,ignore_eos=True)
        rows.append({'case':'warmup_sustained','iteration':i,'status':result.get('status'),'elapsed_s':result.get('elapsed_s')})
        if result.get('status')!=200:raise RuntimeError('Sustained warmup request failed')
        i+=1
    return rows

def document(tokenizer,length,variant):
    rng=random.Random(variant)
    lines=[]
    for i in range(max(1,length//18)):
        lines.append(f'Record {i}: module {rng.randrange(10000)}; latency {rng.randrange(1,900)} ms; '
                     f'region {rng.choice(["north","south","east","west"])}; '
                     f'checksum {rng.getrandbits(40):010x}; retries {rng.randrange(5)}.\n')
    ids=tokenizer.encode(''.join(lines),add_special_tokens=False,verbose=False)
    while len(ids)<length:ids+=ids
    return tokenizer.decode(ids[:length])


def edited_fixtures(body,repeat):
    """Fixed histories: generated primer answers never become paired inputs."""
    base=[{'role':'system','content':f'Benchmark edited history fixture-v4 {repeat}. Ответ по-русски.'},
          {'role':'user','content':body}]
    history=base+[{'role':'assistant','content':'Готов учитывать изменения в задании.'}]
    old=history+[{'role':'user','content':'Напиши подробно код проверки записей, считая допустимыми задержки до 300 миллисекунд. Объясни сложность.'}]
    new=history+[{'role':'user','content':'Исправление: напиши подробно код проверки записей, считая допустимыми задержки до 450 миллисекунд. Объясни сложность.'}]
    yield 'last_message_edit',old,new,None,None
    tools=[{'type':'function','function':{'name':'fixture_add','description':'Сложение двух целых чисел в тесте.',
        'parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b'],'additionalProperties':False}}}]
    history=base+[{'role':'assistant','content':None,'tool_calls':[{'id':'fixture_call_01','type':'function',
              'function':{'name':'fixture_add','arguments':'{"a":17,"b":25}'}}]}]
    question={'role':'user','content':'По текущему результату функции напиши подробный код валидации записей с таким допустимым количеством повторов. Объясни обработку пропусков и сложность.'}
    old=history+[{'role':'tool','tool_call_id':'fixture_call_01','content':'{"result":41}'},question]
    new=history+[{'role':'tool','tool_call_id':'fixture_call_01','content':'{"result":42}'},question]
    yield 'tool_result_edit',old,new,tools,'none'


def expected_cases(lengths):
    cases={('fresh',length) for length in lengths}
    if 4096 in lengths:cases.add(('history_continuation',4096))
    if 16384 in lengths:cases.update({('last_message_edit',16384),('tool_result_edit',16384)})
    return cases

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--lengths',default=','.join(map(str,DEFAULT_LENGTHS)));p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--output',type=int,default=128);p.add_argument('--quality',action='store_true')
    p.add_argument('--repeat-start',type=int,default=0)
    args=p.parse_args()
    lengths=parse_lengths(args.lengths)
    receipt=run_receipt();validate_receipt(receipt,receipt.get('profile'))
    if not 1<=args.output<=8192 or args.repeats<1 or args.repeat_start<0:raise ValueError('Invalid benchmark dimensions')
    target=ROOT/'reports'/f'{args.name}.json'
    # Receipts are never overwritten; a rerun needs a new name.
    if target.exists():raise RuntimeError('Benchmark report already exists; refusing to overwrite a receipt')
    # V6: launcher identity of the serving process is part of the receipt (compare-ab checks it).
    try:launcher=launcher_receipt(receipt['profile'])
    except (OSError,RuntimeError,ValueError) as error:launcher={'status':'unavailable','error_type':type(error).__name__}
    benchmark={'schema_version':1,'harness_version':HARNESS_VERSION,'fixture_version':BENCH_FIXTURE_VERSION,
               'lengths':list(lengths),'output':args.output,
               'repeat_start':args.repeat_start,'repeats':args.repeats,
               'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    def persist():
        current=run_receipt()
        if current.get('run_id')!=receipt['run_id']:raise RuntimeError('Serving generation changed during benchmark')
        target.write_text(json.dumps({'name':args.name,'run':receipt,'launcher':launcher,'benchmark':benchmark,'rows':rows},indent=2))
    tokenizer=AutoTokenizer.from_pretrained('/path/to/LLM/qwen36-api/model',local_files_only=True)
    rows=[]
    warm,_=generate([{'role':'user','content':'Объясни принцип двоичного поиска.'}],64)
    rows.append(dict(case='warmup',**warm))
    if warm['status']!=200:raise RuntimeError('Warmup request failed')
    rows.extend(sustained_warmup(tokenizer))
    for repeat in range(args.repeat_start,args.repeat_start+args.repeats):
        for length in lengths:
            body=document(tokenizer,length,314159+repeat+length)
            messages=[{'role':'system','content':f'Benchmark independent input fixture-v2 {repeat} {length}. Ответ по-русски.'},
                      {'role':'user','content':body+'\nОбъясни подробно алгоритм поиска аномалий в этих записях. Напиши код на Python и разбор сложности.'}]
            result,answer=generate(messages,args.output,100+repeat,ignore_eos=True)
            row=dict(case='fresh',length=length,repeat=repeat,cache_requirement=cache_requirement('fresh'),**result);rows.append(row)
            print(json.dumps(row),flush=True)
            if result['status']!=200: raise RuntimeError('Benchmark request failed')
            if length==4096:
                # The preceding model's text can differ between numeric backends.
                # One fixed synthetic history is a continuation workload, not a
                # guaranteed prefix hit: the last saved GDN snapshot may lie beyond
                # the common prefix. Actual cached tokens/hit label are recorded;
                # real generated-history cache-hit correctness is prefix-quality's job.
                follow=messages+[{'role':'assistant','content':'Проверяем типы полей, пропуски и допустимый диапазон задержки. Для числовых полей можно использовать медиану и абсолютные отклонения.'},{'role':'user','content':'Добавь обработку отсутствующих значений.'}]
                result,_=generate(follow,args.output,100+repeat,ignore_eos=True)
                rows.append(dict(case='history_continuation',length=length,repeat=repeat,
                                 cache_requirement=cache_requirement('history_continuation'),**result))
                if result['status']!=200:raise RuntimeError('History continuation request failed')
            if length==16384:
                for case,primer_messages,target_messages,tools,tool_choice in edited_fixtures(body,repeat):
                    primer,_=generate(primer_messages,args.output,100+repeat,ignore_eos=True,tools=tools,tool_choice=tool_choice)
                    if primer['status']!=200:raise RuntimeError('Edited-history primer failed')
                    result,_=generate(target_messages,args.output,100+repeat,ignore_eos=True,tools=tools,tool_choice=tool_choice)
                    rows.append(dict(case=case,length=length,repeat=repeat,cache_requirement=cache_requirement(case),primer=primer,**result))
                    if result['status']!=200:raise RuntimeError('Edited-history request failed')
            persist()
    print(json.dumps({'result':'complete','rows':len(rows),'report':str(target)}))

if __name__=='__main__':main()
