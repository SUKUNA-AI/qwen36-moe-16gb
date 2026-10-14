"""Launch an explicit experiment; this does not start a tunnel or enable a service."""
import json, os, pathlib, secrets, sys

ROOT = pathlib.Path('/home/USER/qwen36-api/optimization')

def profile_helpers():
    import subprocess
    if getattr(subprocess.Popen,'qwen36_profile_wrapper',False):return
    original_popen=subprocess.Popen
    class ProfiledPopen(original_popen):
        qwen36_profile_wrapper=True
        def __init__(self,args,*pos,**kw):
            command=args[0] if isinstance(args,(list,tuple)) else ''
            if pathlib.Path(command).name=='file':
                env=dict(kw.get('env') or os.environ)
                env.pop('LD_PRELOAD',None)
                kw['env']=env
            super().__init__(args,*pos,**kw)
    subprocess.Popen=ProfiledPopen

if os.environ.get('QWEN36_NSYS_FORWARDS'):profile_helpers()

def main():
    config = json.loads((ROOT/'config'/f'{sys.argv[1]}.json').read_text())
    # A named profile is a complete experiment, not a patch on the caller's
    # environment. Keep ordinary toolchain settings, clear our feature switches.
    for key in list(os.environ):
        if key.startswith('QWEN36_') or key=='FREETOKEN_DISABLE_OVERLAP_SCHEDULING':
            os.environ.pop(key,None)
    os.environ['QWEN36_DATA_ROOT']=str(ROOT)
    import hashlib,datetime,importlib.metadata,uuid
    os.environ['QWEN36_RUN_ID']=uuid.uuid4().hex
    source=ROOT/'python/freetoken'
    hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob('*')) if p.is_file() and p.suffix in ('.py','.cu','.h','.cuh','.cpp')}
    libraries={key:{'path':value,'sha256':hashlib.sha256(pathlib.Path(value).read_bytes()).hexdigest()}
               for key,value in config.get('env',{}).items() if key.endswith('_LIBRARY') and pathlib.Path(value).is_file()}
    receipt={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'profile':sys.argv[1],
             'run_id':os.environ['QWEN36_RUN_ID'],'native_libraries':libraries,
             'configuration':config,'source_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
             'source_files':hashes,'checkpoint':'nvidia/Qwen3.6-35B-A3B-NVFP4',
             'checkpoint_revision':'1355db6a052410cfd62085d94b58866fd0f2c3c5',
             'packages':{name:importlib.metadata.version(name) for name in ('torch','triton','flashinfer-python','sglang-kernel','transformers')}}
    receipt['service_source_files']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('experiment.py','serve_guarded.py','guard.py','ingress.py','request_metrics.py')}
    (ROOT/'active-run.json').write_text(json.dumps(receipt,indent=2))
    (ROOT/'active-run.json').chmod(0o600)
    secret = ROOT/'config/experiment.env'
    if not secret.exists():
        fd = os.open(secret, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(fd,'w') as out: out.write('QWEN36_API_KEY='+secrets.token_urlsafe(32)+'\n')
    os.environ['QWEN36_API_KEY'] = secret.read_text().strip().split('=',1)[1]
    for key,value in config.get('env',{}).items(): os.environ[key] = str(value)
    if config.get('env',{}).get('QWEN36_NSYS_FORWARDS'):
        # Nsight's injection worker survives /usr/bin/file's main thread on this
        # WSL build, leaving Python platform.architecture() waiting on its pipe.
        # Only this non-GPU helper is excluded; model worker injection is intact.
        profile_helpers()
    context = int(config.get('context',65536))
    chunk = int(config.get('chunk',2048))
    os.environ.update(QWEN36_CONTEXT=str(context), QWEN36_PREFILL_CHUNK=str(chunk),
                      QWEN36_MEMORY_RATIO=str(config.get('memory_ratio',.75)),QWEN36_PROFILE=sys.argv[1])
    sys.argv = ['ft','serve','--model','/path/to/LLM/qwen36-api/model',
        '--served-model-name','qwen36-35b','--host','127.0.0.1','--port','1919',
        '--text-model-only','--cors-origins','','--max-running-requests','1',
        '--max-seq-len-override',str(context),'--max-output-tokens','8192',
        '--max-prefill-length',str(chunk),'--memory-ratio',str(config.get('memory_ratio',.75)),
        '--kv-reserve-tokens',str(context),'--moe-cache-auto','--moe-strategy','offload',
        '--moe-cpu-threads','16','--expert-load','serial','--cache-type','radix',
        '--enable-cache-report'] + config.get('flags',[])
    from serve_guarded import main as serve
    serve(run_id=receipt['run_id'])

if __name__ == '__main__': main()
