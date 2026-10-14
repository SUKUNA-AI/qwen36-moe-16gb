"""Chunk-size screening for the bit-exact CUDA NVFP4 MoE prefill (harness V6; NOT acceptance).

One repeat per profile, serial, under the GPU-exclusive flock (non-blocking, exit 75 if
busy); refuses if a model unit is active. For every profile: identity is re-hashed
(bench.current_run_spec) and the workspace config must equal the native one; nvidia-smi
snapshot before start (foreign VRAM); start-experiment.sh; readiness; receipt bound to
the identity; launcher identity; one warmup, then fresh inputs at 16384/65536/120000
tokens with exactly 128 generated tokens (ignore_eos, bench fixture text, seed 100).
Recorded per request: TTFT, elapsed, client prefill tokens/s (prompt/TTFT), engine
prefill tokens/s (prompt/prefill_gpu_ms), decode tokens/s, service-sampled min GPU free,
engine expert counters (prefill_h2d_bytes, prefill_d2d_bytes, decode_missing,
decode_h2d_payload_bytes, ... - logical payload bytes, not PCIe packets) and expert
slots. Only the unit this process started is stopped. Each profile's result is written
once (O_EXCL) to reports/NAME-PROFILE-chunk-screen.json, the summary to
reports/NAME-chunk-screen.json. Exit: 0 all measured; 1 a profile failed; 2 refused; 75 lock busy.
Selection rule suggested in harness-v6-runbook.md; a screening result never qualifies a profile.
"""
import argparse,hashlib,json,os,subprocess,sys,time
import bench
from bench import GpuLockBusy,acquire_gpu_lock,current_run_spec,parse_lengths,validate_receipt
import matrix

DEFAULT_PROFILES=('cuda-bitexact128-c2048','cuda-bitexact128','cuda-bitexact128-c6144','cuda-bitexact128-c8192','cuda-bitexact128-c16384')
DEFAULT_LENGTHS=(16384,65536,120000)
ENGINE_KEYS=('prefill_gpu_ms','decode_gpu_ms','engine_span_s','prefill_forward_tokens','decode_forwards','expert_slots',
             'valid_gpu_expert_slots','expert_counter_deltas','request_torch_peak_allocated_bytes',
             'request_torch_peak_reserved_bytes','request_allocator_peak_above_end_bytes','expert_transfer_accounting')


def counter(row,key):
    deltas=(row.get('engine') or {}).get('expert_counter_deltas')
    return deltas.get(key) if isinstance(deltas,dict) else None


def write_new(path,document):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as target:target.write(json.dumps(document,indent=2,allow_nan=False)+'\n')


def summarize_request(length,result):
    usage=result.get('usage') if isinstance(result.get('usage'),dict) else {}
    engine=result.get('engine') if isinstance(result.get('engine'),dict) else {}
    prompt=usage.get('prompt_tokens');ttft=result.get('ttft_content_s');prefill_ms=engine.get('prefill_gpu_ms')
    peaks=result.get('peaks') if isinstance(result.get('peaks'),dict) else {}
    return {'length':length,'status':result.get('status'),'prompt_tokens':prompt,
            'completion_tokens':usage.get('completion_tokens'),'finish_reason':result.get('finish_reason'),
            'ttft_content_s':ttft,'elapsed_s':result.get('elapsed_s'),'decode_client_tps':result.get('decode_client_tps'),
            'prefill_client_tps':prompt/ttft if isinstance(prompt,int) and isinstance(ttft,float) and ttft>0 else None,
            'prefill_engine_tps':prompt/(prefill_ms/1000) if isinstance(prompt,int) and isinstance(prefill_ms,(int,float)) and prefill_ms>0 else None,
            'min_gpu_free_bytes':peaks.get('min_gpu_free_bytes'),'gpu_total_used_peak_bytes':peaks.get('gpu_total_used_bytes'),
            'cached_prompt_tokens':result.get('cached_prompt_tokens'),'engine_metrics_status':result.get('engine_metrics_status'),
            'engine':{key:engine[key] for key in ENGINE_KEYS if key in engine},
            'content_sha256':result.get('content_sha256'),'request_fixture_sha256':result.get('request_fixture_sha256')}


def measure(lengths,output,tokenizer):
    rows=[]
    warm,_=bench.generate([{'role':'user','content':'Объясни принцип двоичного поиска.'}],64)
    rows.append(dict(summarize_request(0,warm),case='warmup'))
    if warm.get('status')!=200:raise RuntimeError('Warmup request failed')
    rows.extend(bench.sustained_warmup(tokenizer))
    for length in lengths:
        body=bench.document(tokenizer,length,314159+length)
        messages=[{'role':'system','content':f'Benchmark independent input fixture-v2 0 {length}. Ответ по-русски.'},
                  {'role':'user','content':body+'\nОбъясни подробно алгоритм поиска аномалий в этих записях. Напиши код на Python и разбор сложности.'}]
        result,_=bench.generate(messages,output,100,ignore_eos=True)
        row=dict(summarize_request(length,result),case='fresh');rows.append(row)
        print(json.dumps({k:row[k] for k in ('case','length','status','ttft_content_s','elapsed_s','decode_client_tps','min_gpu_free_bytes')}),flush=True)
        if result.get('status')!=200:raise RuntimeError('Screening request failed')
    return rows


def screen_profile(root,name,profile,lengths,output,tokenizer):
    path=root/'reports'/f'{name}-{profile}-chunk-screen.json'
    if path.exists():raise RuntimeError('Screening receipt already exists for this profile; use a new --name')
    row={'kind':'chunk-screen','acceptance':False,'name':name,'profile':profile,'harness_version':bench.HARNESS_VERSION,
         'lengths':list(lengths),'output':output,'repeats':1,
         'screen_sha256':hashlib.sha256(open(__file__,'rb').read()).hexdigest()}
    invocation=None;step='preflight';interrupted=None
    try:
        if not matrix.units_inactive():raise RuntimeError('A model service is active')
        spec=current_run_spec(root,[profile],bench.DEFAULT_LENGTHS,output,1)
        if matrix.workspace_config_drift(root,profile):raise RuntimeError('Workspace profile config differs from native config/')
        row.update(configuration=spec['arms'][profile]['configuration'],native_libraries=spec['arms'][profile]['native_libraries'],
                   launcher_files=spec['launcher_files'],harness_source_files=spec['harness_source_files'])
        row['pre_start_gpu']=bench.gpu_memory_snapshot()
        started=time.monotonic();step='start'
        matrix.run(['bash',str(root/'start-experiment.sh'),profile],60)
        invocation=matrix.state('InvocationID')
        if not invocation:raise RuntimeError('Started experiment has no invocation ID')
        step='readiness'
        matrix.run([matrix.PY,str(root/'check-experiment.py'),'420'],430)
        step='receipt'
        receipt=json.loads((root/'active-run.json').read_text());validate_receipt(receipt,profile,spec['arms'][profile])
        row.update(load_ready_s=time.monotonic()-started,run_id=receipt['run_id'])
        step='launcher'
        launcher=bench.launcher_receipt(profile,root)
        row['launcher']={key:launcher[key] for key in ('files','environment_sha256','problems')}
        if launcher['files']!=spec['launcher_files'] or launcher['problems']:raise RuntimeError('Launcher identity differs')
        row['startup_cgroup_memory_peak_bytes']=matrix.cgroup_peak(invocation)
        step='requests'
        row['rows']=measure(lengths,output,tokenizer)
        current=json.loads((root/'active-run.json').read_text())
        if current.get('run_id')!=receipt['run_id']:raise RuntimeError('Serving generation changed during screening')
        row['min_gpu_free_bytes']=bench.min_gpu_free([{'peaks':{'min_gpu_free_bytes':x.get('min_gpu_free_bytes')}} for x in row['rows']])
        row['vram_gate']=('UNAVAILABLE' if row['min_gpu_free_bytes'] is None else
                          'PASS' if row['min_gpu_free_bytes']>=bench.VRAM_FLOOR_BYTES else 'BELOW_FLOOR')
        row['status']='measured'
    except (RuntimeError,subprocess.TimeoutExpired,ValueError,OSError,KeyError,TypeError) as error:
        row.update(status='failed',failed_step=step,error_type=type(error).__name__,message=str(error)[:200])
    except (KeyboardInterrupt,SystemExit) as error:
        row.update(status='failed',failed_step=step,error_type=type(error).__name__,interrupted=True);interrupted=error
    finally:
        try:
            if invocation is None and step=='start':
                orphan=matrix.state('InvocationID')
                if orphan:invocation=orphan
            if invocation and matrix.state('InvocationID')==invocation:
                matrix.run(['systemctl','--user','stop','qwen36-experiment'],60);row['stopped_own_invocation']=True
        except (RuntimeError,subprocess.TimeoutExpired,OSError) as error:row['stop_error']=type(error).__name__
        write_new(path,row)
    print(json.dumps({k:row.get(k) for k in ('profile','status','failed_step','message','error_type','load_ready_s','min_gpu_free_bytes','vram_gate')}),flush=True)
    if interrupted is not None:raise interrupted
    return row


def main(argv=None,root=None,lock_path=None,tokenizer=None):
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--profiles',default=','.join(DEFAULT_PROFILES))
    p.add_argument('--lengths',default=','.join(map(str,DEFAULT_LENGTHS)));p.add_argument('--output',type=int,default=128)
    a=p.parse_args(argv);profiles=a.profiles.split(',');lengths=parse_lengths(a.lengths)
    if not matrix.NAME_PATTERN.fullmatch(a.name) or not profiles or len(set(profiles))!=len(profiles) \
            or not all(matrix.NAME_PATTERN.fullmatch(x) for x in profiles) or not 1<=a.output<=8192:
        p.error('invalid --name/--profiles/--output')
    root=bench.ROOT if root is None else root
    summary_path=root/'reports'/f'{a.name}-chunk-screen.json'
    if summary_path.exists():
        print(json.dumps({'result':'refused','reason':'summary_exists'}),file=sys.stderr,flush=True);return matrix.EXIT_REFUSED
    try:lock=acquire_gpu_lock(lock_path)
    except GpuLockBusy as error:
        print(json.dumps({'result':'refused','reason':'gpu_lock_busy','message':str(error)}),file=sys.stderr,flush=True)
        return matrix.EXIT_GPU_LOCK_BUSY
    try:
        matrix.install_signal_handlers()
        if not matrix.units_inactive():
            print(json.dumps({'result':'refused','reason':'model_unit_active'}),file=sys.stderr,flush=True);return matrix.EXIT_REFUSED
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained('/path/to/LLM/qwen36-api/model',local_files_only=True)
        results=[screen_profile(root,a.name,profile,lengths,a.output,tokenizer) for profile in profiles]
        summary={'kind':'chunk-screen-summary','acceptance':False,'name':a.name,'profiles':profiles,'lengths':list(lengths),
                 'output':a.output,'vram_floor_bytes':bench.VRAM_FLOOR_BYTES,
                 'results':[{'profile':r['profile'],'status':r['status'],'chunk':(r.get('configuration') or {}).get('chunk'),
                             'memory_ratio':(r.get('configuration') or {}).get('memory_ratio'),'load_ready_s':r.get('load_ready_s'),
                             'min_gpu_free_bytes':r.get('min_gpu_free_bytes'),'vram_gate':r.get('vram_gate'),
                             'foreign_gpu_used_bytes_pre_start':(r.get('pre_start_gpu') or {}).get('used_bytes'),
                             'fresh':[{k:x.get(k) for k in ('length','ttft_content_s','elapsed_s','prefill_client_tps','prefill_engine_tps',
                                                         'decode_client_tps','min_gpu_free_bytes')}
                                      | {'prefill_h2d_bytes':counter(x,'prefill_h2d_bytes'),'decode_missing':counter(x,'decode_missing'),
                                         'expert_slots':(x.get('engine') or {}).get('expert_slots')}
                                      for x in r.get('rows',[]) if x.get('case')=='fresh']} for r in results]}
        write_new(summary_path,summary)
        print(json.dumps({'result':'complete','report':summary_path.name}),flush=True)
        return 0 if all(r['status']=='measured' for r in results) else 1
    finally:os.close(lock)


if __name__=='__main__':raise SystemExit(main())
