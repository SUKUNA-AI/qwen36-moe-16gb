"""Save current identity, resource settings and actual placement; never serialize env secrets."""
import hashlib,importlib.metadata,json,pathlib,platform,subprocess,urllib.request
root=pathlib.Path('/home/USER/qwen36-api');out=pathlib.Path(__file__).resolve().parent
key=(root/'secrets/api.env').read_text().split('=',1)[1].strip()
def get(path):
    req=urllib.request.Request('http://127.0.0.1:1919'+path,headers={'Authorization':'Bearer '+key})
    with urllib.request.urlopen(req,timeout=10) as response:return json.load(response)
health=get('/health');stats=get('/v1/stats');models=get('/v1/models')
assert health['status']=='ok' and health['model']=='qwen36-35b' and models['data'][0]['context_length']==65536
for pattern,dest in [('placement-*.json','placement-offload-64k.json'),('gpu-cache-*.json','gpu-cache-offload-64k.json')]:
    source=max((root/'reports').glob(pattern),key=lambda p:p.stat().st_mtime)
    (out/dest).write_bytes(source.read_bytes())
versions={name:importlib.metadata.version(name) for name in ['freetoken','torch','transformers','triton','flashinfer-python','sglang-kernel','nvidia-ml-py']}
versions.update(python=platform.python_version(),freetoken_commit='cac247a860e316e06580d05aeb05f2e647bde214',
                checkpoint='nvidia/Qwen3.6-35B-A3B-NVFP4',checkpoint_revision='1355db6a052410cfd62085d94b58866fd0f2c3c5',
                caddy='2.11.4',cuda_jit_toolkit='13.4.1 redistrib; nvcc 13.4.59',torch_cuda_build='cu130',
                compiler='Arch gcc15 15.3.0+r0.g4db0e8df15be-2',driver='617.14',quant='NVFP4 experts / dense FP8 / selected BF16',kv_dtype='BF16',
                kernel='NVFP4 via Triton',attention='FlashInfer, hybrid_radix',
                config={'context':65536,'output_max':8192,'memory_ratio':.75,'prefill_chunk':2048,'concurrency':1,'queue':0,'cpu_threads':16,'moe_strategy':'offload','expert_load':'serial','speculation':False,'vision':False,'router_top_k':8})
(out/'versions.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
inv=subprocess.check_output(['systemctl','--user','show','qwen36-inference.service','-p','InvocationID','--value'],text=True).strip()
log=subprocess.check_output(['journalctl','--user','_SYSTEMD_INVOCATION_ID='+inv,'-o','cat','--no-pager'],text=True)
assert key not in log and 'PRIVATE_METRICS_CANARY' not in log
# Native informational logs (no request body logging) from the current service invocation.
(out/'startup-final-64k.log').write_text(log,encoding='utf-8')
resources=subprocess.check_output(['systemctl','--user','show','qwen36-inference.service','-p','MemoryCurrent','-p','MemoryPeak','-p','MemorySwapCurrent','-p','NRestarts'],text=True)
document={'health':health,'models':models,'stats':stats,'resources':dict(line.split('=',1) for line in resources.splitlines()),
          'state_user_unit':subprocess.run(['systemctl','--user','is-enabled','qwen36-inference.service'],capture_output=True,text=True).stdout.strip(),
          'placement_files':['placement-offload-64k.json','gpu-cache-offload-64k.json']}
(out/'runtime-final.json').write_text(json.dumps(document,indent=2),encoding='utf-8')
print(json.dumps({'ready':True,'context':65536,'unit_enabled':False,'resources':document['resources']}))
