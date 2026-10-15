"""Offline evidence/source exporter. Never launches a runtime or changes qualification.

Prepared only; invoke after source/results freeze. Uses stdlib and read-only Git.
JSON evidence is deliberately lossy: numeric values and controlled labels survive;
prompts, generations, credentials, machine identities and raw logs never transfer.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import difflib
import fnmatch
import hashlib
import json
import math
import pathlib
import re
import subprocess
import uuid

BASE = 'cac247a860e316e06580d05aeb05f2e647bde214'
CHECKPOINT = 'nvidia/Qwen3.6-35B-A3B-NVFP4'
CHECKPOINT_REVISION = '1355db6a052410cfd62085d94b58866fd0f2c3c5'
from repo_paths import CHECKOUT, SERVING, FREETOKEN
WORKSPACE = CHECKOUT
PUBLIC = pathlib.Path(__file__).resolve().parent
SERVICE = SERVING
SOURCE_SUFFIXES = {'.py','.cu','.cuh','.cpp','.cc','.c','.h','.hpp'}
SOURCE_LIMIT = 4 * 1024 * 1024
JSON_LIMIT = 32 * 1024 * 1024
DROP = object()

DEFAULT_RESULTS = (
    'public:backend-review-results.json', 'public:engine-metrics-tests.json',
    'public:ops-control-tests.json', 'public:ops-readiness-tests.json',
    'public:acceptance-cpu-review-results.json',
    'public:prefix-snapshot-identity.json', 'public:remaining-layers-review.json',
    'public:windows-wheel-audit.json',
    'public:cuda-kv/results.json', 'public:cuda-kv/bench-bf16.json',
    'public:cuda-kv/bench-fp8.json', 'public:cuda-kv/numeric.json',
    'public:cuda-nvfp4/results.json', 'public:cuda-nvfp4/high-slot.json',
    'public:cuda-nvfp4/manifest.json', 'public:cuda-nvfp4/profile-analysis.json',
    'public:cuda-nvfp4/tuned-comparison.json',
    'public:cuda-nvfp4/matched/manifest.json', 'public:cuda-nvfp4/matched/paired-results.json',
    'public:cuda-nvfp4/matched/stage-results.json', 'public:cuda-nvfp4/matched/profile-analysis.json',
    'public:cuda-layer-fusion/manifest.json', 'public:cuda-layer-fusion/results.json',
    'public:cuda-layer-fusion/profile-analysis.json',
    'public:cuda-transport/manifest.json', 'public:cuda-transport/persistent-test.json',
    'public:cuda-transport/persistent-bench.json', 'public:cuda-transport/transport-test.json',
    'public:cuda-transport/transport-bench.json',
    'public:cuda-transport/nsight-systems-sanitized.json',
    'public:cuda-transport/nsight-compute-sanitized.json', 'service:profiles.json',
)

# Exact names only; never recursively export the service directory.
OVERLAY = (
    'guard.py','ingress.py','request_metrics.py','serve_guarded.py','placement.py',
    'readiness.py','keepalive.py','metrics_summary.py','profile-runtime.py','profiles.json',
    'start-inference.sh','qwen36-inference.service',
    'manage.ps1','supervisor.ps1','watch-supervisor.ps1','host-metrics.ps1',
    'get-api-key.ps1',
)

GROUPS = {
    'A': {'name':'telemetry_api_scheduler_correctness','patterns':[
        'python/freetoken/engine/telemetry.py','python/freetoken/engine/qualification.py',
        'python/freetoken/quality_evidence.py',
        'python/freetoken/engine/engine.py','python/freetoken/engine/sample.py','python/freetoken/core.py',
        'python/freetoken/scheduler/prefill.py','python/freetoken/scheduler/scheduler.py',
        'python/freetoken/server/*','python/freetoken/message/tokenizer.py','python/freetoken/tokenizer/*',
        'python/freetoken/moe/offload_cache.py','tests/scheduler/test_overlap_output_budget.py',
        'tests/scheduler/test_prefix_snapshot_identity.py','python/freetoken/checkpoint/ftw.py','tests/checkpoint/*'],
        'switches':['QWEN36_ENGINE_METRICS','QWEN36_NSYS_FORWARDS','QWEN36_SYNC_LEAN'],
        'disable_note':'ENGINE_METRICS=0 disables numeric engine metrics. Correctness/security fixes have no safe feature-off flag; revert reviewed integration/overlay changes as a release.'},
    'B': {'name':'cuda_kernels_and_configuration','patterns':[
        'python/freetoken/engine/engine.py','python/freetoken/core.py','python/freetoken/layers/embedding.py',
        'python/freetoken/layers/quantization/moe/nvfp4.py','python/freetoken/moe/fused_nvfp4.py',
        'python/freetoken/moe/offload_cache.py','python/freetoken/moe/cuda*',
        'python/freetoken/kernel/csrc/qwen36_*.cu','python/freetoken/kernel/cuda*','python/freetoken/gdn_options.py',
        'python/freetoken/kernel/pinned_ctypes.py',
        'python/freetoken/models/qwen3_5_moe/attention.py','python/freetoken/models/qwen3_5_moe/gdn.py',
        'python/freetoken/kernel/triton/fp8_pertensor_linear.py',
        'python/freetoken/layers/quantization/linear/fp8_tensor.py','tests/kernels/test_cuda_*',
        # NVFP4 FP32 global scales (loader/banks/quantization) and the prebuilt-only kernel loader/provenance.
        'python/freetoken/layers/quantization/linear/nvfp4.py','python/freetoken/layers/quantization/moe/base.py',
        'python/freetoken/models/loader.py','python/freetoken/models/nvfp4_banks.py','python/freetoken/models/*/weight.py',
        'python/freetoken/kernel/utils.py','python/freetoken/kernel/aot_models.py','python/freetoken/kernel/kernel_provenance.py'],
        'switches':['QWEN36_NVFP4_PREFILL','QWEN36_CUDA_NVFP4_LIBRARY','QWEN36_PREFILL_TILE','QWEN36_NVFP4_GLOBAL_FP16_COMPAT',
                    'FREETOKEN_DISABLE_JIT','FREETOKEN_STRICT_PROVENANCE',
                    'QWEN36_FUSED_ATTN_GATE','QWEN36_FUSED_ATTN_GATE_LIBRARY',
                    'QWEN36_CUDA_GDN','QWEN36_CUDA_GDN_LIBRARY','QWEN36_CUDA_GDN_DECODE','QWEN36_CUDA_GDN_FLA',
                    'QWEN36_FP8_GEMV','QWEN36_FP8_GEMV_LIBRARY','QWEN36_OPROJ_GEMV','QWEN36_OPROJ_GEMV_LIBRARY',
                    'QWEN36_MOE_DECODE_CUDA','QWEN36_MOE_DECODE_LIBRARY'],
        'disable_note':'Unset NVFP4_PREFILL, PREFILL_TILE, FUSED_ATTN_GATE, CUDA_GDN and FP8_GEMV for their baseline paths; library path alone does not enable a feature (CUDA_GDN_DECODE only selects the CUDA GDN decode variant). Kernel files still depend on shared integration.'},
    'C': {'name':'expert_residency_and_transport','patterns':[
        'python/freetoken/engine/engine.py','python/freetoken/moe/offload_cache.py',
        'python/freetoken/moe/fused_nvfp4.py','python/freetoken/moe/complement*',
        'python/freetoken/moe/cuda_transport.py','python/freetoken/kernel/csrc/qwen36_complement.cu',
        'python/freetoken/kernel/csrc/qwen36_transport.cu','python/freetoken/*/layer_major.py',
        'python/freetoken/layers/moe.py','python/freetoken/decode_overlap.py','python/freetoken/kernel/decode_gather.py',
        'python/freetoken/kernel/csrc/jit/decode_gather.cuh','python/freetoken/kernel/aot.py',
        'tests/moe/test_decode_gather.py','tests/moe/test_decode_overlap_gpu.py','python/freetoken/moe/slot_refill.py',
        'python/freetoken/engine/host_embeddings.py','python/freetoken/kernel/triton/host_rows.py',
        # D3b pre-gate prefetch and D5 fused decode epilogue (next-candidate levers, default off).
        'python/freetoken/moe/pregate.py','python/freetoken/moe/shared_epilogue.py',
        # T1/T3, C0/C1/D6 transport (offload_kernels, route_trace, batch_memcpy probe), host banks, memory budget.
        'python/freetoken/models/qwen3_5_moe/moe.py','python/freetoken/moe/offload_kernels.py','python/freetoken/moe/route_trace.py',
        'python/freetoken/moe/host_banks.py','python/freetoken/moe/benchbw.py','python/freetoken/moe/cpu_executor.py',
        'python/freetoken/kernel/csrc/cpu_moe/*','python/freetoken/kernel/batch_memcpy.py',
        'python/freetoken/engine/cache_budget.py','python/freetoken/engine/memory_history.py',
        'tests/moe/test_*','tests/engine/test_cache_budget.py','tests/engine/test_host_embeddings.py','tests/layers/test_host_embedding_gpu.py'],
        'switches':['QWEN36_RESIDENCY','QWEN36_CACHE_POLICY','QWEN36_CACHE_HALFLIFE',
                    'QWEN36_COMPLEMENT_TRANSPORT','QWEN36_COMPLEMENT_LIBRARY',
                    'QWEN36_LAYER_MAJOR_PREFILL','QWEN36_LAYER_MAJOR_MIN_TOKENS','QWEN36_LAYER_MAJOR_REFILL','QWEN36_DECODE_GATHER',
                    'QWEN36_LAYER_MAJOR_REFILL_POLICY','QWEN36_LAYER_MAJOR_ATTN_BATCH','TVM_FFI_CUDA_ARCH_LIST','QWEN36_DECODE_SHARED_OVERLAP',
                    'QWEN36_HOST_EMBEDDINGS','QWEN36_PREFILL_COPY_PRIORITY','QWEN36_PREFILL_EPOCH_INVALIDATE',
                    'QWEN36_PREFILL_SELECTED','QWEN36_PREFILL_SELECTED_SYNC','QWEN36_PREFILL_WARM_DECODE','QWEN36_MEMORY_HISTORY',
                    'QWEN36_DECODE_PREGATE','QWEN36_DECODE_PREGATE_ATTN','QWEN36_DECODE_SHARED_EPILOGUE'],
        'disable_note':'Unset RESIDENCY to retain full host banks. Complement/fixed remain experiment choices; exporter does not accept their RAM or speed gates. Unset LAYER_MAJOR_PREFILL (and LAYER_MAJOR_REFILL, its POLICY, ATTN_BATCH) for chunk-major prefill and DECODE_GATHER for the default fast_index_copy gather (TVM_FFI_CUDA_ARCH_LIST is only its pinned build arch); aot.py still lists the decode-gather module, so a rebuilt kernel cache ships its .so either way. Unset DECODE_SHARED_OVERLAP for serial T3 decode, HOST_EMBEDDINGS to keep the embedding table in VRAM, PREFILL_COPY_PRIORITY (with EPOCH_INVALIDATE) for the default C0 copy stream and PREFILL_SELECTED for full-layer C1 prefill copies (SELECTED needs COPY_PRIORITY and the decode-gather prebuilt).'},
    'D': {'name':'kv_gdn_and_prefix_state','patterns':[
        'python/freetoken/engine/engine.py','python/freetoken/core.py',
        'python/freetoken/attention/*','python/freetoken/kvcache/*',
        'python/freetoken/models/qwen3_5_moe/attention.py','python/freetoken/models/qwen3_5_moe/gdn*',
        'python/freetoken/kernel/fla/*','python/freetoken/fla_constants.py','python/freetoken/kernel/triton/fp8_kv.py',
        'python/freetoken/kernel/cuda_kv*','python/freetoken/kernel/csrc/qwen36_kv.cu',
        'python/freetoken/scheduler/prefill.py','python/freetoken/scheduler/scheduler.py',
        'tests/scheduler/test_prefix_snapshot_identity.py','python/freetoken/scheduler/cache.py',
        'tests/scheduler/test_retain_chunk_snapshot.py','tests/kernels/test_gdn_fp32_snapshot.py','tests/kvcache/*'],
        'switches':['QWEN36_KV_DTYPE','QWEN36_KV_SCALES','QWEN36_KV_CALIBRATE',
                    'QWEN36_FUSED_QK_KV','QWEN36_FUSED_KV_LIBRARY','QWEN36_PREFIX_MARGIN',
                    'QWEN36_FP32_GDN_SNAPSHOT','QWEN36_RETAIN_CHUNK_SNAPSHOT'],
        'disable_note':'KV_DTYPE=bf16 and unset FUSED_QK_KV disable optional FP8/fusion. Prefix margin, FP32 snapshot and retained chunk snapshot choices require their own numerical evidence; do not infer acceptance from source presence.'},
    'E': {'name':'speculative_decoding','patterns':[
        'python/freetoken/engine/engine.py','python/freetoken/engine/sample.py',
        'python/freetoken/engine/mtp.py','python/freetoken/engine/speculative.py',
        'python/freetoken/scheduler/speculative.py','python/freetoken/scheduler/scheduler.py',
        'python/freetoken/models/qwen3_5_moe/model.py','python/freetoken/models/qwen3_5_moe/gdn*',
        'python/freetoken/attention/linear.py','python/freetoken/kvcache/*',
        'tests/engine/test_speculative.py','tests/scheduler/test_speculative_state.py'],
        'switches':['QWEN36_SPEC','QWEN36_SPEC_WINDOW','QWEN36_SPEC_SAMPLING'],
        'disable_note':'Unset SPEC; window/sampling flags do not independently enable drafting. No MTP/lookup qualification decision is made here.'},
}

PRIVATE_KEYS = {
    'authorization','headers','api_key','x_api_key','key','password','secret','secrets','credential','credentials',
    'env','environment','environ','bearer','cookie','cookies',
    'token','access_token','refresh_token','messages','prompt','prompts','response','responses','answer','answers',
    'content','text','body','arguments','tool_result','tool_results','tool_calls','token_ids','ids','input_ids',
    'stdout','stderr','test_output','traceback','error_message','raw','raw_profiles_private','raw_profile',
    'pid','ppid','uid','instance_id','session_id','request_id','run_id','hostname','username','account_id',
    'zone_id','gpu_uuid','uuid','ip','address','started','started_utc','finished_utc','ts','utc',
    'note','notes','scope','description','evidence','command','commands','source_files',
}
LABEL_KEYS = {
    'status','result','case','profile','name','dtype','kv_dtype','format','quant','quantization','precision',
    'device','gpu','kernel','baseline','candidate','mode','policy','strategy','decode_target','target',
    'model','checkpoint','checkpoint_revision','compiler','host_compiler','cuda','torch','triton','version',
    'nvcc','nsys','ncu','architecture','scope_type','error_type','engine_metrics_status','test',
    'operation','implementation','arm','profile_arm','layout','memory_type','metric_name','metric_unit',
    'default','selected_profile',
}


def digest(data): return hashlib.sha256(data).hexdigest()


def git(repo,*args):
    return subprocess.run(['git','--no-pager','-C',str(repo),'-c','core.autocrlf=false',*args],
                          check=True,capture_output=True,timeout=60).stdout


def relative(value):
    path=pathlib.PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not re.fullmatch(r'[A-Za-z0-9_./-]+',value):
        raise ValueError('Unsafe relative selector')
    return path


def within(root,name):
    path=root/pathlib.Path(relative(name))
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):raise ValueError('Symlink or path escape')
    return path


def checked_source_data(data):
    if len(data)>SOURCE_LIMIT or b'\0' in data:raise ValueError('Non-source or excessive source file')
    text=data.decode('utf-8')
    if re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',text):raise ValueError('Private key detected in source')
    if re.search(r'Bearer\s+[A-Za-z0-9_-]{24,}',text):raise ValueError('Literal bearer token detected in source')
    if re.search(r'''(?i)(?:api[_-]?key|password|secret|access_token)[\w"'\]\s]{0,20}[:=]\s*["']([A-Za-z0-9_+/=-]{24,})''',text):
        raise ValueError('Possible literal credential detected in source')
    return data


def source_bytes(path):
    if path.stat().st_size>SOURCE_LIMIT:raise ValueError('Excessive source file')
    return checked_source_data(path.read_bytes())


def allowed_source(name):
    return (name.startswith('python/freetoken/') or name.startswith('tests/')) and pathlib.PurePosixPath(name).suffix in SOURCE_SUFFIXES and '__pycache__' not in name.split('/')


def new_file_patch(name,data):
    text=data.decode('utf-8')
    body=list(difflib.unified_diff([],text.splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+name,lineterm='\n'))
    fixed=[]
    for line in body:
        fixed.append(line)
        if not line.endswith('\n'):fixed.append('\n\\ No newline at end of file\n')
    return ('diff --git a/'+name+' b/'+name+'\nnew file mode 100644\n'+''.join(fixed)).encode()


def feature_groups(name):
    return [key for key,group in GROUPS.items() if any(fnmatch.fnmatchcase(name,pattern) for pattern in group['patterns'])]


def observed_switches(data):
    # Extract names only, never values. Source does not need importing/executing.
    return sorted(set(re.findall(r'\b(?:QWEN36|FREETOKEN)_[A-Z0-9_]+\b',data.decode('utf-8'))))


def scrub(value,counts,key='',depth=0):
    if depth>48:counts['depth_limit']+=1;return DROP
    normal=key.lower().replace('-','_')
    sensitive_name=re.search(r'(?:^|_)(?:api_?key|api_token|access_token|refresh_token|secret|password|authorization|credential|cookie)(?:_|$)',normal)
    if normal in PRIVATE_KEYS or sensitive_name or normal.endswith(('_pid','_started','_utc','_address','_uuid')):
        counts['private_or_freeform_field']+=1;return DROP
    if value is None or type(value) is bool:return value
    if type(value) in (int,float):
        if isinstance(value,float) and not math.isfinite(value):counts['nonfinite']+=1;return None
        return value
    if isinstance(value,list):
        # A token array can reconstruct private text; token counts remain scalars.
        if normal.endswith('_tokens') or normal in ('tokens','logits','input','output'):
            counts['reconstructable_array']+=1;return DROP
        items=[scrub(item,counts,key,depth+1) for item in value]
        return [item for item in items if item is not DROP]
    if isinstance(value,dict):
        result={}
        for child_key,child in value.items():
            if not isinstance(child_key,str):continue
            # Keys can themselves be absolute paths in file manifests. Strip
            # owner directories and retain a relative source/library basename.
            export_key=child_key
            if '\\' in export_key or export_key.startswith('/') or re.match(r'^[A-Za-z]:',export_key):
                export_key=export_key.replace('\\','/').rsplit('/',1)[-1]
                counts['path_key_reduced']+=1
            if len(export_key)>200 or not re.fullmatch(r'[A-Za-z0-9_.:+ /<>()%\[\],=-]+',export_key):
                counts['non_schema_key']+=1;continue
            clean=scrub(child,counts,child_key,depth+1)
            if clean is not DROP:
                if export_key in result:raise ValueError('Sanitization key collision')
                result[export_key]=clean
        return result
    if isinstance(value,str):
        if re.fullmatch(r'[a-fA-F0-9]{40}|[a-fA-F0-9]{64}',value):
            # A hexadecimal credential is not a hash merely because its length
            # matches SHA. Require an explicit hash/revision or file-map key.
            if (re.search(r'(?:sha|hash|checksum|revision|commit)',normal) or
                pathlib.PurePosixPath(normal).suffix in SOURCE_SUFFIXES|{'.so','.json','.ps1','.cmd','.sh','.service'}):
                return value.lower()
            counts['untyped_hex_string']+=1;return DROP
        if value==CHECKPOINT:return value
        if re.search(r'(?:[A-Za-z]:[\\/]|/home/|/mnt/|/Users/|https?://|@|\b(?:\d{1,3}\.){3}\d{1,3}\b|GPU-[0-9a-f-]+)',value):
            counts['location_or_identity_string']+=1;return DROP
        if normal not in LABEL_KEYS and re.fullmatch(r'[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?',value):
            number=float(value) if any(mark in value for mark in '.eE') else int(value)
            if isinstance(number,float) and not math.isfinite(number):counts['nonfinite']+=1;return None
            return number
        if normal in ('nvcc','nsys','ncu'):
            match=re.search(r'(?i)(?:release |version |V)(\d+(?:\.\d+)+(?:[A-Za-z0-9_.+-]*))',value)
            if match:return match.group(1)
        if normal in LABEL_KEYS and len(value)<=200 and re.fullmatch(r'[A-Za-z0-9_.:+ (),*/<>\[\]=-]+',value):return value
        # Package versions and uppercase status enums are useful and content-free.
        if re.fullmatch(r'\d+\.\d+(?:[A-Za-z0-9_.+()-]*)',value) or re.fullmatch(r'[A-Z][A-Z0-9_]{1,80}',value):return value
        counts['non_label_string']+=1;return DROP
    counts['unsupported_type']+=1;return DROP


def write(path,data):
    path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    with path.open('xb') as out:out.write(data)
    path.chmod(0o600)


def json_bytes(value):return (json.dumps(value,ensure_ascii=True,indent=2,allow_nan=False)+'\n').encode()


def export(args):
    repo=(FREETOKEN if WORKSPACE == CHECKOUT else WORKSPACE/'work/FreeToken-opt').resolve()
    if git(repo,'rev-parse',BASE+'^{commit}').decode().strip()!=BASE:raise ValueError('Pinned base is unavailable')
    head=git(repo,'rev-parse','HEAD').decode().strip()
    if head!=BASE:raise ValueError('HEAD moved from pinned base; review before exporting this working-tree release')
    tracked=set(filter(None,git(repo,'diff','--no-ext-diff','--no-textconv','--no-renames','--name-only','-z',BASE,'--').decode().split('\0')))
    extra=set(filter(None,git(repo,'ls-files','--others','--exclude-standard','-z').decode().split('\0')))
    names=sorted(name for name in tracked|extra if allowed_source(name))
    rejected_tracked=tracked-set(names)
    if rejected_tracked:raise ValueError('Tracked changes outside declared runtime/test source scope')
    if not names:raise ValueError('No source changes to export')
    snapshots={};parts=[];files=[]
    for name in names:
        relative(name)
        path=within(repo,name)
        data=source_bytes(path) if path.exists() else None
        snapshots[name]=None if data is None else digest(data)
        groups=feature_groups(name)
        if not groups:raise ValueError('Changed source lacks an explicit feature mapping: '+name)
        if name in tracked:
            patch=git(repo,'diff','--no-ext-diff','--no-textconv','--no-renames','--no-color','--full-index',
                      '--src-prefix=a/','--dst-prefix=b/',BASE,'--',name)
        else:
            if data is None:raise ValueError('Untracked source disappeared')
            patch=new_file_patch(name,data)
        if b'GIT binary patch' in patch or b'Binary files ' in patch:raise ValueError('Unexpected binary source patch')
        # A removed line is still part of a diff. Check the patch too: inspecting
        # only the current file could publish a credential deleted from the base.
        checked_source_data(patch)
        parts.append(patch)
        files.append({'path':name,'change':'new' if name in extra else 'modified_or_deleted',
                      'sha256':snapshots[name],'groups':groups,'shared_integration':len(groups)>1,
                      'switch_names_observed':observed_switches(data) if data is not None else []})
    scope=PUBLIC/'evidence-exports'
    scope.mkdir(mode=0o700,exist_ok=True)
    name=args.name or 'release-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,80}',name):raise ValueError('Invalid export name')
    target=scope/name
    if target.exists():raise ValueError('Export destination already exists')
    staging=scope/(name+'.preparing-'+uuid.uuid4().hex[:8])
    staging.mkdir(mode=0o700)
    # A failed export is kept as .preparing for inspection, never overwrites a
    # previous export, and never deletes source or another agent's work.
    integration=b''.join(parts)
    write(staging/'runtime/integration.patch',integration)
    git(repo,'apply','--check','--reverse',str((staging/'runtime/integration.patch').resolve()))
    feature_receipts=[]
    for key,group in GROUPS.items():
        selected=[entry for entry in files if key in entry['groups']]
        manifest={'group':key,'name':group['name'],'patch':'../integration.patch',
                  'independently_applicable':False,'files':selected,'switches':group['switches'],
                  'disable_note':group['disable_note'],'qualification':'not_decided_by_exporter'}
        write(staging/f'runtime/features/{key}.json',json_bytes(manifest))
        feature_receipts.append({'group':key,'file_count':len(selected),'shared_count':sum(row['shared_integration'] for row in selected)})
    overlay=[]
    for name in OVERLAY:
        path=within(SERVICE,name)
        if not path.is_file():raise ValueError('Required service overlay source is missing: '+name)
        data=source_bytes(path);write(staging/'service-overlay'/name,data)
        overlay.append({'path':name,'sha256':digest(data)})
    launcher=source_bytes(PUBLIC/'qwen36.cmd')
    write(staging/'service-overlay/bin/qwen36.cmd',launcher)
    overlay.append({'path':'bin/qwen36.cmd','sha256':digest(launcher)})
    evidence=[];missing=[]
    selected=list(dict.fromkeys(([] if args.no_default_results else list(DEFAULT_RESULTS))+args.result))
    native=pathlib.Path(args.native_root).resolve()
    roots={'public':PUBLIC,'service':SERVICE,'native':native}
    for selector in selected:
        label,sep,rel=selector.partition(':')
        if not sep or label not in roots:raise ValueError('Use public:, service: or native: relative JSON selectors')
        part=relative(rel)
        if part.suffix!='.json' or any(value.lower() in ('secrets','logs','__pycache__') or 'raw' in value.lower() for value in part.parts):
            raise ValueError('Forbidden evidence selector')
        if label=='service' and rel!='profiles.json':raise ValueError('Service evidence permits profiles.json only')
        if label=='native' and rel!='active-run.json' and not rel.startswith('reports/'):
            raise ValueError('Native evidence permits selected reports/*.json or active-run.json only')
        path=within(roots[label],rel)
        if not path.is_file():
            if selector in args.result:raise ValueError('Explicit selected evidence is missing: '+selector)
            missing.append(selector);continue
        if path.stat().st_size>JSON_LIMIT:raise ValueError('Selected JSON exceeds evidence size limit')
        raw=path.read_bytes()
        if len(raw)>JSON_LIMIT:raise ValueError('Selected JSON exceeds evidence size limit')
        value=json.loads(raw.decode('utf-8-sig'))
        counts=collections.Counter();clean=scrub(value,counts)
        if clean is DROP:raise ValueError('Selected evidence has no exportable structure')
        dest='evidence/'+label+'/'+rel
        payload=json_bytes(clean);write(staging/dest,payload)
        scope_class='synthetic_component_not_model_speed' if rel.startswith('cuda-') else 'qualification_scope_not_inferred_by_exporter'
        if part.name in ('backend-review-results.json','engine-metrics-tests.json','ops-control-tests.json','ops-readiness-tests.json','acceptance-cpu-review-results.json'):
            scope_class='cpu_mock_or_static_validation_not_live_service'
        evidence.append({'selector':selector,'path':dest,'original_sha256':digest(raw),'export_sha256':digest(payload),
                         'original_bytes':len(raw),'export_bytes':len(payload),'sanitization_counts':dict(counts)})
        evidence[-1]['evidence_scope']=scope_class
    # Detect source or overlay edits made during export; do not publish a mixed snapshot.
    for name,expected in snapshots.items():
        path=within(repo,name)
        if (digest(source_bytes(path)) if path.exists() else None)!=expected:raise ValueError('Source changed during export')
    current_tracked=set(filter(None,git(repo,'diff','--no-ext-diff','--no-textconv','--no-renames','--name-only','-z',BASE,'--').decode().split('\0')))
    current_extra=set(filter(None,git(repo,'ls-files','--others','--exclude-standard','-z').decode().split('\0')))
    if sorted(name for name in current_tracked|current_extra if allowed_source(name))!=names or git(repo,'rev-parse','HEAD').decode().strip()!=BASE:
        raise ValueError('Source inventory or HEAD changed during export')
    for row in overlay:
        path=PUBLIC/'qwen36.cmd' if row['path']=='bin/qwen36.cmd' else within(SERVICE,row['path'])
        if digest(source_bytes(path))!=row['sha256']:raise ValueError('Overlay changed during export')
    manifest={'schema':1,'base_commit':BASE,'head_commit':head,'checkpoint':CHECKPOINT,
              'checkpoint_revision':CHECKPOINT_REVISION,'donor_commit':'a0116b2782a5b82622358c04a06496485d12b2d6',
              'patch_sha256':digest(integration),'patch_reverse_check':'PASS_against_current_worktree',
              'patch_format':'single_shared_integration_plus_feature_file_switch_manifests',
              'runtime_files':files,'feature_groups':feature_receipts,'service_overlay':overlay,
              'service_overlay_relationship':'separate_file_bundle_not_an_upstream_git_patch',
              'evidence':evidence,'missing_optional_evidence':missing,
              'excluded_untracked_files':len(extra-set(names)),
              'qualification':'not_evaluated_or_modified_by_exporter','qualified_flags_changed':False,
              'raw_logs_profiles_binaries_secrets_included':False,'json_sanitization':'lossy_numeric_and_controlled_labels',
              'source_privacy':'exact_source_bytes_may_contain_owner_deployment_paths_or_public_hostnames; review before external publication',
              'runtime_actions_performed':False}
    write(staging/'manifest.json',json_bytes(manifest))
    write(staging/'README.md',(
        '# Local release evidence bundle\n\n'
        'Runtime changes are one integration.patch against '+BASE+'. Feature manifests A-E explain shared files and opt-out switches; they are not independent patches.\n\n'
        'Apply only in a separate clean checkout of the pinned base after `git -c core.autocrlf=false apply --check runtime/integration.patch`; use the same core.autocrlf=false for applying. No commit, install or qualification is implied.\n\n'
        'service-overlay contains separate deployment-specific scripts, exact bytes including configured paths. It is not installed by this export. Evidence JSON is lossy and anonymized; inspect manifest sanitization counts. Source scripts need owner review before external publication.\n\n'
        'No weights, credentials, raw logs, binary libraries or Nsight raw profiles are included. Library hashes in JSON do not ship the libraries. Qualified flags are copied unchanged where present; this exporter never sets them.\n'
    ).encode())
    staging.rename(target)
    return {'export':str(target.relative_to(WORKSPACE)),'runtime_files':len(files),'overlay_files':len(overlay),
            'evidence_files':len(evidence),'qualification':'NOT_EVALUATED','runtime_actions':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',help='New subdirectory below outputs/qwen36-api/optimization/evidence-exports')
    parser.add_argument('--native-root',default='/home/USER/qwen36-api/optimization')
    parser.add_argument('--result',action='append',default=[],help='Exact selector public:relative.json, native:reports/relative.json, or native:active-run.json')
    parser.add_argument('--no-default-results',action='store_true')
    args=parser.parse_args()
    print(json.dumps(export(args),indent=2))


if __name__=='__main__':main()
