"""Owner-local, hash-conditional release/rollback. Never touches keys or Caddy.

Install is gated by the staged ``profile-runtime.py`` validator: exact-bool
qualified profiles, a schema-1 qualification receipt whose configuration and
source hashes match the staged bytes, and a strict staging resolver (no
fallback to installed files). Only ``release/lib/libqwen_attention_gate.so`` (fused gate),
``release/lib/libqwen_nvfp4_bitexact.so`` (bit-exact CUDA NVFP4 prefill), ``release/lib/libqwen36_gdn.so``
(bit-exact CUDA GDN) and ``release/lib/libqwen_fp8_gemv.so`` (bit-exact FP8 decode GEMV) may be shipped,
each only when a qualified profile enables it (the bit-exact ones only with receipt evidence).
The staged runtime (source tree + the compiled FreeToken extensions) must be
byte-identical to the native tree the qualification runs imported; the venv
runtime environment and the checkpoint directory must match the receipt/pin, and
a saved profile selection must stay startable.
Kernel provenance is fail-closed: kernel_provenance inventory()/check() over the staged kernel bytes
and the venv kernel cache must pass and equal the receipt; the ctypes libraries must match their
receipt sha256 and ABI probe. The inventory and library bindings are written to the rollback manifest.
"""
import argparse,datetime,hashlib,importlib.util,json,os,pathlib,re,shutil,subprocess,uuid

ROOT=pathlib.Path('/home/USER/qwen36-api')
from repo_paths import CHECKOUT, SERVING, FREETOKEN
PROJECT=CHECKOUT
OUTPUT=SERVING
RUNTIME_SOURCE=FREETOKEN/'python/freetoken'
# Qualification runs import this tree (run-experiment.sh: PYTHONPATH=$root/optimization/python):
# the source tree plus native-only build products that sync keeps there. Production imports
# release/python first (start-inference.sh PYTHONPATH), so the venv copies are never found.
QUALIFIED_RUNTIME=ROOT/'optimization/python/freetoken'
# Required by the offload path (kernel/pinned.py raises ImportError without _pinned_tensor).
NATIVE_EXTENSIONS=tuple('kernel/'+name+'.cpython-312-x86_64-linux-gnu.so' for name in ('_cpu_moe','_pinned_tensor','_ple_store'))
MODEL_DIR=None  # None: the staged validator's MODEL
# Same bytes the qualification runs loaded (private-libraries == build at review time).
GATE_LIBRARY_SOURCE=ROOT/'optimization/private-libraries/libqwen_attention_gate.so'
# Bit-exact CUDA NVFP4 prefill library the qualification runs loaded (QWEN36_CUDA_NVFP4_LIBRARY).
NVFP4_LIBRARY_SOURCE=ROOT/'optimization/private-libraries/libqwen_nvfp4_bitexact.so'
# Bit-exact CUDA GDN kernels the qualification runs load (QWEN36_CUDA_GDN_LIBRARY, ABI 2, pinned sha256 a8a806ed...).
GDN_LIBRARY_SOURCE=ROOT/'optimization/private-cuda-gdn/pinned/libqwen36_gdn-a8a806eda3a90a65.so'
# Bit-exact FP8 decode GEMV the qualification runs load (QWEN36_FP8_GEMV_LIBRARY); freetoken/kernel/
# cuda_fp8_gemv.py itself refuses any bytes but sha256 685c78cf...
FP8_GEMV_LIBRARY_SOURCE=ROOT/'optimization/private-fp8gemv/libqwen_fp8_gemv-685c78cf89e60d02.so'
# Bit-exact decode o-projection / NVFP4 MoE decode GEMV (kernels/decode-oproj, kernels/decode-moe); the
# FreeToken modules refuse any bytes but the sha256 they pin.
OPROJ_GEMV_LIBRARY_SOURCE=ROOT/'optimization/private-decode-kernels/libqwen_oproj_gemv-fa9302e31cb88a02.so'
MOE_DECODE_LIBRARY_SOURCE=ROOT/'optimization/private-decode-kernels/libqwen_moe_decode-83e90c43cb63ce11.so'
QUALIFICATION_SOURCE=OUTPUT/'qualification.json'
QUALIFICATION_TARGET='release/qualification.json'
# Receipt evidence paths are relative to this private root (e.g. 'reports/x.json');
# install re-verifies their hashes. Runtime start never reads private reports.
EVIDENCE_ROOT=ROOT/'optimization'
FILES=['guard.py','ingress.py','request_metrics.py','serve_guarded.py','readiness.py','placement.py','keepalive.py',
       'metrics_summary.py','profile-runtime.py','profiles.json','start-inference.sh']
HASH=re.compile(r'[0-9a-f]{64}')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

def valid_relative(relative):
    return (isinstance(relative,str) and bool(relative) and '\\' not in relative
            and not pathlib.PurePosixPath(relative).is_absolute()
            and all(part not in ('','.','..') for part in relative.split('/')))

def safe(relative):
    if not valid_relative(relative):raise RuntimeError('Invalid target path')
    path=ROOT/relative
    if not path.resolve().is_relative_to(ROOT.resolve()):raise RuntimeError('Target escapes service root')
    if path.is_symlink():raise RuntimeError('Refusing a symlink target')
    if path.resolve()!=ROOT.resolve().joinpath(relative):raise RuntimeError('Refusing a target below a symlink')
    return path

def stopped():
    for unit in ('qwen36-inference.service','qwen36-experiment.service'):
        result=subprocess.run(['systemctl','--user','show',unit,'--no-pager',
            '--property=LoadState','--property=ActiveState','--property=SubState'],
            capture_output=True,text=True,timeout=15)
        if result.returncode!=0:
            raise RuntimeError(f'Cannot verify {unit}; release refused')
        state={}
        for line in result.stdout.splitlines():
            key,sep,value=line.partition('=')
            if not sep or key in state:raise RuntimeError('Invalid systemctl state response')
            state[key]=value
        if set(state)!={'LoadState','ActiveState','SubState'}:
            raise RuntimeError('Incomplete systemctl state response')
        missing=(state['LoadState']=='not-found' and state['ActiveState']=='inactive' and state['SubState']=='dead')
        loaded=(state['LoadState'] in ('loaded','masked') and state['ActiveState'] in ('inactive','failed'))
        if not (missing or loaded):raise RuntimeError(f'{unit} is not verified stopped')

def atomic(src,dest):
    dest.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    temp=dest.with_name(dest.name+'.qwen36-new')
    with temp.open('xb') as out:out.write(src.read_bytes())
    temp.chmod(0o700 if dest.suffix=='.sh' else 0o600)
    temp.replace(dest)

def load_validator(path=None):
    """Import the staged validator that will itself be installed (hyphenated file name)."""
    path=OUTPUT/'profile-runtime.py' if path is None else pathlib.Path(path)
    spec=importlib.util.spec_from_file_location('qwen36_profile_runtime_staged',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if set(FILES)!=set(module.MANAGED_FILES):raise RuntimeError('Release FILES differ from validator MANAGED_FILES')
    return module

def staged_sources(validator,profiles):
    """Exact release mapping {production relative path: staged file}. Receipt excluded."""
    sources={name:OUTPUT/name for name in FILES}
    if not RUNTIME_SOURCE.is_dir() or RUNTIME_SOURCE.is_symlink():raise RuntimeError('Staged runtime tree is missing')
    try:runtime=validator.regular_files(RUNTIME_SOURCE,RUNTIME_SOURCE)
    except (OSError,ValueError) as error:raise RuntimeError('Staged runtime tree refused: '+type(error).__name__+': '+str(error)[:200]) from None
    for relative in sorted(runtime):
        sources[validator.RUNTIME_PREFIX+relative]=RUNTIME_SOURCE/relative
    for relative in NATIVE_EXTENSIONS:
        if relative not in runtime:sources[validator.RUNTIME_PREFIX+relative]=QUALIFIED_RUNTIME/relative
    verify_qualified_runtime(validator,sources)
    # Only a library some qualified profile switches on enters the release (and its inventory).
    libraries={validator.GATE_LIBRARY:GATE_LIBRARY_SOURCE,validator.NVFP4_LIBRARY:NVFP4_LIBRARY_SOURCE,
               validator.GDN_LIBRARY:GDN_LIBRARY_SOURCE,validator.FP8_GEMV_LIBRARY:FP8_GEMV_LIBRARY_SOURCE,
               validator.OPROJ_GEMV_LIBRARY:OPROJ_GEMV_LIBRARY_SOURCE,validator.MOE_DECODE_LIBRARY:MOE_DECODE_LIBRARY_SOURCE}
    for library in sorted(validator.required_libraries(profiles)):
        if library not in libraries:raise RuntimeError('Unexpected native library request')
        sources[library]=libraries[library]
    return sources

def verify_qualified_runtime(validator,sources):
    """Staged runtime == the qualified native tree: same file set, same bytes (no __pycache__/.pyc)."""
    if not QUALIFIED_RUNTIME.is_dir() or QUALIFIED_RUNTIME.is_symlink():raise RuntimeError('Qualified runtime tree is missing')
    try:qualified=validator.regular_files(QUALIFIED_RUNTIME,QUALIFIED_RUNTIME)
    except (OSError,ValueError) as error:raise RuntimeError('Qualified runtime tree refused: '+type(error).__name__+': '+str(error)[:200]) from None
    staged={name[len(validator.RUNTIME_PREFIX):]:path for name,path in sources.items() if name.startswith(validator.RUNTIME_PREFIX)}
    if set(staged)!=qualified:
        raise RuntimeError('Staged runtime file set differs from the qualified runtime tree: '+','.join(sorted(set(staged)^qualified)[:5]))
    for relative in sorted(staged):
        try:same=validator.sha256_regular_file(staged[relative])==validator.sha256_regular_file(QUALIFIED_RUNTIME/relative)
        except (OSError,ValueError) as error:raise RuntimeError('Runtime file unreadable: '+relative+': '+type(error).__name__) from None
        if not same:raise RuntimeError('Staged runtime bytes differ from the qualified runtime tree: '+relative)

def runtime_environment(validator):
    """Venv fingerprint (package versions/RECORDs, kernel cache bytes) of the service ROOT."""
    try:return validator.runtime_environment(ROOT/validator.VENV_SITE_PACKAGES)
    except (OSError,ValueError) as error:raise RuntimeError('Runtime environment refused: '+type(error).__name__+': '+str(error)[:200]) from None

def kernel_cache_dir(validator):
    return validator.kernel_cache_dir(ROOT/validator.VENV_SITE_PACKAGES)

def verify_provenance(validator,profiles,qualification,sources):
    """Kernel inventory of the staged bytes + venv cache, and the ctypes library ABI, equal the receipt."""
    resolve=strict_resolver(sources)
    try:kernel=validator.verify_kernel_provenance(profiles,qualification,resolve,kernel_cache_dir(validator))
    except (OSError,ValueError) as error:raise RuntimeError('Kernel provenance refused: '+type(error).__name__+': '+str(error)[:300]) from None
    try:libraries=validator.verify_native_libraries(profiles,qualification,resolve)
    except (OSError,ValueError) as error:raise RuntimeError('Native library refused: '+type(error).__name__+': '+str(error)[:300]) from None
    return {'kernel_provenance':kernel,'native_libraries':libraries}

def verify_checkpoint(validator):
    try:validator.verify_checkpoint(MODEL_DIR)
    except (OSError,ValueError) as error:raise RuntimeError('Checkpoint refused: '+type(error).__name__+': '+str(error)[:200]) from None

def start_profile(validator,profiles,qualified):
    """Profile a start without --profile runs after this release (config/selected-profile survives it)."""
    selected=safe(validator.SELECTED_PROFILE)
    if not os.path.lexists(selected):return profiles['default']
    try:name=validator._read_regular(selected).decode('ascii').strip()
    except (OSError,ValueError) as error:raise RuntimeError('Saved profile selection unreadable: '+type(error).__name__) from None
    if name not in qualified:
        raise RuntimeError('Saved profile selection '+validator.SELECTED_PROFILE+'='+repr(name[:20])+' is not qualified in this release; '
                           'remove it (start then uses default '+profiles['default']+') before installing')
    return name

def strict_resolver(sources):
    def resolve(relative):
        # Strictly the staged mapping: an unstaged key never falls back to ROOT.
        if relative not in sources:raise ValueError('Source is not staged for this release')
        return sources[relative]
    return resolve

def stale_installed(validator,sources):
    """Installed runtime/library files that the new release would not carry."""
    stale=[]
    python_dir=ROOT/'release/python'
    if python_dir.exists() or python_dir.is_symlink():
        if python_dir.is_symlink() or not python_dir.is_dir():raise RuntimeError('Invalid installed runtime directory')
        unexpected={entry.name for entry in python_dir.iterdir()}-{'freetoken','__pycache__'}
        if unexpected:raise RuntimeError('Unexpected installed entries next to the runtime package; inspect manually')
    for relative_dir in (validator.RUNTIME_TREE,validator.LIBRARY_DIR):
        top=ROOT/relative_dir
        if not (top.exists() or top.is_symlink()):continue
        if top.is_symlink() or not top.is_dir():raise RuntimeError('Invalid installed release directory')
        try:found=validator.regular_files(top,ROOT)
        except (OSError,ValueError) as error:raise RuntimeError('Installed release tree refused: '+type(error).__name__+': '+str(error)[:200]) from None
        stale.extend(sorted(name for name in found if name not in sources))
    return stale

def verify_evidence(validator,qualification):
    """Every referenced evidence file must still exist with the recorded hash (no symlinks)."""
    base=EVIDENCE_ROOT.resolve()
    for name in sorted(qualification['profiles']):
        for row in qualification['profiles'][name]['evidence']:
            path=base.joinpath(row['path'])
            if path.resolve()!=path:raise RuntimeError('Evidence path crosses a symlink: '+row['path'])
            try:actual=validator.sha256_regular_file(path)
            except (OSError,ValueError):raise RuntimeError('Qualification evidence unavailable: '+row['path']) from None
            if actual!=row['sha256']:raise RuntimeError('Qualification evidence changed: '+row['path'])

def preflight():
    validator=load_validator()
    try:
        profiles=validator.read_json(OUTPUT/'profiles.json')
        validator.validate_profiles_document(profiles)
    except (OSError,ValueError) as error:
        raise RuntimeError('Staged profiles refused: '+type(error).__name__+': '+str(error)[:200]) from None
    if not QUALIFICATION_SOURCE.is_file() or QUALIFICATION_SOURCE.is_symlink():
        raise RuntimeError('Release qualification receipt is missing; install refused')
    try:
        receipt_sha256=validator.sha256_regular_file(QUALIFICATION_SOURCE)
        qualification=validator.read_json(QUALIFICATION_SOURCE)
    except (OSError,ValueError) as error:
        raise RuntimeError('Release qualification receipt unreadable: '+type(error).__name__) from None
    sources=staged_sources(validator,profiles)
    environment=runtime_environment(validator)
    try:
        validator.validate_profiles(profiles,qualification,strict_resolver(sources),expected_sources=set(sources),
                                    environment=environment)
    except (OSError,ValueError) as error:
        raise RuntimeError('Release qualification refused: '+type(error).__name__+': '+str(error)[:200]) from None
    provenance=verify_provenance(validator,profiles,qualification,sources)
    verify_evidence(validator,qualification)
    verify_checkpoint(validator)
    sources[QUALIFICATION_TARGET]=QUALIFICATION_SOURCE
    for name in sources:safe(name)
    stale=stale_installed(validator,sources)
    for name in stale:safe(name)
    qualified=sorted(name for name,profile in profiles['profiles'].items() if profile['qualified'] is True)
    start=start_profile(validator,profiles,qualified)
    # Exact bytes the install must copy: receipt manifest plus the receipt itself.
    expected=dict(qualification['source_files'],**{QUALIFICATION_TARGET:receipt_sha256})
    return profiles,sources,stale,qualified,expected,start,provenance

def provenance_summary(provenance):
    kernel=provenance['kernel_provenance'];inventory=kernel['inventory']
    return {'kernel_inventory_sha256':kernel['inventory_sha256'],'required_extensions':kernel['required_extensions'],
            'required_modules':kernel['required_modules'],'kernel_modules':len(inventory['modules']),
            'kernel_module_statuses':sorted({row['status'] for row in inventory['modules'].values()}),
            'kernel_extensions':{name:row['status'] for name,row in sorted(inventory['extensions'].items())},
            'native_library_sha256':{name:row['sha256'] for name,row in sorted(provenance['native_libraries'].items())}}

def install(check):
    profiles,sources,stale,qualified,expected,start,provenance=preflight()
    stopped()
    libraries=sorted(name for name in sources if name.startswith('release/lib/'))
    summary={'files':len(sources),'default':profiles['default'],'qualified_profiles':qualified,
             'native_libraries':libraries,'stale_files_removed':len(stale),'start_profile':start,
             'provenance':provenance_summary(provenance)}
    if check:print(json.dumps({'check':'PASS',**summary}));return
    rows=[]
    for name,sourcefile in sources.items():
        row={'path':name,'before':digest(safe(name)),'after':digest(sourcefile)}
        if row['after']!=expected[name]:raise RuntimeError('Staged source changed after validation; release refused: '+name)
        rows.append(row)
    for name in stale:
        rows.append({'path':name,'before':digest(safe(name)),'after':None})
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (ROOT/'backups').mkdir(mode=0o700,exist_ok=True)
    backup=ROOT/'backups'/('release-'+stamp+'-'+uuid.uuid4().hex[:8]);backup.mkdir(mode=0o700)
    for row in rows:
        if row['before'] is None:continue
        saved=backup/'files'/row['path'];saved.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        shutil.copy2(safe(row['path']),saved);saved.chmod(0o600)
        if digest(saved)!=row['before']:raise RuntimeError('Backup copy differs; release refused before changes')
    receipt={'schema':1,'backup':str(backup),'files':rows,'state':'prepared','default':profiles['default'],
             'qualified_profiles':qualified,'native_libraries':libraries,
             'qualification_sha256':digest(sources[QUALIFICATION_TARGET]),
             # Full kernel inventory (statuses, .so/source sha256) and ctypes bindings of this release.
             'kernel_provenance':provenance['kernel_provenance'],'native_library_bindings':provenance['native_libraries']}
    manifest=backup/'manifest.json';manifest.write_text(json.dumps(receipt,indent=2));manifest.chmod(0o600)
    for row in rows:
        if row['after'] is None:safe(row['path']).unlink()
        elif row['before']!=row['after']:atomic(sources[row['path']],safe(row['path']))
    drift=[row['path'] for row in rows if digest(safe(row['path']))!=row['after']]
    if drift:
        receipt['state']='partial';manifest.write_text(json.dumps(receipt,indent=2))
        raise RuntimeError('Installed bytes differ from the validated release; use rollback manifest '+str(manifest)+': '+','.join(drift[:5]))
    receipt['state']='installed';manifest.write_text(json.dumps(receipt,indent=2))
    print(json.dumps({'installed_files':len(rows),'rollback_manifest':str(manifest),'auto_start':False,**summary}))

def validate_rollback(manifest):
    manifest=pathlib.Path(manifest).resolve()
    if not manifest.is_relative_to((ROOT/'backups').resolve()):raise RuntimeError('Manifest must be a protected service backup')
    receipt=json.loads(manifest.read_text());rows=receipt['files']
    if not isinstance(rows,list) or not rows:raise RuntimeError('Empty or invalid rollback manifest')
    seen=set();conflicts=[]
    for row in rows:
        name=row['path'];before=row['before'];after=row['after']
        if not valid_relative(name):
            raise RuntimeError('Invalid rollback path')
        if name in seen:raise RuntimeError('Duplicate rollback entry')
        seen.add(name)
        # after=None marks a stale file removed by install; it must have a backup.
        if (after is None and before is None) or (after is not None and (not isinstance(after,str) or not HASH.fullmatch(after))) \
           or (before is not None and (not isinstance(before,str) or not HASH.fullmatch(before))):
            raise RuntimeError('Invalid rollback hashes')
        actual=digest(safe(name))
        if actual not in (row['before'],row['after']):conflicts.append(row['path'])
        if before is not None:
            original=manifest.parent/'files'/name
            if not original.resolve().is_relative_to((manifest.parent/'files').resolve()) or original.is_symlink():
                raise RuntimeError('Backup escapes protected files')
            if digest(original)!=before:raise RuntimeError('Backup hash mismatch')
    if conflicts:raise RuntimeError('Later edits detected; no rollback performed: '+','.join(conflicts))
    return manifest,rows

def rollback(manifest,check):
    manifest,rows=validate_rollback(manifest)
    if check:print(json.dumps({'rollback_check':'PASS','files':len(rows)}));return
    stopped()
    # Validate the complete set again after the service-state check, before the
    # first unlink/replace. A corrupt later backup must never cause partial undo.
    manifest,rows=validate_rollback(manifest)
    for row in reversed(rows):
        target=safe(row['path'])
        if digest(target)==row['before']:continue
        if row['before'] is None:target.unlink()
        else:
            original=manifest.parent/'files'/row['path']
            if digest(original)!=row['before']:raise RuntimeError('Backup hash mismatch')
            atomic(original,target)
    print(json.dumps({'rollback':'PASS','service_started':False,'keys_weights_dns_caddy':'untouched'}))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['install','rollback']);p.add_argument('--check',action='store_true');p.add_argument('--manifest')
    a=p.parse_args()
    if a.action=='install':install(a.check)
    elif a.manifest:rollback(a.manifest,a.check)
    else:p.error('--manifest required for rollback')

if __name__=='__main__':main()
