"""Download one pinned public upstream checkpoint; resumable, hash checked.
No access token is required. Does not print signed redirect URLs.
"""
import concurrent.futures, hashlib, json, pathlib, time, urllib.request

REPO = 'nvidia/Qwen3.6-35B-A3B-NVFP4'
REVISION = '1355db6a052410cfd62085d94b58866fd0f2c3c5'
ROOT = pathlib.Path('C:/path/to/LLM/qwen36-api/model')
ROOT.mkdir(parents=True, exist_ok=True)
FILES = [
    'README.md', 'chat_template.jinja', 'config.json', 'configuration.json',
    'generation_config.json', 'hf_quant_config.json', 'model.safetensors.index.json',
    'tokenizer.json', 'tokenizer_config.json', 'vocab.json',
    'model-00001-of-00003.safetensors', 'model-00002-of-00003.safetensors',
    'model-00003-of-00003.safetensors',
]
with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/tree/{REVISION}', timeout=30) as r:
    metadata = {f['path']: f for f in json.load(r)}
manifest = {'repo': REPO, 'revision': REVISION, 'files': [metadata[n] for n in FILES]}
(ROOT/'checkpoint-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(16*1024*1024): h.update(chunk)
    return h.hexdigest()

def download(name):
    dest=ROOT/name
    partial=ROOT/(name+'.partial')
    m=metadata[name]
    size=m['size']
    expected=m.get('lfs',{}).get('oid')
    if dest.exists() and dest.stat().st_size==size and (not expected or digest(dest)==expected):
        return {'file':name,'bytes':size,'sha256':digest(dest),'status':'verified_existing'}
    url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
    for attempt in range(5):
        offset=partial.stat().st_size if partial.exists() else 0
        if offset==size: break
        request=urllib.request.Request(url, headers={'Range':f'bytes={offset}-'} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=90) as r:
                if offset and (r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                    raise RuntimeError('Server did not honor resume range; keeping partial file')
                mode='ab' if offset and r.status==206 else 'wb'
                print(json.dumps({'file':name,'resume_offset':offset,'http_status':r.status}),flush=True)
                with partial.open(mode) as f:
                    while chunk:=r.read(4*1024*1024): f.write(chunk)
            if partial.stat().st_size==size: break
        except Exception as e:
            print(json.dumps({'file':name,'attempt':attempt+1,'error_type':type(e).__name__}),flush=True)
            time.sleep(min(2**attempt,16))
    if not partial.exists() or partial.stat().st_size!=size:
        raise RuntimeError(f'Incomplete download: {name}')
    actual=digest(partial)
    if expected and actual!=expected: raise RuntimeError(f'Hash mismatch: {name}')
    partial.replace(dest)
    result={'file':name,'bytes':size,'sha256':actual,'status':'downloaded_verified'}
    print(json.dumps(result),flush=True)
    return result

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    receipts=list(pool.map(download,FILES))
(ROOT/'download-receipt.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
print('CHECKPOINT_DOWNLOAD_COMPLETE',flush=True)
