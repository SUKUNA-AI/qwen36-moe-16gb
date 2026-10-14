"""Compare the actual private key to owned artifacts/logs without printing it."""
import base64,json,os,pathlib,subprocess
root=pathlib.Path('/home/USER/qwen36-api');out=pathlib.Path(__file__).resolve().parent
secret=root/'secrets/api.env';key=secret.read_text().split('=',1)[1].strip()
assert len(base64.urlsafe_b64decode(key+'='*((-len(key))%4)))>=32
assert secret.stat().st_uid==os.getuid() and secret.stat().st_mode&0o777==0o600
assert secret.parent.stat().st_mode&0o777==0o700
hits=[];checked=0
for directory in [out,out.parent.parent/'work',root/'logs',pathlib.Path('/path/to/LLM/qwen36-api/logs')]:
    for path in directory.iterdir():
        if path.is_file() and path.stat().st_size<20*1024*1024:
            checked+=1
            if key.encode() in path.read_bytes():hits.append(str(path))
journal=subprocess.check_output(['journalctl','--user','-u','qwen36-inference.service','--no-pager','-o','cat'],text=True)
assert key not in journal,'Private key found in native journal'
assert not hits,'Private key found in owned output/log file; inspect privately'
permissions={p.name:oct(p.stat().st_mode&0o777) for p in (root/'logs').iterdir() if p.is_file()}
assert all(v=='0o600' for v in permissions.values()),permissions
row={'result':'PASS','files_checked':checked,'actual_api_key_found':False,'native_journal_key_found':False,
     'secret_file_mode':'0600','secret_directory_mode':'0700','owner_matches':True,
     'entropy_bytes':32,'numeric_log_modes':permissions,
     'scope':'Owned deliverables, immediate work files, Linux metrics logs, Windows supervisor logs, unit journal; not a whole-machine credential audit'}
(out/'secret-hygiene.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
print(json.dumps(row))
