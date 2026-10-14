"""Remove only the owned import/site; retain concurrent edits and protected backups."""
import hashlib,json,os,pathlib,subprocess,sys,tempfile
root=pathlib.Path('/var/lib/qwen36-api');receipt=root/'activation.json'
if not receipt.exists():print('NO_ACTIVATION_TO_ROLL_BACK');raise SystemExit(0)
meta=json.loads(receipt.read_text());source=pathlib.Path('/etc/caddy/Caddyfile')
text=source.read_text();marker=meta['marker']
if marker not in text:print('OWNED_IMPORT_ALREADY_ABSENT');raise SystemExit(0)
assert text.count(marker)==1,'Ambiguous import block; inspect manually'
site=pathlib.Path('/etc/caddy/qwen36-api.caddy')
assert hashlib.sha256(site.read_bytes()).hexdigest()==meta['site_sha256'],'Site changed since deployment; preserve it and inspect manually'
fd,name=tempfile.mkstemp(prefix='qwen36-api-rollback-',suffix='.Caddyfile',dir='/etc/caddy');os.close(fd)
candidate=pathlib.Path(name);candidate.write_text(text.replace(marker,'',1))
candidate.chmod(0o644)
subprocess.run(['sudo','-u','caddy','/usr/bin/caddy','validate','--config',str(candidate),'--adapter','caddyfile'],check=True)
assert source.read_text()==text,'Concurrent Caddyfile change during validation'
if '--check' in sys.argv:
    candidate.unlink();print('OWNED_CADDY_REMOVAL_VALIDATED_NO_CHANGE');raise SystemExit(0)
st=source.stat();os.chown(candidate,st.st_uid,st.st_gid);candidate.chmod(st.st_mode&0o777)
os.replace(candidate,source);subprocess.run(['systemctl','reload','caddy'],check=True)
site.unlink();receipt.rename(root/'rolled-back-activation.json');print('OWNED_CADDY_IMPORT_REMOVED_OTHER_SITES_RETAINED')
