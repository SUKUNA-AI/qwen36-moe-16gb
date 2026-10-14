"""Run as root on edge, after authenticated backend smoke passes."""
import datetime,hashlib,json,os,pathlib,shutil,subprocess
root=pathlib.Path('/var/lib/qwen36-api')
source=pathlib.Path('/etc/caddy/Caddyfile')
site=pathlib.Path('/etc/caddy/qwen36-api.caddy')
marker='\n# BEGIN qwen36-api\nimport /etc/caddy/qwen36-api.caddy\n# END qwen36-api\n'
text=source.read_text()
if marker in text:
    print('ALREADY_ACTIVE');raise SystemExit(0)
assert 'llm.example.com' not in text and 'qwen36-api.caddy' not in text
before=hashlib.sha256(source.read_bytes()).hexdigest()
candidate=pathlib.Path('/etc/caddy/qwen36-api-candidate.Caddyfile')
candidate.write_text(text+marker);candidate.chmod(0o644)
subprocess.run(['sudo','-u','caddy','/usr/bin/caddy','validate','--config',str(candidate),'--adapter','caddyfile'],check=True)
assert hashlib.sha256(source.read_bytes()).hexdigest()==before,'Caddyfile changed during validation'
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup=root/'backups'/('Caddyfile.activate.'+stamp)
shutil.copy2(source,backup);backup.chmod(0o600)
st=source.stat();os.chown(candidate,st.st_uid,st.st_gid);candidate.chmod(st.st_mode&0o777)
os.replace(candidate,source)
subprocess.run(['systemctl','reload','caddy'],check=True)
receipt={'activated_utc':stamp,'before_sha256':before,'site_sha256':hashlib.sha256(site.read_bytes()).hexdigest(),'backup':str(backup),'marker':marker}
(root/'activation.json').write_text(json.dumps(receipt,indent=2));(root/'activation.json').chmod(0o600)
print(json.dumps({k:v for k,v in receipt.items() if k!='marker'}))
