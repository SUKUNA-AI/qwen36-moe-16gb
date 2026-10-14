"""Ablate kernel and chunk changes on the already-failed frozen diagnostic case."""
import json
from matrix import run,PY
from bench import ROOT

rows=[]
try:
    for profile in ('bf128','chunk4096','cuda2048','cuda4096-safe'):
        run(['systemctl','--user','stop','qwen36-experiment'],60)
        run(['bash',str(ROOT/'start-experiment.sh'),profile],60)
        run([PY,str(ROOT/'check-experiment.py'),'300'],310)
        output=run([PY,str(ROOT/'diagnose-facts.py'),'--name','facts-'+profile,'--repeats','2'])
        print(output,flush=True);rows.append({'profile':profile,'report':'facts-'+profile+'.json'})
        (ROOT/'reports/facts-isolation.json').write_text(json.dumps(rows,indent=2))
finally:run(['systemctl','--user','stop','qwen36-experiment'],60)
