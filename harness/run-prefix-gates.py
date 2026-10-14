"""Serial clean-process prefix qualification; a functional FAIL is never erased."""
import argparse,json,subprocess
from bench import ROOT
from matrix import PY,run

p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--name',required=True)
p.add_argument('--report-prefix')
p.add_argument('--lengths',default='16384,60000');a=p.parse_args()
for unit in ('qwen36-inference','qwen36-experiment'):
    if subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode==0:
        raise RuntimeError('Refusing to take over an active unit')
rows=[];report_name=a.report_prefix or a.name
if '/' in report_name or '\\' in report_name:raise ValueError('Simple report prefix required')
for phase in ('fresh','cached'):
    try:
        run(['bash',str(ROOT/'start-experiment.sh'),a.profile],60)
        run([PY,str(ROOT/'check-experiment.py'),'300'],310)
        command=[PY,str(ROOT/'prefix-quality.py'),'--phase',phase,'--name',a.name,
                 '--lengths',a.lengths,'--report-prefix',str(ROOT/'reports'/report_name)]
        result=subprocess.run(command,capture_output=True,text=True,timeout=1800)
        row={'phase':phase,'exit_code':result.returncode};rows.append(row)
        print(json.dumps(row),flush=True)
        if result.returncode not in (0,1):raise RuntimeError('Prefix test did not complete')
    finally:run(['systemctl','--user','stop','qwen36-experiment'],60)
(ROOT/'reports'/(report_name+'-runner.json')).write_text(json.dumps(rows,indent=2))
