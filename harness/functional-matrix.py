"""Keep the entire frozen functional history when isolating the failed case."""
import argparse,json
from matrix import run,PY
from bench import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--profiles',required=True);p.add_argument('--name',required=True)
    a=p.parse_args();rows=[]
    try:
        for profile in a.profiles.split(','):
            run(['systemctl','--user','stop','qwen36-experiment'],60)
            run(['bash',str(ROOT/'start-experiment.sh'),profile],60)
            run([PY,str(ROOT/'check-experiment.py'),'300'],310)
            name=a.name+'-'+profile
            print(run([PY,str(ROOT/'quality.py'),'--name',name]),flush=True)
            rows.append({'profile':profile,'report':name+'.json'})
            (ROOT/'reports'/(a.name+'-matrix.json')).write_text(json.dumps(rows,indent=2))
    finally:run(['systemctl','--user','stop','qwen36-experiment'],60)

if __name__=='__main__':main()
