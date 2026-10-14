"""Serial 128K teacher/logit and functional gates for an explicit candidate."""
import argparse,json,pathlib,subprocess
from matrix import run,PY
from bench import ROOT

def start(profile):
    run(['systemctl','--user','stop','qwen36-experiment'],60)
    run(['bash',str(ROOT/'start-experiment.sh'),profile],60)
    run([PY,str(ROOT/'check-experiment.py'),'300'],310)

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--baseline-teacher',action='store_true')
    a=p.parse_args()
    candidate=json.loads((ROOT/'config'/f'{a.candidate}.json').read_text())
    teacher='teacher-'+a.candidate
    candidate.setdefault('env',{}).update(QWEN36_TEACHER_FIXTURE=str(ROOT/'config/teacher-fixture.json'),
        QWEN36_LOGIT_REFERENCE=str(ROOT/'reports/logit-reference'))
    # Persist on the workspace side too: start-experiment refreshes profile JSONs.
    workspace=pathlib.Path('/path/to/qwen36-workspace/outputs/qwen36-api/optimization')
    (workspace/(teacher+'.json')).write_text(json.dumps(candidate))
    try:
        if a.baseline_teacher:
            print('Starting teacher-bf128',flush=True);start('teacher-bf128')
            print(run([PY,str(ROOT/'quality.py'),'--name','teacher-bf128-logits','--teacher']),flush=True)
        print('Starting '+teacher,flush=True);start(teacher)
        print(run([PY,str(ROOT/'quality.py'),'--name',teacher,'--teacher']),flush=True)
        print('Starting functional '+a.candidate,flush=True);start(a.candidate)
        print(run([PY,str(ROOT/'quality.py'),'--name','functional-'+a.candidate]),flush=True)
        print(run([PY,str(ROOT/'compare-quality.py'),'--candidate',teacher,
                   '--functional-candidate','functional-'+a.candidate,'--name','quality-'+a.candidate]),flush=True)
    finally:run(['systemctl','--user','stop','qwen36-experiment'],60)

if __name__=='__main__':main()
