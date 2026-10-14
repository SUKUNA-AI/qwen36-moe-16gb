"""Serial 128K teacher/logit and functional gates for an explicit candidate."""
import argparse,json,os,pathlib,subprocess
from matrix import run,PY
from bench import ROOT

def start(profile):
    run(['systemctl','--user','stop','qwen36-experiment'],60)
    run(['bash',str(ROOT/'start-experiment.sh'),profile],60)
    run([PY,str(ROOT/'check-experiment.py'),'300'],310)

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--baseline-teacher',action='store_true');p.add_argument('--reference-report',default='teacher-bf128-logits');p.add_argument('--functional-baseline',default='functional-bf128');p.add_argument('--name',help='unique report prefix; profile names stay unchanged')
    a=p.parse_args()
    teacher_report=(a.name+'-teacher') if a.name else 'teacher-'+a.candidate
    functional_report=(a.name+'-functional') if a.name else 'functional-'+a.candidate
    comparison_report='quality-'+(a.name or a.candidate)
    candidate=json.loads((ROOT/'config'/f'{a.candidate}.json').read_text())
    teacher='teacher-'+a.candidate
    candidate.setdefault('env',{}).update(QWEN36_TEACHER_FIXTURE=str(ROOT/'config/teacher-fixture.json'),
        QWEN36_LOGIT_REFERENCE=str(ROOT/'reports/logit-reference'),
        QWEN36_TEACHER_REFERENCE_REPORT=str(ROOT/'reports'/(a.reference_report+'.json')))
    # Persist on the workspace side too: start-experiment refreshes profile JSONs.
    workspace=pathlib.Path('/path/to/qwen36-workspace/outputs/qwen36-api/optimization')
    workspace=pathlib.Path(os.environ.get('QWEN36_HARNESS_SOURCE',workspace))
    (workspace/(teacher+'.json')).write_text(json.dumps(candidate))
    try:
        if a.baseline_teacher:
            print('Starting teacher-bf128',flush=True);start('teacher-bf128')
            print(run([PY,str(ROOT/'quality.py'),'--name',a.reference_report,'--teacher']),flush=True)
        print('Starting '+teacher,flush=True);start(teacher)
        print(run([PY,str(ROOT/'quality.py'),'--name',teacher_report,'--teacher']),flush=True)
        print('Starting functional '+a.candidate,flush=True);start(a.candidate)
        print(run([PY,str(ROOT/'quality.py'),'--name',functional_report]),flush=True)
        print(run([PY,str(ROOT/'compare-quality.py'),'--candidate',teacher,
                   '--functional-candidate',functional_report,'--candidate-report',teacher_report,'--baseline-report',a.reference_report,
                   '--functional-baseline',a.functional_baseline,'--name',comparison_report]),flush=True)
    finally:run(['systemctl','--user','stop','qwen36-experiment'],60)

if __name__=='__main__':main()
