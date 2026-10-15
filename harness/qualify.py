"""Serial 128K teacher/logit and functional gates for an explicit candidate."""
import argparse,json,os,pathlib,subprocess
from matrix import run,PY
from bench import ROOT

def start(profile):
    run(['systemctl','--user','stop','qwen36-experiment'],60)
    run(['bash',str(ROOT/'start-experiment.sh'),profile],60)
    run([PY,str(ROOT/'check-experiment.py'),'300'],310)

def preflight(a, outputs):
    """CPU-only, before any unit starts: a run that cannot produce a valid comparison costs no GPU pass."""
    from freetoken.quality_evidence import bind_teacher, evidence_records, functional_rows, strict_json
    reports=ROOT/'reports'
    taken=[name for name in outputs if (reports/f'{name}.json').exists()]
    if taken:raise SystemExit('report names already exist (pass a new --name): '+', '.join(taken))
    try:
        if not a.baseline_teacher:
            report=strict_json((reports/f'{a.reference_report}.json').read_text())
            bind_teacher(report,evidence_records(reports/'teacher.jsonl',[report]),'teacher-bf128')
        functional_rows(strict_json((reports/f'{a.functional_baseline}.json').read_text()))
    except (OSError,ValueError,TypeError,KeyError) as error:
        raise SystemExit(f'reference or functional baseline is not run-bound schema-2 evidence: {error}')

def main():
    # Historical f1 reports (teacher-bf128-logits, functional-bf128) are not schema-2 evidence: no defaults.
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--baseline-teacher',action='store_true');p.add_argument('--reference-report',required=True);p.add_argument('--functional-baseline',required=True);p.add_argument('--name',help='unique report prefix; profile names stay unchanged')
    a=p.parse_args()
    teacher_report=(a.name+'-teacher') if a.name else 'teacher-'+a.candidate
    functional_report=(a.name+'-functional') if a.name else 'functional-'+a.candidate
    comparison_report='quality-'+(a.name or a.candidate)
    preflight(a,[teacher_report,functional_report,comparison_report]+([a.reference_report] if a.baseline_teacher else []))
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
