"""Informational comparison of two teacher-bf128 logit references (CPU only; NOT a gate).

Use: after the R1 sync (NVFP4 global scale in FP32), `qualify.py --baseline-teacher`
overwrites reports/logit-reference/{seed}-{offset}.npy in place (engine/qualification.py
saves when QWEN36_PROFILE == teacher-bf128). Copy the old directory first (e.g.
reports/logit-reference-pre-r1), then run this to quantify how much the numeric basis moved:
  logits  every {seed}-{offset}.npy present in both: max_abs, relative_l2 (max/mean),
          top-1 agreement, files only in one side
  NLL     teacher.jsonl rows of profile teacher-bf128 for two run_ids (default: the last two
          distinct run_ids in file order = old, new): per-case token count/sum NLL, max
          per-token |delta|, relative perplexity change exp(sum(new-old)/tokens)-1
Writes reports/NAME.json once (O_EXCL). Exit 0 written; 2 usage/refused; 1 input error.
"""
import argparse,json,math,os,pathlib,re,sys

ROOT=pathlib.Path('/home/USER/qwen36-api/optimization')
NAME_PATTERN=re.compile(r'[a-z0-9][a-z0-9.-]{0,80}')
FILE_PATTERN=re.compile(r'(\d+)-(\d+)\.npy')


def compare_logits(old_dir,new_dir):
    import numpy as np
    old={p.name for p in old_dir.iterdir() if FILE_PATTERN.fullmatch(p.name)}
    new={p.name for p in new_dir.iterdir() if FILE_PATTERN.fullmatch(p.name)}
    rows=[]
    for name in sorted(old&new):
        a=np.load(old_dir/name,allow_pickle=False).astype(np.float64);b=np.load(new_dir/name,allow_pickle=False).astype(np.float64)
        if a.shape!=b.shape:rows.append({'file':name,'shape_mismatch':True});continue
        error=b-a
        rows.append({'file':name,'max_abs':float(np.max(np.abs(error))),
                     'relative_l2':float(np.linalg.norm(error)/max(np.linalg.norm(a),1e-12)),
                     'top1_equal':bool(int(np.argmax(a))==int(np.argmax(b))),
                     'finite':bool(np.isfinite(a).all() and np.isfinite(b).all())})
    good=[r for r in rows if not r.get('shape_mismatch')]
    return {'compared':len(good),'shape_mismatch':len(rows)-len(good),'only_old':len(old-new),'only_new':len(new-old),
            'bit_identical':sum(r['max_abs']==0 for r in good),
            'max_abs':max((r['max_abs'] for r in good),default=None),
            'max_relative_l2':max((r['relative_l2'] for r in good),default=None),
            'mean_relative_l2':sum(r['relative_l2'] for r in good)/len(good) if good else None,
            'top1_agreement':sum(r['top1_equal'] for r in good)/len(good) if good else None,
            'all_finite':all(r['finite'] for r in good)}


def teacher_runs(path,profile='teacher-bf128'):
    from freetoken.quality_evidence import teacher_records
    runs={}
    for (row_profile,run,case),row in teacher_records(path).items():
        if row_profile == profile:
            runs.setdefault(run,{})[case] = row['nll']
    return runs


def compare_nll(path,old_run=None,new_run=None):
    runs=teacher_runs(path);order=list(runs)
    if old_run is None or new_run is None:
        if len(order)<2:raise ValueError('teacher.jsonl needs two teacher-bf128 runs')
        old_run,new_run=order[-2],order[-1]
    a,b=runs[old_run],runs[new_run]
    cases=sorted(set(a)&set(b));tokens=0;delta=0.;max_token=0.;rows=[]
    for case in cases:
        if len(a[case])!=len(b[case]):rows.append({'case':case,'length_mismatch':True});continue
        diffs=[y-x for x,y in zip(a[case],b[case])]
        tokens+=len(diffs);delta+=sum(diffs);max_token=max([max_token]+[abs(d) for d in diffs])
        rows.append({'case':case,'tokens':len(diffs),'old_sum':sum(a[case]),'new_sum':sum(b[case]),'delta_sum':sum(diffs)})
    return {'old_run_id':old_run,'new_run_id':new_run,'cases_compared':len(cases),'cases_only_old':len(set(a)-set(b)),
            'cases_only_new':len(set(b)-set(a)),'tokens':tokens,'max_token_abs_delta':max_token,
            'relative_perplexity_change':math.exp(delta/tokens)-1 if tokens else None,'rows':rows}


def main(argv=None,root=None):
    root=ROOT if root is None else root
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--old-dir',default=str(root/'reports/logit-reference-pre-r1'));p.add_argument('--new-dir',default=str(root/'reports/logit-reference'))
    p.add_argument('--teacher-jsonl',default=str(root/'reports/teacher.jsonl'))
    p.add_argument('--old-run-id');p.add_argument('--new-run-id')
    a=p.parse_args(argv)
    if not NAME_PATTERN.fullmatch(a.name):p.error('invalid --name')
    target=root/'reports'/f'{a.name}.json'
    if target.exists():print(json.dumps({'result':'refused','reason':'report_exists'}),file=sys.stderr);return 2
    try:
        report={'kind':'logit-reference-comparison','gate':False,'old_dir':a.old_dir,'new_dir':a.new_dir,
                'logits':compare_logits(pathlib.Path(a.old_dir),pathlib.Path(a.new_dir)),
                'nll':compare_nll(pathlib.Path(a.teacher_jsonl),a.old_run_id,a.new_run_id)}
    except (OSError,ValueError,KeyError) as error:
        print(json.dumps({'result':'error','error_type':type(error).__name__,'message':str(error)[:200]}),file=sys.stderr);return 1
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as out:out.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'report':target.name,'logits':report['logits'],
                      'nll':{k:v for k,v in report['nll'].items() if k!='rows'}}))
    return 0


if __name__=='__main__':raise SystemExit(main())
