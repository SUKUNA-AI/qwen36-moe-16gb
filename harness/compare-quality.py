import argparse,json,pathlib,math
import numpy as np
ROOT=pathlib.Path('/home/USER/qwen36-api/optimization')
parser=argparse.ArgumentParser()
parser.add_argument('--baseline',default='teacher-bf128');parser.add_argument('--candidate',default='teacher-fp128')
parser.add_argument('--functional-baseline',default='functional-bf128');parser.add_argument('--functional-candidate',default='functional-fp128')
parser.add_argument('--name',default='quality-comparison');args=parser.parse_args()
groups={};records={}
for line in (ROOT/'reports/teacher.jsonl').read_text().splitlines():
    r=json.loads(line);groups.setdefault(r['profile'],{})[r['case']]=r['nll']
    records.setdefault(r['profile'],{})[r['case']]=r
a=groups[args.baseline];b=groups[args.candidate]
assert a.keys()==b.keys() and len(a)==20
keys=sorted(a);rows=[]
for key in keys:
    assert len(a[key])==len(b[key]) and len(a[key])>0
    assert all(math.isfinite(value) and value>=0 for value in a[key]+b[key])
    rows.append({'case':key,'tokens':len(a[key]),'baseline_sum':sum(a[key]),'candidate_sum':sum(b[key]),
                 'mean_nll_delta':float(np.mean(b[key])-np.mean(a[key]))})
length=np.array([r['tokens'] for r in rows]);diff=np.array([r['candidate_sum']-r['baseline_sum'] for r in rows])
rng=np.random.default_rng(4172026);idx=rng.integers(0,len(rows),(20000,len(rows)))
bootstrap=np.exp(diff[idx].sum(1)/length[idx].sum(1))-1
report={'method':'paired cluster bootstrap over 20 fixed cases, 20000 resamples, equal token weights',
        'baseline':args.baseline,'candidate':args.candidate,
        'tokens':int(length.sum()),'baseline_perplexity':math.exp(sum(map(sum,a.values()))/length.sum()),
        'relative_perplexity_growth':math.exp(diff.sum()/length.sum())-1,
        'relative_growth_ci95':np.quantile(bootstrap,[.025,.975]).tolist(),'rows':rows}
report['ppl_gate_pass']=report['relative_growth_ci95'][1]<=.01
logits=[comparison for row in records[args.candidate].values() for comparison in row.get('logit_comparisons',[])]
expected_logits=sum((len(values)+7)//8 for values in b.values())
logits_ok=len(logits)==expected_logits and all(row['finite'] and math.isfinite(row['max_abs']) and math.isfinite(row['relative_l2']) for row in logits)
report['logit_comparisons']={'expected':expected_logits,'observed':len(logits),'finite_and_complete':logits_ok,
    'max_abs':max((row['max_abs'] for row in logits),default=None),
    'max_relative_l2':max((row['relative_l2'] for row in logits),default=None)}
teacher_report=json.loads((ROOT/'reports'/f'{args.candidate}.json').read_text())
teacher_rows=teacher_report['rows']
teacher_ok=len(teacher_rows)==20 and {r['case'] for r in teacher_rows}==set(keys) and all(
    r['status']==200 and r['forced_text_matches'] and r['usage']['completion_tokens']==r['expected_tokens'] for r in teacher_rows)
report['forced_targets_complete']=teacher_ok
fa=json.load(open(ROOT/'reports'/f'{args.functional_baseline}.json'));fb=json.load(open(ROOT/'reports'/f'{args.functional_candidate}.json'))
failures=[c['case'] for c in fb['rows'] if not c['pass'] and next(x['pass'] for x in fa['rows'] if x['case']==c['case'])]
report['new_functional_failures']=failures
report['quality_gate_pass']=report['ppl_gate_pass'] and logits_ok and teacher_ok and not failures
(ROOT/'reports'/f'{args.name}.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
