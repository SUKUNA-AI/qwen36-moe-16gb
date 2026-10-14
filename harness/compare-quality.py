"""Compare explicit run-bound teacher and complete functional evidence (CPU only)."""
import argparse
import json
import math
import os
from pathlib import Path
import numpy as np
from freetoken.quality_evidence import (EvidenceError, bind_teacher, functional_rows, require,
                                       strict_json, evidence_records, digest)

ROOT = Path(os.environ.get('QWEN36_DATA_ROOT', Path(__file__).resolve().parent))

def compare(baseline, candidate, functional_baseline, functional_candidate, records, baseline_profile, candidate_profile):
    a = bind_teacher(baseline, records, baseline_profile)
    b = bind_teacher(candidate, records, candidate_profile)
    require(baseline['fixture_sha256'] == candidate['fixture_sha256'] and baseline['targets'] == candidate['targets'],
            'baseline/candidate fixture differs')
    for case in a:
        require(a[case]['reference'] == b[case]['reference'], f'{case}: reference differs')
        ref = a[case]['reference']
        require(ref.get('profile') == baseline_profile and ref.get('run_id') == a[case]['run_id'] and
                ref.get('case') == case and ref.get('fixture_sha256') == baseline['fixture_sha256'] and
                ref.get('targets_sha256') == a[case]['targets_sha256'] and type(ref.get('files')) is dict and
                set(ref['files']) == {str(p) for p in range(0, len(a[case]['nll']), 8)} and
                all(type(v) is str and len(v) == 64 for v in ref['files'].values()), f'{case}: unbound reference')
    # Config/source/library differences are legitimate; each arm is bound to its own receipt.
    for key in ('checkpoint', 'checkpoint_revision'):
        require(baseline['run'][key] == candidate['run'][key], f'reference {key} differs')
    fa, fb = functional_rows(functional_baseline), functional_rows(functional_candidate)
    failures = []
    for case, row in fb.items():
        baseline_case = case if case in fa else f'long-facts-{functional_baseline["functional_matrix"]["long"]}'
        if not row['pass'] and fa[baseline_case]['pass']:
            failures.append(case)
    rows = [{'case':k, 'tokens':len(a[k]['nll']), 'baseline_sum':sum(a[k]['nll']), 'candidate_sum':sum(b[k]['nll']),
             'mean_nll_delta':float(np.mean(b[k]['nll']) - np.mean(a[k]['nll']))} for k in sorted(a)]
    length = np.array([r['tokens'] for r in rows]); diff = np.array([r['candidate_sum']-r['baseline_sum'] for r in rows])
    rng = np.random.default_rng(4172026); idx = rng.integers(0, len(rows), (20000, len(rows)))
    bootstrap = np.exp(diff[idx].sum(1)/length[idx].sum(1))-1
    logits = [c for row in b.values() for c in row['logit_comparisons']]
    report = {'schema':2, 'evidence_valid':True, 'method':'paired cluster bootstrap over 20 fixed cases, 20000 resamples, equal token weights',
              'baseline':baseline_profile, 'candidate':candidate_profile, 'tokens':int(length.sum()),
              'baseline_perplexity':math.exp(sum(r['baseline_sum'] for r in rows)/length.sum()),
              'relative_perplexity_growth':math.exp(diff.sum()/length.sum())-1,
              'relative_growth_ci95':np.quantile(bootstrap,[.025,.975]).tolist(), 'rows':rows,
              'fixture_sha256':baseline['fixture_sha256'], 'case_runs':{
                  'baseline':{k:v['run_id'] for k,v in a.items()}, 'candidate':{k:v['run_id'] for k,v in b.items()}},
              'reports_sha256':{'baseline':digest(baseline), 'candidate':digest(candidate),
                                'functional_baseline':digest(functional_baseline), 'functional_candidate':digest(functional_candidate)}}
    report['ppl_gate_pass'] = bool(report['relative_growth_ci95'][1] <= .01)
    report['logit_comparisons'] = {'expected':sum((len(r['nll'])+7)//8 for r in b.values()), 'observed':len(logits),
                                   'finite_and_complete':True, 'all_byte_equal':all(c['byte_equal'] for c in logits), 'max_abs':max(c['max_abs'] for c in logits),
                                   'max_relative_l2':max(c['relative_l2'] for c in logits)}
    report['forced_targets_complete'] = True
    report['new_functional_failures'] = failures
    report['quality_gate_pass'] = report['ppl_gate_pass'] and not failures
    report['bit_exact_pass'] = (report['quality_gate_pass'] and all(c['byte_equal'] and c['max_abs'] == 0 and c['relative_l2'] == 0 for c in logits))
    return report

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--baseline', default='teacher-bf128'); p.add_argument('--candidate', default='teacher-fp128')
    p.add_argument('--baseline-report', default='teacher-bf128-logits')
    p.add_argument('--candidate-report')
    p.add_argument('--functional-baseline', default='functional-bf128'); p.add_argument('--functional-candidate', default='functional-fp128')
    p.add_argument('--name', default='quality-comparison'); a = p.parse_args(argv)
    reports = a.root/'reports'
    read = lambda name: strict_json((reports/f'{name}.json').read_text())
    try:
        baseline, candidate = read(a.baseline_report), read(a.candidate_report or a.candidate)
        report = compare(baseline, candidate, read(a.functional_baseline), read(a.functional_candidate),
                         evidence_records(reports/'teacher.jsonl', [baseline,candidate]), a.baseline, a.candidate)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as e:
        report = {'schema':2, 'evidence_valid':False, 'quality_gate_pass':False, 'bit_exact_pass':False,
                  'error':str(e), 'baseline':a.baseline, 'candidate':a.candidate}
    (reports/f'{a.name}.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, allow_nan=False))
    return 0 if report['bit_exact_pass'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
