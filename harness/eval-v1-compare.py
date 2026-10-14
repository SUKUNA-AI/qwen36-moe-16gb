"""eval-v1 paired comparison + PRE-REGISTERED basis decision: fp32 NVFP4 global vs fp16-compat.

Inputs (all on the native root):
  --baseline NAME    reports/NAME-eval-v1.json of profile bf128-fp16compat (old fp16-global numbers)
  --candidate NAME   reports/NAME-eval-v1.json of profile bf128 (fp32 global)
  --fp16-teacher-run-id / --fp32-teacher-run-id   run_ids of the two teacher-bf128 runs in
                     reports/teacher.jsonl (pre-R1 fp16 and R1 fp32; e.g. from compare-logit-reference)
Output: reports/NAME.json once (O_EXCL). Text replies are never read (reports hold hashes only).

RULE (frozen in eval-v1/freeze.json before any eval-v1 GPU run; this file's sha256 is frozen too):
  fp32 becomes the basis iff ALL hold, otherwise the basis is fp16-compat:
  (1) teacher NLL, fp32 vs fp16 on the 20 fixed teacher cases: relative perplexity growth
      point estimate <= +0.1 % AND paired cluster-bootstrap CI95 upper bound <= +1 % (the project's
      existing NLL gate of compare-quality.py: 20000 resamples, seed 4172026, token-weighted);
  (2) eval-v1: PASS(fp32) >= PASS(fp16-compat) - 1, and in every category the net loss
      #(fp16 PASS & fp32 FAIL) - #(fp16 FAIL & fp32 PASS) <= 1;
  (3) tool calling: no tool case that passes on fp16-compat fails on fp32.
  Inputs that are not comparable (an INVALID run, different fixture/suite/sources/runtime/packages,
  configurations differing in anything but the compat flag, incomplete NLL data) give NO_DECISION.
  McNemar's exact two-sided test (overall and per category) is reported, not a decision input:
  with 60 pairs it has little power, so the decision uses the pre-registered margins above.
Exit: 0 decision written; 1 NO_DECISION written; 2 refused/usage.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import sys

ROOT = pathlib.Path('/home/USER/qwen36-api/optimization')
NAME_PATTERN = re.compile(r'[a-z0-9][a-z0-9.-]{0,90}')
CATEGORIES = ('code', 'ru', 'tool', 'agent', 'long')
COMPAT_FLAG = 'QWEN36_NVFP4_GLOBAL_FP16_COMPAT'
RULE = {
    'id': 'eval-v1-basis-rule-1',
    'baseline_profile': 'bf128-fp16compat',
    'candidate_profile': 'bf128',
    'nll': {'teacher_profile': 'teacher-bf128', 'cases': 20, 'max_relative_ppl_growth_point': 0.001,
            'max_relative_ppl_growth_ci95_upper': 0.01, 'bootstrap_resamples': 20000, 'bootstrap_seed': 4172026},
    'eval': {'max_total_pass_deficit': 1, 'max_category_net_loss': 1, 'max_tool_losses': 0},
    'outcomes': {'all_criteria_pass': 'fp32', 'any_criterion_fails': 'fp16-compat', 'inputs_invalid': 'NO_DECISION'},
    'mcnemar': 'exact two-sided binomial test on discordant pairs; informational only',
    'protocol': 'the first VALID eval-v1 run of each profile under this freeze counts (checked over '
                'reports/*-eval-v1.json); an INVALID run may be repeated under a new --name; a VALID run is never '
                'repeated, replaced or cherry-picked',
}


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value: 2 * P(X <= min(b, c)), X ~ Bin(b + c, 1/2), capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_report(root, name):
    return json.loads((root / 'reports' / f'{name}-eval-v1.json').read_text(encoding='utf-8'))


def comparable(a, b):
    """Static problems that make the two eval reports incomparable (empty = comparable)."""
    problems = []
    for label, report, profile in (('baseline', a, RULE['baseline_profile']), ('candidate', b, RULE['candidate_profile'])):
        if report.get('kind') != 'eval-v1' or report.get('schema_version') != 1:
            problems.append(f'{label}_schema')
        if report.get('status') != 'VALID':
            problems.append(f'{label}_invalid')
        if report.get('profile') != profile:
            problems.append(f'{label}_profile')
    for key in ('freeze_sha256', 'suite_sha256', 'sources', 'sampling', 'decision_rule_sha256', 'bench_sha256'):
        if a.get(key) != b.get(key):
            problems.append(f'differs_{key}')
    rows_a = [(r.get('id'), r.get('sha256'), r.get('category')) for r in a.get('rows', [])]
    rows_b = [(r.get('id'), r.get('sha256'), r.get('category')) for r in b.get('rows', [])]
    if rows_a != rows_b or not rows_a:
        problems.append('differs_cases')
    ra, rb = a.get('run') or {}, b.get('run') or {}
    for key in ('source_sha256', 'packages', 'checkpoint', 'checkpoint_revision', 'native_libraries',
                'service_source_files'):
        if ra.get(key) != rb.get(key):
            problems.append(f'differs_run_{key}')
    if ra.get('run_id') == rb.get('run_id'):
        problems.append('same_serving_run')
    ca, cb = ra.get('configuration'), rb.get('configuration')
    if not isinstance(ca, dict) or not isinstance(cb, dict):
        problems.append('missing_configuration')
    else:
        stripped = dict(ca, env={k: v for k, v in ca.get('env', {}).items() if k != COMPAT_FLAG})
        if ca.get('env', {}).get(COMPAT_FLAG) != '1' or COMPAT_FLAG in cb.get('env', {}) or stripped != cb:
            problems.append('configurations_differ_beyond_compat_flag')
    return problems


def first_valid_problems(root, freeze_sha, chosen):
    """Every earlier VALID run of the same profile under the same freeze must not exist (no cherry-picking)."""
    problems = []
    for path in sorted((root / 'reports').glob('*-eval-v1.json')):
        try:
            other = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        for label, report in chosen.items():
            if (other.get('kind') == 'eval-v1' and other.get('status') == 'VALID'
                    and other.get('freeze_sha256') == freeze_sha and other.get('profile') == report.get('profile')
                    and other.get('name') != report.get('name')
                    and str(other.get('started_utc')) < str(report.get('started_utc'))):
                problems.append(f'{label}_not_first_valid_run')
    return problems


def paired(a, b):
    """Per-case pairs and per-category/overall discordance (a = baseline fp16-compat, b = candidate fp32)."""
    pa = {r['id']: bool(r['pass']) for r in a['rows']}
    pb = {r['id']: bool(r['pass']) for r in b['rows']}
    category = {r['id']: r['category'] for r in a['rows']}
    out = {'by_category': {}}
    for cat in CATEGORIES + ('all',):
        ids = [i for i in pa if cat == 'all' or category[i] == cat]
        lost = sorted(i for i in ids if pa[i] and not pb[i])
        won = sorted(i for i in ids if not pa[i] and pb[i])
        entry = {'cases': len(ids), 'pass_baseline': sum(pa[i] for i in ids), 'pass_candidate': sum(pb[i] for i in ids),
                 'both_pass': sum(pa[i] and pb[i] for i in ids), 'both_fail': sum(not pa[i] and not pb[i] for i in ids),
                 'baseline_only_pass': lost, 'candidate_only_pass': won, 'net_loss': len(lost) - len(won),
                 'mcnemar_exact_two_sided_p': mcnemar_exact(len(lost), len(won))}
        if cat == 'all':
            out['overall'] = entry
        else:
            out['by_category'][cat] = entry
    return out


def teacher_runs(path, profile):
    runs = {}
    for line in pathlib.Path(path).read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get('profile') == profile and isinstance(row.get('nll'), list):
            runs.setdefault(row.get('run_id'), {})[row.get('case')] = row['nll']
    return runs


def nll_comparison(path, fp16_run, fp32_run):
    """Relative perplexity growth fp32 vs fp16 with the compare-quality.py bootstrap; None + problems if unusable."""
    import numpy as np
    rule = RULE['nll']
    runs = teacher_runs(path, rule['teacher_profile'])
    problems = []
    if fp16_run == fp32_run:
        problems.append('nll_same_run_id')
    a, b = runs.get(fp16_run), runs.get(fp32_run)
    if not a or not b:
        return None, problems + ['nll_run_missing']
    if set(a) != set(b) or len(a) != rule['cases']:
        return None, problems + ['nll_case_set']
    keys = sorted(a)
    for key in keys:
        if len(a[key]) != len(b[key]) or not a[key]:
            return None, problems + ['nll_length_mismatch']
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in a[key] + b[key]):
            return None, problems + ['nll_nonfinite']
    if problems:
        return None, problems
    length = np.array([len(a[k]) for k in keys])
    diff = np.array([sum(b[k]) - sum(a[k]) for k in keys])
    rng = np.random.default_rng(rule['bootstrap_seed'])
    idx = rng.integers(0, len(keys), (rule['bootstrap_resamples'], len(keys)))
    boot = np.exp(diff[idx].sum(1) / length[idx].sum(1)) - 1
    growth = math.exp(diff.sum() / length.sum()) - 1
    ci = [float(v) for v in np.quantile(boot, [.025, .975])]
    return {'fp16_run_id': fp16_run, 'fp32_run_id': fp32_run, 'cases': len(keys), 'tokens': int(length.sum()),
            'relative_perplexity_growth': growth, 'relative_growth_ci95': ci,
            'pass': bool(growth <= rule['max_relative_ppl_growth_point'] and ci[1] <= rule['max_relative_ppl_growth_ci95_upper'])}, []


def decide(pairs, nll):
    e = RULE['eval']
    overall, cats = pairs['overall'], pairs['by_category']
    criteria = {
        'nll': {'pass': bool(nll and nll['pass'])},
        'eval_total': {'pass_baseline': overall['pass_baseline'], 'pass_candidate': overall['pass_candidate'],
                       'pass': overall['pass_candidate'] >= overall['pass_baseline'] - e['max_total_pass_deficit']},
        'eval_categories': {'net_loss': {c: cats[c]['net_loss'] for c in CATEGORIES},
                            'pass': all(cats[c]['net_loss'] <= e['max_category_net_loss'] for c in CATEGORIES)},
        'tool': {'losses': cats['tool']['baseline_only_pass'],
                 'pass': len(cats['tool']['baseline_only_pass']) <= e['max_tool_losses']},
    }
    ok = all(v['pass'] for v in criteria.values())
    return ('fp32' if ok else 'fp16-compat'), criteria


def file_sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main(argv=None, root=None, freeze_path=None):
    root = ROOT if root is None else pathlib.Path(root)
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--fp16-teacher-run-id', required=True)
    p.add_argument('--fp32-teacher-run-id', required=True)
    p.add_argument('--teacher-jsonl')
    a = p.parse_args(argv)
    if not all(NAME_PATTERN.fullmatch(v) for v in (a.name, a.baseline, a.candidate)):
        p.error('invalid name')
    target = root / 'reports' / f'{a.name}.json'
    if target.exists():
        print(json.dumps({'result': 'refused', 'reason': 'report_exists'}), file=sys.stderr)
        return 2
    here = pathlib.Path(__file__).resolve().parent
    freeze_path = pathlib.Path(freeze_path or here / 'eval-v1' / 'freeze.json')
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    rule_sha = hashlib.sha256(json.dumps(RULE, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    problems = []
    if freeze.get('decision_rule') != RULE or freeze.get('decision_rule_sha256') != rule_sha:
        problems.append('rule_differs_from_freeze')
    if freeze.get('sources', {}).get('eval-v1-compare.py') != file_sha256(__file__):
        problems.append('compare_source_differs_from_freeze')
    try:
        base, cand = load_report(root, a.baseline), load_report(root, a.candidate)
    except (OSError, ValueError):
        print(json.dumps({'result': 'refused', 'reason': 'report_unreadable'}), file=sys.stderr)
        return 2
    problems += comparable(base, cand)
    if base.get('freeze_sha256') != file_sha256(freeze_path):
        problems.append('reports_use_other_freeze')
    problems += first_valid_problems(root, file_sha256(freeze_path), {'baseline': base, 'candidate': cand})
    try:
        nll, nll_problems = nll_comparison(a.teacher_jsonl or root / 'reports' / 'teacher.jsonl',
                                           a.fp16_teacher_run_id, a.fp32_teacher_run_id)
    except (OSError, ValueError) as error:
        nll, nll_problems = None, ['nll_unreadable:' + type(error).__name__]
    problems += nll_problems
    pairs = paired(base, cand) if 'differs_cases' not in problems and base.get('rows') else None
    report = {'kind': 'eval-v1-comparison', 'rule': RULE, 'rule_sha256': rule_sha,
              'freeze_sha256': file_sha256(freeze_path),
              'baseline': {'name': a.baseline, 'profile': base.get('profile'), 'run_id': (base.get('run') or {}).get('run_id')},
              'candidate': {'name': a.candidate, 'profile': cand.get('profile'), 'run_id': (cand.get('run') or {}).get('run_id')},
              'problems': sorted(set(problems)), 'nll': nll, 'pairs': pairs}
    if problems or pairs is None:
        report['decision'] = RULE['outcomes']['inputs_invalid']
        report['criteria'] = None
    else:
        report['decision'], report['criteria'] = decide(pairs, nll)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as out:
        out.write(json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False) + '\n')
    print(json.dumps({'decision': report['decision'], 'problems': report['problems'], 'criteria': report['criteria'],
                      'overall': pairs and pairs['overall'], 'report': target.name}, ensure_ascii=False))
    return 0 if report['decision'] != 'NO_DECISION' else 1


if __name__ == '__main__':
    raise SystemExit(main())
