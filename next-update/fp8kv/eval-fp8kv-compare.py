"""FP8-KV quality decision for the released build (PRE-REGISTERED; this file's RULE is frozen in
eval-fp8kv-freeze.json before any GPU run).

Inputs (native root /home/USER/qwen36-api/optimization):
  --quality NAME        reports/NAME.json from compare-quality.py (teacher-bf128 vs teacher-main-release-fp8kv;
                        teacher-bf128 is the release reference: teacher-main-release is 0-diff to it)
  --baseline NAME       reports/NAME-eval-v1.json, profile main-release      (eval-fp8kv.py run)
  --candidate NAME      reports/NAME-eval-v1.json, profile main-release-fp8kv (eval-fp8kv.py run)
  --functional-baseline / --functional-candidate   reports/functional-*.json (quality.py, 5 fixtures)
  engine.jsonl          kv_clipped_values_total and kv_dtype of every FP8 serving run used above
Output: --out PATH (O_EXCL), never overwritten. Reads verdicts/hashes only, never reply text.

RULE (QUALITY_PASS iff ALL hold; otherwise QUALITY_FAIL; non-comparable inputs -> NO_DECISION):
  (1) teacher: relative perplexity growth point <= +0.25 % AND bootstrap CI95 upper <= +1 %
      (compare-quality.py: 20 cases / 943 tokens, 20000 resamples, seed 4172026); forced targets complete;
      125 logit comparisons complete and finite (they are NOT required to be 0: FP8 KV is not bit-exact);
  (2) functional: no new failure vs the release (known release failures: executable-code, long-facts-16384);
  (3) eval-v1 (60 cases, same build, same day): PASS(fp8) >= PASS(release) - 1; every category net loss <= 1;
      no tool case that passes on the release fails on FP8;
  (4) kv_clipped_values_total == 0 in every FP8 serving run (analytic no-clip scales: > 0 means a bound bug).
  Informational only: McNemar exact p, the 4 long cases at ~120K, per-case NLL deltas, max logit deltas.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import sys

ROOT = pathlib.Path('/home/USER/qwen36-api/optimization')
CATEGORIES = ('code', 'ru', 'tool', 'agent', 'long')
FP8_ENV = {'QWEN36_KV_DTYPE': 'fp8', 'QWEN36_KV_SCALES': '/home/USER/qwen36-api/optimization/config/kv-scales-bound.json'}
RULE = {
    'id': 'fp8kv-quality-rule-1',
    'baseline_profile': 'main-release',
    'candidate_profile': 'main-release-fp8kv',
    'kv_scales': {'path': FP8_ENV['QWEN36_KV_SCALES'], 'method': 'analytic-bound-v1',
                  'sha256': 'e9888b62c30f04e355902bad68ae908bc2912d143a2c26738974e7026d0b1de0'},
    'nll': {'reference': 'teacher-bf128 (release reference, teacher-main-release 0-diff)', 'cases': 20, 'tokens': 943,
            'max_relative_ppl_growth_point': 0.0025, 'max_relative_ppl_growth_ci95_upper': 0.01,
            'require_forced_targets_complete': True, 'require_logits_finite_complete': True},
    'functional': {'max_new_failures': 0},
    'eval': {'max_total_pass_deficit': 1, 'max_category_net_loss': 1, 'max_tool_losses': 0},
    'clipping': {'max_kv_clipped_values_total': 0},
    'outcomes': {'all_pass': 'QUALITY_PASS', 'any_fail': 'QUALITY_FAIL', 'inputs_invalid': 'NO_DECISION'},
    'protocol': 'first VALID eval run of each profile under eval-fp8kv-freeze.json counts; INVALID may be repeated '
                'under a new name; VALID runs are never repeated or cherry-picked',
}


def sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def rule_sha256():
    return hashlib.sha256(json.dumps(RULE, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load(root, name, suffix=''):
    return json.loads((root / 'reports' / f'{name}{suffix}.json').read_text(encoding='utf-8'))


def eval_problems(a, b, freeze_sha):
    problems = []
    for label, rep, profile in (('baseline', a, RULE['baseline_profile']), ('candidate', b, RULE['candidate_profile'])):
        if rep.get('kind') != 'eval-v1' or rep.get('status') != 'VALID' or rep.get('profile') != profile:
            problems.append(f'{label}_not_valid_or_wrong_profile')
        if rep.get('freeze_sha256') != freeze_sha:
            problems.append(f'{label}_other_freeze')
    for key in ('suite_sha256', 'sources', 'sampling', 'decision_rule_sha256', 'bench_sha256'):
        if a.get(key) != b.get(key):
            problems.append(f'differs_{key}')
    if [(r['id'], r['sha256']) for r in a.get('rows', [])] != [(r['id'], r['sha256']) for r in b.get('rows', [])]:
        problems.append('differs_cases')
    ra, rb = a.get('run') or {}, b.get('run') or {}
    for key in ('source_sha256', 'packages', 'checkpoint_revision', 'native_libraries', 'service_source_files'):
        if ra.get(key) != rb.get(key):
            problems.append(f'differs_run_{key}')
    ca, cb = ra.get('configuration') or {}, rb.get('configuration') or {}
    stripped = dict(cb, env={k: v for k, v in (cb.get('env') or {}).items() if k not in FP8_ENV})
    if any((cb.get('env') or {}).get(k) != v for k, v in FP8_ENV.items()) or stripped != ca:
        problems.append('configurations_differ_beyond_fp8_kv_env')
    return problems


def first_valid(root, freeze_sha, reports):
    problems = []
    for path in sorted((root / 'reports').glob('*-eval-v1.json')):
        try:
            other = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        for label, rep in reports.items():
            if (other.get('status') == 'VALID' and other.get('freeze_sha256') == freeze_sha
                    and other.get('profile') == rep.get('profile') and other.get('name') != rep.get('name')
                    and str(other.get('started_utc')) < str(rep.get('started_utc'))):
                problems.append(f'{label}_not_first_valid_run')
    return problems


def paired(a, b):
    pa = {r['id']: bool(r['pass']) for r in a['rows']}
    pb = {r['id']: bool(r['pass']) for r in b['rows']}
    cat = {r['id']: r['category'] for r in a['rows']}
    out = {'by_category': {}}
    for c in CATEGORIES + ('all', 'long-120k'):
        ids = [i for i in pa if c == 'all' or cat[i] == c or (c == 'long-120k' and i.startswith('long-120k'))]
        lost = sorted(i for i in ids if pa[i] and not pb[i])
        won = sorted(i for i in ids if not pa[i] and pb[i])
        entry = {'cases': len(ids), 'pass_release': sum(pa[i] for i in ids), 'pass_fp8': sum(pb[i] for i in ids),
                 'release_only_pass': lost, 'fp8_only_pass': won, 'net_loss': len(lost) - len(won),
                 'mcnemar_exact_two_sided_p': mcnemar_exact(len(lost), len(won))}
        if c == 'all':
            out['overall'] = entry
        else:
            out['by_category'][c] = entry
    return out


def clipping(root, run_ids):
    rows = {}
    for line in (root / 'reports' / 'engine.jsonl').read_text(encoding='utf-8').splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get('run_id') in run_ids:
            entry = rows.setdefault(r['run_id'], {'requests': 0, 'kv_dtype': set(), 'max_clipped': 0, 'expert_slots': set(),
                                                  'kv_bytes': set()})
            entry['requests'] += 1
            entry['kv_dtype'].add(r.get('kv_dtype'))
            entry['expert_slots'].add(r.get('expert_slots'))
            entry['kv_bytes'].add(r.get('kv_bytes'))
            entry['max_clipped'] = max(entry['max_clipped'], int(r.get('kv_clipped_values_total') or 0))
    return {k: dict(v, kv_dtype=sorted(map(str, v['kv_dtype'])), expert_slots=sorted(v['expert_slots']),
                    kv_bytes=sorted(v['kv_bytes'])) for k, v in rows.items()}


def main(argv=None, root=None):
    root = ROOT if root is None else pathlib.Path(root)
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    for arg in ('--quality', '--baseline', '--candidate', '--functional-baseline', '--functional-candidate', '--out'):
        p.add_argument(arg, required=True)
    p.add_argument('--fp8-run-ids', required=True, help='comma list: teacher, functional and eval FP8 serving run_ids')
    a = p.parse_args(argv)
    out = pathlib.Path(a.out)
    if out.exists():
        print(json.dumps({'result': 'refused', 'reason': 'report_exists'}), file=sys.stderr)
        return 2
    freeze_path = here / 'eval-fp8kv-freeze.json'
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    problems = []
    if freeze.get('decision_rule') != RULE or freeze.get('decision_rule_sha256') != rule_sha256():
        problems.append('rule_differs_from_freeze')
    q = load(root, a.quality)
    base, cand = load(root, a.baseline, '-eval-v1'), load(root, a.candidate, '-eval-v1')
    fb, fc = load(root, a.functional_baseline), load(root, a.functional_candidate)
    freeze_sha = sha256_file(freeze_path)
    problems += eval_problems(base, cand, freeze_sha)
    problems += first_valid(root, freeze_sha, {'baseline': base, 'candidate': cand})
    if q.get('baseline') not in (None, 'teacher-bf128') or q.get('candidate') != 'teacher-main-release-fp8kv':
        problems.append('quality_report_wrong_profiles')
    fp8_runs = [x for x in a.fp8_run_ids.split(',') if x]
    clip = clipping(root, set(fp8_runs))
    if set(clip) != set(fp8_runs) or any(v['kv_dtype'] != ['torch.float8_e4m3fn'] for v in clip.values()):
        problems.append('fp8_runs_missing_or_not_fp8')
    lc = q.get('logit_comparisons') or {}
    fail_new = [c['case'] for c in fc['rows'] if not c['pass']
                and next(x['pass'] for x in fb['rows'] if x['case'] == c['case'])]
    pairs = paired(base, cand) if not problems else None
    criteria = None
    if pairs is not None:
        e, n = RULE['eval'], RULE['nll']
        growth, ci = q['relative_perplexity_growth'], q['relative_growth_ci95']
        criteria = {
            'teacher': {'relative_perplexity_growth': growth, 'ci95': ci, 'forced_targets_complete': q.get('forced_targets_complete'),
                        'logits': lc, 'pass': bool(growth <= n['max_relative_ppl_growth_point'] and ci[1] <= n['max_relative_ppl_growth_ci95_upper']
                                                   and q.get('forced_targets_complete') is True and lc.get('finite_and_complete') is True)},
            'functional': {'new_failures': fail_new, 'pass': len(fail_new) <= RULE['functional']['max_new_failures']},
            'eval_total': {'pass_release': pairs['overall']['pass_release'], 'pass_fp8': pairs['overall']['pass_fp8'],
                           'pass': pairs['overall']['pass_fp8'] >= pairs['overall']['pass_release'] - e['max_total_pass_deficit']},
            'eval_categories': {'net_loss': {c: pairs['by_category'][c]['net_loss'] for c in CATEGORIES},
                                'pass': all(pairs['by_category'][c]['net_loss'] <= e['max_category_net_loss'] for c in CATEGORIES)},
            'tool': {'losses': pairs['by_category']['tool']['release_only_pass'],
                     'pass': len(pairs['by_category']['tool']['release_only_pass']) <= e['max_tool_losses']},
            'clipping': {'max': max(v['max_clipped'] for v in clip.values()),
                         'pass': max(v['max_clipped'] for v in clip.values()) <= RULE['clipping']['max_kv_clipped_values_total']},
        }
    decision = (RULE['outcomes']['inputs_invalid'] if criteria is None else
                RULE['outcomes']['all_pass'] if all(v['pass'] for v in criteria.values()) else RULE['outcomes']['any_fail'])
    report = {'kind': 'fp8kv-quality-decision', 'rule': RULE, 'rule_sha256': rule_sha256(), 'freeze_sha256': freeze_sha,
              'problems': sorted(set(problems)), 'decision': decision, 'criteria': criteria, 'pairs': pairs,
              'fp8_serving_runs': clip, 'teacher_rows': q.get('rows')}
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False) + '\n')
    print(json.dumps({'decision': decision, 'problems': report['problems'],
                      'criteria': {k: v['pass'] for k, v in (criteria or {}).items()}}))
    return 0 if decision != 'NO_DECISION' else 1


if __name__ == '__main__':
    raise SystemExit(main())
