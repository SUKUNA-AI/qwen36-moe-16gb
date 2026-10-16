"""CPU-only stage Z: release-gate stage verdicts of the main128 candidate, with sha256 of every bound report.

python3 -I ladder_gates.py --reports <native>/reports --logs DIR --run-ts TS --out gates.json
Stage D is NOT an own run of release-gates.sh: it is the external release A/B named below (explicit, with hashes).
"""
import argparse, hashlib, json, pathlib


def sha(path):
    try:
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def jl(path):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reports', type=pathlib.Path, required=True)
    p.add_argument('--logs', type=pathlib.Path, required=True)
    p.add_argument('--run-ts', required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    p.add_argument('--tree', required=True)
    a = p.parse_args()
    R, TS = a.reports, a.run_ts
    T = f'rg-s5-{TS}'
    steps = {}
    for line in (a.logs / f'gates-{TS}.steps').read_text().splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2:
            steps[parts[0]] = {'verdict': parts[1], 'detail': parts[2] if len(parts) > 2 else ''}
    soak, pilot = f'soak-{T}', f'soak-{T}-pilot'
    qcheck = jl(a.logs / f'gB-qcheck-{TS}.json') or {}
    ctx = jl(a.logs / f'gC-check-{TS}.json') or {}
    sck = jl(a.logs / f'gE-check-{soak}-{TS}.json') or {}
    ana = jl(R / f'{soak}-analysis.json') or {}
    ext = {'name': 's5-rab-20261015t2100', 'tree': '68605f5 (runtime identical to 3052011 and 19f8d1f for freetoken/)',
           'baseline': 'main-release', 'candidate': 'main-release-next-d5', 'pairs': 5}
    cmp_ = jl(R / 's5-rab-20261015t2100-main-release-next-d5-vs-main-release-comparison.json') or {}
    ext.update(verdict=cmp_.get('verdict'), contract_pass=cmp_.get('contract_pass'), reasons=cmp_.get('verdict_reasons'),
               manifest_sha256=sha(R / 's5-rab-20261015t2100-manifest.json'),
               comparison_sha256=sha(R / 's5-rab-20261015t2100-main-release-next-d5-vs-main-release-comparison.json'),
               matrix_sha256=sha(R / 's5-rab-20261015t2100-matrix.json'))
    l256 = None
    for line in (a.logs / f'gates-{TS}.steps').read_text().splitlines():
        if line.startswith('L256CTX'):
            _, rc, cc, name = line.split()
            l256 = {'name': name, 'run_context_gates_rc': int(rc), 'context_check_rc': int(cc),
                    'check': jl(a.logs / f'g256-check-{TS}.json')}
    out = {'schema': 'ladder-gates-1', 'candidate': 'main-release-next-d5', 'tree': a.tree, 'run': T,
           'stages': {'A': steps.get('A'), 'B': steps.get('B'), 'C': steps.get('C'),
                      'D': {'verdict': ext['verdict'], 'detail': 'EXTERNAL: release A/B ' + ext['name'] + ', not an own release-gates.sh run', 'external': ext},
                      'E.pilot': steps.get(f'E.{pilot}'), 'E.soak': steps.get(f'E.{soak}')},
           'qcheck': qcheck, 'context_check_c128': ctx, 'soak_check': sck,
           'reports_sha256': {n: sha(R / f'{n}.json') for n in (f'quality-{T}-q-main-release-next-d5', f'{T}-q-main-release-next-d5-teacher',
                              f'{T}-q-main-release-next-d5-functional', f'{T}-qrun-teacher', f'{T}-qrun-functional', f'{T}-c128-functional',
                              f'{T}-c128-boundary', f'{T}-c128-cancel', soak, f'{soak}-analysis', pilot, f'{T}-kernel-provenance',
                              's5-ref-20261015t1545', 's5-fb-20261015t1545')},
           'soak': {'summary': ana.get('summary'), 'reference_comparison': ana.get('reference_comparison'), 'verdict': ana.get('verdict')},
           'long256_context_gates': l256,
           'driver_notes': ['release-gates.sh --dry-run failed on "overlay harness == checkout" for qualify.py (sync renders site placeholders into the overlay copy, so a byte compare cannot match)',
                            '--reference-soak needs the .json file name; stages A, B, C, E were run one by one with the commands of --dry-run']}
    out['all_stages_pass'] = all((v or {}).get('verdict') == 'PASS' for v in out['stages'].values()) and (l256 or {}).get('context_check_rc') == 0
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'all_stages_pass': out['all_stages_pass'], 'stages': {k: (v or {}).get('verdict') for k, v in out['stages'].items()}}))


if __name__ == '__main__':
    main()
