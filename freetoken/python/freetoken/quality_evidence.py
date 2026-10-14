"""CPU-only, fail-closed teacher evidence shared by engine and harness.

Identity is (profile, run_id, case). Rotation order never selects a run. Only an
identical serialized record may repeat; conflicting records and damaged JSON fail.
"""
import hashlib
import json
import math
import re
from pathlib import Path

TEACHER_CASES = tuple([f'ru-{i}' for i in range(6)] + [f'code-{i}' for i in range(6)] +
                      [f'json-{i}' for i in range(4)] + [f'long-{n}' for n in (4096, 16384, 65536, 120000)])
FUNCTIONAL_LENGTHS = (60000, 120000, 250000)

class EvidenceError(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise EvidenceError(message)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def strict_json(text):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, f'duplicate JSON field: {key}')
            out[key] = value
        return out
    def constant(value):
        raise EvidenceError(f'nonfinite JSON: {value}')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)

def teacher_records(path):
    path = Path(path)
    rotated = sorted((p for p in path.parent.glob(path.name + '.*') if p.name[len(path.name)+1:].isdigit()),
                     key=lambda p: int(p.name[len(path.name)+1:]), reverse=True)
    files = rotated + ([path] if path.exists() else [])
    require(bool(files), f'no teacher evidence: {path}')
    records, serialized = {}, {}
    for file in files:
        for number, line in enumerate(file.read_text().splitlines(), 1):
            try:
                row = strict_json(line)
                require(type(row) is dict, 'teacher record must be an object')
                key = tuple(row.get(k) for k in ('profile', 'run_id', 'case'))
                require(all(type(x) is str and x for x in key), 'teacher identity missing')
                if key in records:
                    require(serialized[key] == line, f'conflicting duplicate: {key}')
                else:
                    records[key], serialized[key] = row, line
            except (ValueError, TypeError) as e:
                raise EvidenceError(f'{file.name}:{number}: {e}') from e
    return records

def evidence_records(path, reports):
    path = Path(path)
    files = [path] if path.exists() else []
    files += [p for p in path.parent.glob(path.name+'.*') if p.name[len(path.name)+1:].isdigit()]
    records = teacher_records(path) if files else {}
    for report in reports:
        snapshot = report.get('teacher_evidence')
        if snapshot is None:
            continue
        require(type(snapshot) is list and snapshot, 'empty/invalid teacher snapshot')
        seen = set()
        for row in snapshot:
            require(type(row) is dict, 'snapshot row type')
            key = tuple(row.get(k) for k in ('profile','run_id','case'))
            require(all(type(k) is str and k for k in key) and key not in seen, 'snapshot duplicate/missing identity')
            seen.add(key)
            require(key not in records or records[key] == row, f'snapshot conflicts with teacher log: {key}')
            records[key] = row
    require(bool(records), 'no teacher evidence or run-scoped snapshots')
    return records


def hash_valid(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None

def validate_receipt(r):
    sources = r['source_files']
    require(type(sources) is dict and sources and all(type(k) is str and not Path(k).is_absolute()
            and '..' not in Path(k).parts and hash_valid(v) for k,v in sources.items()), 'invalid source files')
    require(r['source_sha256'] == hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
            'source aggregate mismatch')
    for field in ('service_source_files','packages','native_libraries'):
        require(type(r[field]) is dict, f'invalid {field}')
    require(set(r['service_source_files']) == {'experiment.py','serve_guarded.py','guard.py','ingress.py','request_metrics.py'}
            and all(hash_valid(v) for v in r['service_source_files'].values()), 'invalid service hashes')
    require(set(r['packages']) == {'torch','triton','flashinfer-python','sglang-kernel','transformers'}
            and all(type(v) is str and v for v in r['packages'].values()), 'invalid package provenance')
    cfg = r['configuration']
    require(type(cfg) is dict and type(cfg.get('env')) is dict and type(cfg.get('context')) is int and cfg['context'] > 0
            and type(cfg.get('chunk')) is int and cfg['chunk'] > 0
            and type(cfg.get('flags', [])) is list and all(type(f) is str for f in cfg.get('flags', [])), 'invalid teacher configuration')
    declared = {k:v for k,v in cfg['env'].items() if k.endswith('_LIBRARY')}
    require(set(r['native_libraries']) == set(declared), 'missing library provenance')
    for key, path in declared.items():
        lib = r['native_libraries'][key]
        require(type(lib) is dict and lib.get('path') == path and hash_valid(lib.get('sha256')), 'library provenance mismatch')
    require(all(type(r[k]) is str and r[k] for k in ('checkpoint','checkpoint_revision')), 'missing checkpoint provenance')


def receipt_map(report, profile):
    first = report.get('run')
    require(type(first) is dict, 'missing run receipt')
    clean = report.get('clean_cache', {})
    require(type(clean) is dict, 'invalid clean_cache')
    extra = clean.get('runs', [])
    require(type(extra) is list, 'invalid clean-cache receipts')
    receipts = {}
    for r in [first] + extra:
        require(type(r) is dict and r.get('profile') == profile and type(r.get('run_id')) is str and r['run_id'],
                'receipt identity differs/missing')
        require(r['run_id'] not in receipts, 'duplicate run receipt')
        for k in ('configuration', 'source_sha256', 'source_files', 'service_source_files', 'packages',
                  'checkpoint', 'checkpoint_revision', 'native_libraries'):
            require(k in r, f'receipt missing {k}')
        require(type(r['configuration']) is dict and type(r['packages']) is dict, 'invalid receipt provenance')
        validate_receipt(r)
        receipts[r['run_id']] = r
    if extra:
        require(clean.get('mode') == 'restart_same_profile' and clean.get('cases') == ['long-120000'],
                'unexpected clean-cache specification')
        same = lambda r: {k:v for k,v in r.items() if k not in ('run_id', 'started_utc')}
        require(all(same(r) == same(first) for r in extra), 'restart changed receipt provenance')
    return receipts

def unique_rows(rows, expected, label):
    require(type(rows) is list and rows, f'{label}: empty/missing rows')
    by_case = {}
    for row in rows:
        require(type(row) is dict and type(row.get('case')) is str, f'{label}: invalid case')
        require(row['case'] not in by_case, f'{label}: duplicate case {row["case"]}')
        by_case[row['case']] = row
    require(set(by_case) == set(expected), f'{label}: case set differs; missing={sorted(set(expected)-set(by_case))}, extra={sorted(set(by_case)-set(expected))}')
    return by_case

def finite_number(x):
    return type(x) in (int, float) and math.isfinite(x) and x >= 0

def bind_teacher(report, records, profile):
    require(type(report.get('fixture_version')) is int and report['fixture_version'] == 1, 'teacher fixture version missing/invalid')
    receipts = receipt_map(report, profile)
    rows = unique_rows(report.get('rows'), TEACHER_CASES, profile)
    fixture = report.get('fixture_sha256')
    require(hash_valid(fixture), 'missing fixture identity')
    targets = report.get('targets')
    require(type(targets) is dict and set(targets) == set(TEACHER_CASES), 'missing target specification')
    selected = {}
    for case, r in rows.items():
        run = r.get('run_id')
        require(run in receipts, f'{case}: case run is outside report receipts')
        clean = case == 'long-120000' and bool(report.get('clean_cache', {}).get('runs'))
        require(type(r.get('clean_cache')) is bool and r['clean_cache'] == clean, f'{case}: invalid clean-cache binding')
        require((run != report['run']['run_id']) == clean, f'{case}: incorrect case->run binding')
        key = (profile, run, case)
        require(key in records, f'missing teacher row: {key}')
        row = records[key]
        spec = targets[case]
        require(type(spec) is dict and type(spec.get('count')) is int and spec['count'] > 0 and
                hash_valid(spec.get('sha256')), f'{case}: invalid target spec')
        require(row.get('fixture_sha256') == fixture and row.get('targets_sha256') == spec['sha256'], f'{case}: fixture/targets mismatch')
        require(row.get('receipt_sha256') == digest(receipts[run]), f'{case}: receipt provenance mismatch')
        require(type(row.get('seed')) is int and row['seed'] == 900000 + TEACHER_CASES.index(case), f'{case}: seed mismatch')
        nll = row.get('nll')
        require(type(nll) is list and len(nll) == spec['count'] and all(map(finite_number, nll)), f'{case}: incomplete/nonfinite NLL')
        require(type(r.get('status')) is int and r['status'] == 200 and r.get('forced_text_matches') is True and
                type(r.get('expected_tokens')) is int and r['expected_tokens'] == spec['count'] and
                type(r.get('usage')) is dict and type(r['usage'].get('completion_tokens')) is int and
                r['usage']['completion_tokens'] == spec['count'], f'{case}: forced targets incomplete')
        comps = row.get('logit_comparisons')
        require(type(comps) is list and len(comps) == (spec['count']+7)//8, f'{case}: incorrect logit count')
        require([c.get('position') for c in comps if type(c) is dict] == list(range(0, spec['count'], 8)), f'{case}: logit positions differ')
        for c in comps:
            require(type(c.get('position')) is int and c.get('finite') is True and
                    type(c.get('byte_equal')) is bool and finite_number(c.get('max_abs')) and finite_number(c.get('relative_l2')), f'{case}: invalid/nonfinite logits')
        require(type(row.get('reference')) is dict, f'{case}: missing reference identity')
        selected[case] = row
    require(set(r['run_id'] for r in rows.values()) == set(receipts), 'unused/ambiguous receipt')
    for run in set(receipts) - {report['run']['run_id']}:
        own = {k[2] for k in records if k[:2] == (profile, run)}
        require(own == {'long-120000'}, 'clean-cache run served other cases')
    return selected

def functional_rows(report):
    matrix = report.get('functional_matrix')
    require(type(matrix) is dict and type(matrix.get('version')) is int and matrix['version'] == 1 and type(matrix.get('long')) is int and
            matrix['long'] in FUNCTIONAL_LENGTHS, 'missing/unsupported functional matrix')
    expected = ('russian', 'executable-code', 'tool-cycle', 'long-facts-16384', f'long-facts-{matrix["long"]}')
    rows = unique_rows(report.get('rows'), expected, 'functional')
    require(type(report.get('fixture_version')) is int and report['fixture_version'] == 1, 'functional fixture differs')
    for row in rows.values():
        require(type(row.get('pass')) is bool, 'functional pass must be boolean')
        require(type(row.get('status')) is int and row['status'] == 200, 'functional request status missing/invalid')
    return rows
