"""main128 next release helper for release-main128.sh: owner decision + staged profiles from named reports.

Offline, stdlib only. Reads the staged profiles document, the staged validator, the installed release receipt
(f1, release/qualification.json) and the named evidence paths (relative to the evidence root, e.g.
reports/x.json). Writes two NEW files (never overwrites): the staged profiles.json with the decided
``qualified`` flags and the schema-1 decision for make-qualification.py. It does not judge the evidence;
make-qualification.py checks every report and run receipt against the staged bytes.

    python release-main128.py decision --validator STAGE/serving/profile-runtime.py \
        --profiles-in serving/profiles.json --f1-receipt ROOT/release/qualification.json \
        --evidence-root ROOT/optimization --run reports/qrun-teacher-main128.json \
        --teacher reports/quality-main128.json --fla-component reports/gdn-wu.json \
        --overlap-component reports/overlap-component.json --ab reports/ab-main128-vs-f1.json \
        [--evidence reports/soak.json ...] [--limitation TEXT ...] [--limitations-file F.json] \
        [--long256 carry|rerun|drop] [--long256-run R --long256-teacher Q [--long256-bitexact Q]
         [--long256-evidence E] [--long256-limitation TEXT] [--long256-limitations-file F.json]] \
        [--baseline64 carry|drop] \
        [--profiles-out STAGE/serving/profiles.json --decision-out STAGE/decision.json]

Without --profiles-out/--decision-out it prints both documents and writes nothing (dry run).

    python release-main128.py export-tree --source CHECKOUT/freetoken --clone STAGE/export/FreeToken-opt

replaces python/freetoken and tests of a fresh upstream clone (checked out at the pinned base) with this
checkout's trees for export-release-evidence.py. The checkout stores most sources with CRLF, upstream with LF:
a file that differs from upstream only in line endings keeps the upstream bytes, a changed text source is
written with LF, so the exported patch shows exactly the real changes.

Profiles other than main128 keep their f1 decision only when their configuration is unchanged (``carry``).
A carried CUDA profile (long256) keeps its f1 run receipts, which are bound to the f1 runtime bytes: on a new
runtime tree make-qualification.py refuses them. Then either ``rerun`` (new run receipts and teacher report
on this tree) or ``drop`` (qualified=false: that profile is not startable in this release).

rerun: the long256 run receipts (qrun of teacher-long256-release on this tree) bind its CUDA switches;
their bit-exact reports are --long256-bitexact, default the long256 teacher report. At 262144 context the
f1 long256 teacher differed from teacher-bf128 on long-120000 only (max_abs 1.23, shared chat-header KV in
the radix cache; bit-identical on a clean cache) and f1 bound the same libraries to the 0-diff main128 teacher
report instead. If the new long256 teacher shows that again, pass the main128 candidate teacher report (same
libraries, same tree) as --long256-bitexact, keep the long256 teacher as evidence and state the deviation as a
known limitation. The carried f1 limitations stay unless --long256-limitations-file replaces them;
--long256-limitation appends.
"""
import argparse
import importlib.util
import json
import os
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
EXPORT_TREES = ('python/freetoken', 'tests')
TEXT_SUFFIXES = frozenset({'.py', '.cu', '.cuh', '.cpp', '.cc', '.c', '.h', '.hpp', '.json', '.md', '.txt', '.toml',
                           '.cfg', '.ini', '.yaml', '.yml', '.sh'})
MODES = {'long256': ('carry', 'rerun', 'drop'), 'baseline64': ('carry', 'drop')}


class DecisionError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise DecisionError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unique(items):
    return list(dict.fromkeys(items))


def evidence_paths(validator, root, paths, label):
    """Relative POSIX evidence paths that exist as regular files below root (hashing is make-qualification's)."""
    base = pathlib.Path(root).resolve()
    for relative in paths:
        _require(validator._relative_valid(relative), label + ': invalid evidence path ' + repr(relative)[:120])
        path = base.joinpath(relative)
        _require(path.resolve() == path and path.is_file() and not path.is_symlink(),
                 label + ': evidence file missing or not a regular file: ' + relative)
    return list(paths)


def carried_entry(validator, name, profile, record):
    """The f1 decision of an unchanged profile: status, limitations, evidence and its CUDA evidence lists."""
    _require(record is not None, name + ': the installed receipt has no record to carry')
    _require(record['configuration_sha256'] == validator.canonical_profile_sha256(profile),
             name + ': configuration changed since the installed receipt; carry is impossible (rerun or drop)')
    entry = {'acceptance_status': record['acceptance_status'], 'known_limitations': list(record['known_limitations']),
             'evidence': [row['path'] for row in record['evidence']]}
    return entry


def switch_evidence(maker, validator, profile, runs, teachers, fla, overlap, ab, name):
    """Decision keys for every CUDA and experimental switch the profile enables, from the named reports."""
    result = {}
    for switch, key in sorted(maker.CUDA_DECISION_EVIDENCE.items()):
        if switch not in validator.CUDA_EVIDENCE_SWITCHES or not validator.cuda_evidence_enabled(profile, switch):
            continue
        _require(runs and teachers, name + ': ' + switch + ' needs run receipts (--run) and teacher reports (--teacher)')
        result[key] = {'run_receipts': list(runs), 'bitexact_reports': list(teachers)}
        if switch == 'cuda_gdn_fla':
            _require(fla, name + ': cuda_gdn_fla needs the FLA component report (--fla-component)')
            result[key]['component_reports'] = list(fla)
    for switch, keys in sorted(getattr(validator, 'EXPERIMENTAL_EVIDENCE', {}).items()):
        if profile.get(switch) is not True:
            continue
        given = {'component_reports': overlap, 'run_receipts': runs, 'bitexact_reports': teachers, 'ab_reports': ab}
        missing = [key for key in keys if not given.get(key)]
        _require(not missing, name + ': ' + switch + ' needs ' + ', '.join(missing)
                 + ' (--overlap-component, --run, --teacher, --ab)')
        result[maker.EXPERIMENTAL_DECISION_EVIDENCE[switch]] = {key: list(given[key]) for key in keys}
    return result


def read_limits(validator, limits, path=None):
    """Known limitations: the given strings plus a JSON list file; every item non-blank."""
    limits = list(limits)
    if path:
        extra = validator.read_json(pathlib.Path(path))
        _require(type(extra) is list and all(type(item) is str and item.strip() for item in extra),
                 'a limitations file must hold a JSON list of non-blank strings')
        limits += extra
    _require(all(type(item) is str and item.strip() for item in limits), 'known limitations must be non-blank')
    return limits


def compose(validator, maker, profiles, f1, args):
    """Return (staged profiles, decision) for main128 plus the carried/rerun/dropped other profiles."""
    validator.validate_profiles_document(profiles)
    staged = json.loads(json.dumps(profiles))
    records = f1.get('profiles', {}) if type(f1) is dict else {}
    root = args.evidence_root
    decision = {}
    main = staged['profiles']['main128']
    main['qualified'] = True
    limits = read_limits(validator, list(args.limitation), args.limitations_file)
    runs = evidence_paths(validator, root, unique(args.run), 'main128 --run')
    teachers = evidence_paths(validator, root, unique(args.teacher), 'main128 --teacher')
    fla = evidence_paths(validator, root, unique(args.fla_component), 'main128 --fla-component')
    overlap = evidence_paths(validator, root, unique(args.overlap_component), 'main128 --overlap-component')
    ab = evidence_paths(validator, root, unique(args.ab), 'main128 --ab')
    extra = evidence_paths(validator, root, unique(args.evidence), 'main128 --evidence')
    entry = {'acceptance_status': 'accepted_with_known_limits' if limits else 'accepted', 'known_limitations': limits,
             'evidence': unique(runs + teachers + fla + overlap + ab + extra)}
    _require(entry['evidence'], 'main128: no evidence named')
    if runs:
        entry['run_receipts'] = list(runs)
    entry.update(switch_evidence(maker, validator, main, runs, teachers, fla, overlap, ab, 'main128'))
    decision['main128'] = entry
    rerun_only = [flag for flag, value in (('--long256-run', args.long256_run), ('--long256-teacher', args.long256_teacher),
                                           ('--long256-bitexact', args.long256_bitexact),
                                           ('--long256-evidence', args.long256_evidence),
                                           ('--long256-limitation', args.long256_limitation),
                                           ('--long256-limitations-file', args.long256_limitations_file)) if value]
    _require(args.long256 == 'rerun' or not rerun_only, ', '.join(rerun_only) + ' only with --long256 rerun')
    for name in ('long256', 'baseline64'):
        mode = getattr(args, name)
        profile = staged['profiles'][name]
        if mode == 'drop':
            profile['qualified'] = False
            continue
        profile['qualified'] = True
        entry = carried_entry(validator, name, profile, records.get(name))
        if mode == 'rerun':
            new_runs = evidence_paths(validator, root, unique(args.long256_run), name + ' --long256-run')
            new_teachers = evidence_paths(validator, root, unique(args.long256_teacher), name + ' --long256-teacher')
            _require(new_runs and new_teachers, name + ' rerun needs --long256-run and --long256-teacher on this tree')
            bitexact = evidence_paths(validator, root, unique(args.long256_bitexact), name + ' --long256-bitexact') or new_teachers
            entry['evidence'] = unique(entry['evidence'] + new_runs + new_teachers + bitexact
                                       + evidence_paths(validator, root, unique(args.long256_evidence), name))
            entry['run_receipts'] = list(new_runs)
            carried = [] if args.long256_limitations_file else entry['known_limitations']
            entry['known_limitations'] = read_limits(validator, carried + list(args.long256_limitation),
                                                     args.long256_limitations_file)
            entry['acceptance_status'] = 'accepted_with_known_limits' if entry['known_limitations'] else 'accepted'
            entry.update(switch_evidence(maker, validator, profile, new_runs, bitexact, [], [], [], name))
        else:
            record = records[name]
            for switch, key in sorted(maker.CUDA_DECISION_EVIDENCE.items()):
                if switch in record:
                    entry[key] = {item: list(record[switch][item]) for item in
                                  ('run_receipts', 'bitexact_reports', 'component_reports') if item in record[switch]}
            for switch, key in sorted(maker.EXPERIMENTAL_DECISION_EVIDENCE.items()):
                if switch in record:
                    entry[key] = {item: list(values) for item, values in record[switch].items()}
        evidence_paths(validator, root, entry['evidence'], name + ' (carried f1 evidence)')
        decision[name] = entry
    _require(staged['profiles'][staged['default']]['qualified'] is True, 'the default profile must be qualified')
    validator.validate_profiles_document(staged)
    return staged, {'schema': 1, 'profiles': decision}


def lf(data):
    return data.replace(b'\r\n', b'\n')


def export_tree(source, clone):
    """Overlay this checkout's FreeToken trees on an upstream clone; returns {changed, new, deleted} counts."""
    source, clone = pathlib.Path(source).resolve(), pathlib.Path(clone).resolve()
    _require((clone / '.git').is_dir(), 'export clone is not a git checkout: ' + str(clone))
    counts = {'changed': 0, 'new': 0, 'deleted': 0}
    for top in EXPORT_TREES:
        origin, target = source / top, clone / top
        _require(origin.is_dir() and not origin.is_symlink(), 'source tree missing: ' + top)
        upstream = {}
        if target.exists():
            _require(target.is_dir() and not target.is_symlink(), 'clone tree is not a directory: ' + top)
            upstream = {path.relative_to(target).as_posix(): path.read_bytes()
                        for path in target.rglob('*') if path.is_file() and not path.is_symlink()}
            shutil.rmtree(target)
        seen = set()
        for path in sorted(origin.rglob('*')):
            if '__pycache__' in path.parts or path.suffix == '.pyc' or path.is_dir():
                continue
            _require(path.is_file() and not path.is_symlink(), 'refusing a non-regular source: ' + str(path))
            relative = path.relative_to(origin).as_posix()
            seen.add(relative)
            data = path.read_bytes()
            old = upstream.get(relative)
            if old is not None and lf(old) == lf(data):
                data = old
            else:
                counts['new' if old is None else 'changed'] += 1
                if path.suffix in TEXT_SUFFIXES:
                    data = lf(data)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        counts['deleted'] += len(set(upstream) - seen)
    return counts


def write_new(path, document):
    data = (json.dumps(document, indent=2, sort_keys=False, ensure_ascii=True, allow_nan=False) + '\n').encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, data)
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('decision', help='compose the staged profiles and the owner decision')
    p.add_argument('--validator', required=True, help='staged profile-runtime.py')
    p.add_argument('--profiles-in', required=True)
    p.add_argument('--f1-receipt', required=True, help='installed release/qualification.json (f1)')
    p.add_argument('--evidence-root', required=True, help='release.EVIDENCE_ROOT (ROOT/optimization)')
    for flag in ('--run', '--teacher', '--fla-component', '--overlap-component', '--ab', '--evidence', '--limitation',
                 '--long256-run', '--long256-teacher', '--long256-bitexact', '--long256-evidence',
                 '--long256-limitation'):
        p.add_argument(flag, action='append', default=[])
    p.add_argument('--limitations-file')
    p.add_argument('--long256-limitations-file')
    p.add_argument('--long256', choices=MODES['long256'], default='carry')
    p.add_argument('--baseline64', choices=MODES['baseline64'], default='carry')
    p.add_argument('--profiles-out')
    p.add_argument('--decision-out')
    e = sub.add_parser('export-tree', help='overlay the FreeToken trees on an upstream clone for the exporter')
    e.add_argument('--source', required=True, help='checkout freetoken/ directory')
    e.add_argument('--clone', required=True, help='upstream FreeToken clone at the pinned base')
    args = parser.parse_args(argv)
    if args.command == 'export-tree':
        try:
            print(json.dumps({'export_tree': export_tree(args.source, args.clone)}))
        except (OSError, ValueError) as error:
            raise SystemExit('release-main128 export-tree refused: ' + type(error).__name__ + ': ' + str(error)[:300]) from None
        return 0
    if bool(args.profiles_out) != bool(args.decision_out):
        parser.error('--profiles-out and --decision-out go together')
    try:
        validator = load_module('qwen36_release_main128_validator', args.validator)
        maker = load_module('qwen36_release_main128_maker', HERE / 'make-qualification.py')
        profiles = validator.read_json(pathlib.Path(args.profiles_in))
        f1 = validator.read_json(pathlib.Path(args.f1_receipt)) if args.long256 != 'drop' or args.baseline64 != 'drop' else {}
        staged, decision = compose(validator, maker, profiles, f1, args)
    except (OSError, ValueError) as error:
        raise SystemExit('release-main128 decision refused: ' + type(error).__name__ + ': ' + str(error)[:400]) from None
    summary = {'qualified': sorted(n for n, p in staged['profiles'].items() if p['qualified'] is True),
               'default': staged['default'],
               'evidence': {name: len(entry['evidence']) for name, entry in decision['profiles'].items()},
               'switch_evidence': {name: sorted(k for k in entry if k.endswith('_evidence'))
                                   for name, entry in decision['profiles'].items()}}
    if args.profiles_out:
        try:
            write_new(args.profiles_out, staged)
            write_new(args.decision_out, decision)
        except FileExistsError:
            raise SystemExit('release-main128 decision refused: output exists; never overwritten') from None
        summary['written'] = [args.profiles_out, args.decision_out]
    else:
        print(json.dumps({'profiles': staged, 'decision': decision}, indent=2))
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    sys.exit(main())
