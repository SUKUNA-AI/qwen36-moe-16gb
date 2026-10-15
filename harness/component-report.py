"""Turn a pytest JUnit XML of a GPU component test into strict JSON qualification evidence.

python component-report.py --junit run.xml --suite freetoken/tests/moe/test_decode_overlap_gpu.py --out report.json
The report lists every case with pass/fail; skips and errors count as not passed. Refuses to overwrite.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import xml.etree.ElementTree as ET


def build(junit, suite_path, suite):
    cases = []
    for case in ET.parse(junit).getroot().iter('testcase'):
        outcome = 'passed'
        for child in case:
            if child.tag in ('failure', 'error', 'skipped'):
                outcome = child.tag
        cases.append({'case': case.get('name'), 'pass': outcome == 'passed', 'outcome': outcome})
    return {'schema': 1, 'kind': 'gpu-component-report', 'suite': suite,
            'suite_sha256': hashlib.sha256(pathlib.Path(suite_path).read_bytes()).hexdigest(),
            'cases': len(cases), 'skipped': sum(c['outcome'] == 'skipped' for c in cases),
            'all_pass': bool(cases) and all(c['pass'] for c in cases), 'results': cases}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--junit', required=True)
    p.add_argument('--suite', required=True, help='repository-relative test file')
    p.add_argument('--root', default=str(pathlib.Path(__file__).resolve().parent.parent))
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    report = build(a.junit, pathlib.Path(a.root) / a.suite, a.suite)
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as out:
        out.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('suite', 'cases', 'skipped', 'all_pass')}))
    return 0 if report['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
