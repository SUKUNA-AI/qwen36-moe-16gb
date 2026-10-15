"""Mandatory CPU suites only. Child isolation also prevents test-module stub leaks."""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from repo_paths import CHECKOUT, FREETOKEN, SERVING, HARNESS

# Private eval-v1 cases/sandbox are absent from the public checkout.
PRIVATE = {'test-eval-v1-cpu.py'}

def command_for_suite(f):
    if f.suffix == '.sh':
        return ['bash', str(f)]
    cmd = [sys.executable, str(f)]
    if f.name in ('test-tool-choice-guard-cpu.py', 'test-tool-semantics-guard-cpu.py'):
        cmd += ['--guard', os.environ.get('QWEN36_GUARD_UNDER_TEST', str(SERVING/'guard.py'))]
    if f.name == 'test-tool-semantics-guard-cpu.py':
        cmd += ['--gate-script', str(HARNESS/'cpu_checks.py')]
    return cmd


def counts(log, name):
    match = re.search(r'Ran (\d+) tests?', log)
    if match:
        total = int(match[1])
        fields = {k:int(v) for k,v in re.findall(r'(failures|errors|skipped)=(\d+)', log)}
        return total, fields.get('failures',0)+fields.get('errors',0), fields.get('skipped',0)
    docs=[]
    decoder=json.JSONDecoder()
    for i,ch in enumerate(log):
        if ch != '{':continue
        try:
            doc,_=decoder.raw_decode(log[i:])
            if isinstance(doc,dict):docs.append(doc)
        except ValueError:pass
    for doc in reversed(docs):
        total=next((doc[k] for k in ('case_count','test_count','count','tests_run') if type(doc.get(k)) is int),None)
        if total is None and isinstance(doc.get('tests'),list):total=len(doc['tests'])
        if total is not None:return total,doc.get('failed_count',0),doc.get('skipped',0)
    # These legacy scripts each execute one integrated scenario with multiple asserts.
    if name in ('test-guard.py','test-metrics.py') and any(d.get('result') == 'PASS' for d in docs):
        return 1,0,0
    if name == 'test-log-failure.py' and any(all(d.get(k) is True for k in ('bounded_queue','disk_full_isolated','writer_recovers','threads_joined')) for d in docs):
        return 1,0,0
    return 0,0,0

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True, help='CPU logs/summary directory')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',os.defpath), QWEN36_CPU_REPORT_DIR=str(a.output.resolve()), CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', TVM_FFI_DISABLE_TORCH_C_DLPACK='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join(map(str, (FREETOKEN/'python', SERVING, HARNESS))))
    # Script-tests execute at import or in __main__; default pytest discovery misses them.
    suites = [f for f in sorted(HARNESS.glob('test-*-cpu.py')) if f.name not in PRIVATE]
    suites += sorted(HARNESS.glob('test-*-cpu.sh'))
    suites += [SERVING/'test-guard.py', SERVING/'test-metrics.py', HARNESS/'test-log-failure.py']
    rows = []
    for f in suites:
        cmd = command_for_suite(f)
        try:
            r = subprocess.run(cmd, env=env, cwd=a.output.resolve(), capture_output=True, text=True, timeout=180)
            log = r.stdout + r.stderr
            rc = r.returncode
        except subprocess.TimeoutExpired as e:
            log = str(e); rc = 124
        (a.output/(f.name+'.log')).write_text(log)
        total, failed, skipped = counts(log, f.name)
        if rc == 0 and total <= skipped:
            rc = 3
        rows.append({'suite': f.name, 'exit_code': rc, 'executed': total-skipped, 'passed':total-failed-skipped if rc == 0 else None,
                     'failed':failed, 'skipped':skipped})
        print(f'{f.name}: {"PASS" if rc == 0 else "FAIL"} (exit {rc})', flush=True)
    summary = {'python': sys.version, 'checkout': str(CHECKOUT), 'suites': rows,
               'not_run': {'private-fixture': sorted(PRIVATE), 'Windows/WSL': 'pwsh/owner runtime required',
                           'GPU/model': 'CPU runner excludes GPU suites', 'CUDA compile-only': 'separate toolchain check'},
               'pass': sum(r['exit_code'] == 0 for r in rows), 'fail': sum(r['exit_code'] != 0 for r in rows)}
    summary['executed'] = sum(r['executed'] for r in rows)
    summary['skipped'] = sum(r['skipped'] for r in rows)
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return int(not rows or summary['fail'] > 0)

if __name__ == '__main__':
    raise SystemExit(main())
