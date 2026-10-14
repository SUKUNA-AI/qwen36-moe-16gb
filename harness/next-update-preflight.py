"""Read-only owner preflight. Does not start units, sync files, or create receipts."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from freetoken.quality_evidence import strict_json, evidence_records, bind_teacher, file_digest

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True,help='reviewed experiment overlay, never production release/python')
    p.add_argument('--python',type=Path,required=True,help='owner GPU venv Python')
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference-report',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--wsl-interop',action='store_true',help='also require Windows interop for later Windows release tooling')
    a=p.parse_args(argv);errors=[]
    if not a.python.is_file():errors.append('owner Python missing')
    if not (a.model/'config.json').is_file():errors.append('local model/checkpoint missing')
    for name in ('start-experiment.sh','check-experiment.py','quality.py','compare-quality.py','bench.py','matrix.py',
                 'repo_paths.py','code_runner.py','python/freetoken/quality_evidence.py'):
        if not (a.native/name).is_file():errors.append('experiment overlay missing '+name)
    for name in ('run-experiment.sh','experiment.py'):
        file=a.native/name
        if file.is_file() and any(marker in file.read_text() for marker in ('/home/USER/','/path/to/qwen36-workspace','/path/to/LLM/')):
            errors.append('unconfigured site placeholder in '+name)
    try:
        report=strict_json(a.reference_report.read_text())
        selected=bind_teacher(report,evidence_records(a.reference_report.parent/'teacher.jsonl',[report]),'teacher-bf128')
        reference_files=sum(len(r['reference'].get('files',{})) for r in selected.values())
        expected=sum((len(r['nll'])+7)//8 for r in selected.values())
        if reference_files != expected or expected <= 0:
            raise ValueError('reference checkpoint set incomplete')
        for row in selected.values():
            ref=row['reference']
            if (ref.get('profile') != 'teacher-bf128' or ref.get('run_id') != row['run_id'] or
                ref.get('case') != row['case'] or ref.get('fixture_sha256') != row['fixture_sha256'] or
                ref.get('targets_sha256') != row['targets_sha256'] or
                set(ref['files']) != {str(p) for p in range(0,len(row['nll']),8)}):
                raise ValueError('reference identity differs')
            for position,sha in ref['files'].items():
                if file_digest(a.reference_dir/f'{row["seed"]}-{position}.npy') != sha:
                    errors.append('pinned reference file differs')
    except (OSError,ValueError,KeyError,TypeError) as e:
        errors.append('run-bound reference unavailable: '+str(e));reference_files=0
    if a.wsl_interop and (not Path('/proc/sys/fs/binfmt_misc/WSLInterop').exists() or not shutil.which('cmd.exe')):
        errors.append('WSL Windows interop unavailable; do not run release-management.ps1')
    for unit in ('qwen36-inference.service','qwen36-experiment.service'):
        try:
            r=subprocess.run(['systemctl','--user','show',unit,'--property=ActiveState','--value'],capture_output=True,text=True,timeout=5)
            if r.returncode != 0 or r.stdout.strip() not in ('inactive','failed'):
                errors.append(unit+' is not verified inactive')
        except (OSError,subprocess.TimeoutExpired) as e:errors.append(unit+': '+type(e).__name__)
    print(json.dumps({'mode':'preflight_only','pass':not errors,'errors':errors,'reference_checkpoints':reference_files,
                      'gpu_execution':'NOT_RUN','receipt_created':False},indent=2))
    return int(bool(errors))
if __name__=='__main__':raise SystemExit(main())
