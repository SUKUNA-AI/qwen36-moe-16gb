"""Opt-in, local-only teacher forcing. Never enabled in a serving profile.

The file contains public synthetic fixtures keyed by request seed. Results contain
only token losses and aggregate numeric logit differences, never text or tokens.

teacher-bf128 verifies the pinned logit reference byte for byte. Only with
QWEN36_TEACHER_CREATE_REFERENCE=1 (owner opt-in: new fixture or new stack) does it
create a missing reference file, exclusively; an existing file is never replaced.
"""
import hashlib
import json
import os
import pathlib
import torch
from freetoken.quality_evidence import (bind_teacher, evidence_records, strict_json,
    file_digest, digest, require)

CREATE_REFERENCE_ENV = 'QWEN36_TEACHER_CREATE_REFERENCE'

_fixtures = None
_counts = {}
_results = {}
_fixture_sha256 = None
_reference = None


def force(logits, batch):
    global _fixtures, _fixture_sha256, _reference
    path = os.environ.get('QWEN36_TEACHER_FIXTURE')
    if not path: return None
    if os.environ.get('QWEN36_PROFILE','') not in ('teacher-bf128','teacher-fp128','teacher-chunk8192',
            'teacher-cuda128','teacher-cuda-prefix128','teacher-cuda-both128','teacher-cuda8192-safe','teacher-cuda2048','teacher-cuda4096-safe',
            'teacher-cuda-matched128','teacher-b12x-fixed','teacher-main-candidate',
            'teacher-main-retain','teacher-main-retain2048','teacher-main-final','teacher-cuda-bitexact128-2048','teacher-cuda-bitexact128','teacher-cuda-bitexact128-c6144','teacher-cuda-bitexact128-c8192','teacher-bf128-fp16compat','teacher-long256-final-cuda2048','teacher-baseline64-final','teacher-main-layermajor','teacher-main-layermajor-c4096','teacher-long256-layermajor','teacher-main-layermajor-gdn','teacher-main-layermajor-fp8gemv','teacher-main-layermajor-gather','teacher-main-layermajor-refill','teacher-main-layermajor-refill-restore','teacher-main-combined','teacher-main-release','teacher-long256-release','teacher-main-candidate-r3','teacher-main-candidate-r2','teacher-main-combined-attn','teacher-long256-combined','teacher-main-release-shortroute','teacher-main-release-fla-wu','teacher-main-release-shortroute-fla-wu','teacher-main-release-overlap','teacher-main-release-shortroute-fla-wu-overlap',
            'teacher-main-release-next-he','teacher-main-release-next-r080','teacher-main-release-next-r082','teacher-main-release-next-r0845',
            'teacher-main-release-next-he-r085','teacher-main-release-next-he-r090','teacher-main-release-next-he-r094',
            'teacher-main-release-next-k','teacher-main-release-next-kc0','teacher-main-release-next-candidate',
            'teacher-main-release-next-d5','teacher-main-release-next-d5-d3b','teacher-main-release-next-all',
            'teacher-long256-release-next-candidate'):
        raise RuntimeError('Teacher forcing is restricted to explicit experiment profiles')
    if _fixtures is None:
        # Bind everything first: a failed binding must not leave a half-initialised module state.
        data = pathlib.Path(path).read_bytes()
        fixtures, fixture_sha256, reference = strict_json(data.decode())['targets'], hashlib.sha256(data).hexdigest(), None
        if os.environ['QWEN36_PROFILE'] != 'teacher-bf128':
            reference_path = os.environ.get('QWEN36_TEACHER_REFERENCE_REPORT')
            require(bool(reference_path), 'candidate teacher needs explicit reference report')
            report = strict_json(pathlib.Path(reference_path).read_text())
            require(report.get('fixture_sha256') == fixture_sha256, 'reference fixture mismatch')
            reference = bind_teacher(report, evidence_records(pathlib.Path(reference_path).parent/'teacher.jsonl', [report]), 'teacher-bf128')
        _fixtures, _fixture_sha256, _reference = fixtures, fixture_sha256, reference
    if batch.size != 1: raise RuntimeError('Teacher fixture expects one request')
    req = batch.reqs[0]
    from freetoken.scheduler.prefill import ChunkedReq
    if isinstance(req,ChunkedReq): return torch.zeros(1,device=logits.device,dtype=torch.int64)
    seed = str(req.sampling_params.seed)
    if seed not in _fixtures: return None
    entry = _fixtures[seed]
    offset = _counts.get(req.uid,0)
    if offset >= len(entry['ids']): return None
    target = entry['ids'][offset]
    row = logits[0].float()
    loss = torch.logsumexp(row,0)-row[target]
    # CPU sync is intentional in qualification only; exclude this profile from speed tests.
    if req.uid not in _results:
        receipt = strict_json((pathlib.Path(os.environ['QWEN36_DATA_ROOT'])/'active-run.json').read_text())
        require(receipt.get('run_id') == os.environ.get('QWEN36_RUN_ID') and
                receipt.get('profile') == os.environ['QWEN36_PROFILE'], 'active receipt identity mismatch')
        ref = ({'profile':'teacher-bf128', 'run_id':receipt['run_id'], 'case':entry['case'],
                'fixture_sha256':_fixture_sha256, 'targets_sha256':digest(entry['ids']), 'files':{}}
               if _reference is None else _reference[entry['case']]['reference'])
        require(ref['targets_sha256'] == digest(entry['ids']), 'reference targets mismatch')
        _results[req.uid] = dict(case=entry['case'], seed=int(seed), profile=os.environ['QWEN36_PROFILE'],
            run_id=receipt['run_id'], receipt_sha256=digest(receipt), fixture_sha256=_fixture_sha256,
            targets_sha256=digest(entry['ids']), reference=ref, nll=[])
    _results[req.uid]['nll'].append(float(loss))
    if os.environ.get('QWEN36_LOGIT_REFERENCE') and offset%8==0:
        import numpy as np
        root=pathlib.Path(os.environ['QWEN36_LOGIT_REFERENCE']);root.mkdir(mode=0o700,exist_ok=True)
        file=root/f'{seed}-{offset}.npy'
        values=row.detach().cpu().numpy()
        if os.environ['QWEN36_PROFILE']=='teacher-bf128':
            require(bool(np.isfinite(values).all()), 'nonfinite reference logits')
            if not file.is_file():
                require(os.environ.get(CREATE_REFERENCE_ENV) == '1',
                        'preexisting baseline logit reference required; '+CREATE_REFERENCE_ENV+'=1 creates a missing one')
                with open(file,'xb') as out:np.save(out,values)
                file.chmod(0o600)
            pinned=np.load(file,allow_pickle=False)
            require(pinned.shape == values.shape and pinned.dtype == values.dtype and
                    pinned.tobytes() == values.tobytes(), 'baseline differs from pinned reference bytes')
            _results[req.uid]['reference']['files'][str(offset)] = file_digest(file)
            _results[req.uid].setdefault('logit_comparisons',[]).append({
                'position':offset, 'max_abs':0.0, 'relative_l2':0.0, 'finite':True, 'byte_equal':True})
        else:
            require(file_digest(file) == _results[req.uid]['reference']['files'].get(str(offset)),
                    'logit reference file missing/replaced')
            ref=np.load(file, allow_pickle=False)
            require(ref.shape == values.shape and bool(np.isfinite(ref).all()), 'reference shape/nonfinite logits')
            error=values-ref
            _results[req.uid].setdefault('logit_comparisons',[]).append({
                'position':offset,'max_abs':float(np.max(np.abs(error))),
                'relative_l2':float(np.linalg.norm(error)/max(np.linalg.norm(ref),1e-12)),
                'finite':bool(np.isfinite(values).all() and np.isfinite(ref).all()),
                'byte_equal':bool(values.dtype == ref.dtype and values.tobytes() == ref.tobytes())})
    _counts[req.uid] = offset+1
    return torch.tensor([target],device=logits.device,dtype=torch.int64)


def complete(uid):
    _counts.pop(uid,None)
    row = _results.pop(uid,None)
    if row is None:return
    from freetoken.engine import telemetry
    telemetry.get_writer().write('teacher.jsonl',row)
