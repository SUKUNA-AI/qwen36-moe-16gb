"""Opt-in, local-only teacher forcing. Never enabled in a serving profile.

The file contains public synthetic fixtures keyed by request seed. Results contain
only token losses and aggregate numeric logit differences, never text or tokens.
"""
import json
import os
import pathlib
import torch

_fixtures = None
_counts = {}
_results = {}


def force(logits, batch):
    global _fixtures
    path = os.environ.get('QWEN36_TEACHER_FIXTURE')
    if not path: return None
    if os.environ.get('QWEN36_PROFILE','') not in ('teacher-bf128','teacher-fp128','teacher-chunk8192',
            'teacher-cuda128','teacher-cuda-prefix128','teacher-cuda-both128','teacher-cuda8192-safe','teacher-cuda2048','teacher-cuda4096-safe',
            'teacher-cuda-matched128','teacher-b12x-fixed','teacher-main-candidate',
            'teacher-main-retain','teacher-main-retain2048','teacher-main-final','teacher-cuda-bitexact128-2048','teacher-cuda-bitexact128','teacher-cuda-bitexact128-c6144','teacher-cuda-bitexact128-c8192','teacher-bf128-fp16compat','teacher-long256-final-cuda2048','teacher-baseline64-final','teacher-main-layermajor','teacher-main-layermajor-c4096','teacher-long256-layermajor','teacher-main-layermajor-gdn','teacher-main-layermajor-fp8gemv','teacher-main-layermajor-gather','teacher-main-layermajor-refill','teacher-main-layermajor-refill-restore','teacher-main-combined','teacher-main-release','teacher-long256-release','teacher-main-candidate-r3','teacher-main-candidate-r2','teacher-main-combined-attn','teacher-long256-combined'):
        raise RuntimeError('Teacher forcing is restricted to explicit experiment profiles')
    if _fixtures is None:
        _fixtures = json.loads(pathlib.Path(path).read_text())['targets']
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
    _results.setdefault(req.uid,dict(case=entry['case'],seed=int(seed),profile=os.environ['QWEN36_PROFILE'],
        run_id=os.environ.get('QWEN36_RUN_ID'),nll=[]))['nll'].append(float(loss))
    if os.environ.get('QWEN36_LOGIT_REFERENCE') and offset%8==0:
        import numpy as np
        root=pathlib.Path(os.environ['QWEN36_LOGIT_REFERENCE']);root.mkdir(mode=0o700,exist_ok=True)
        file=root/f'{seed}-{offset}.npy'
        values=row.detach().cpu().numpy()
        if os.environ['QWEN36_PROFILE']=='teacher-bf128':
            np.save(file,values);file.chmod(0o600)
        else:
            ref=np.load(file)
            error=values-ref
            _results[req.uid].setdefault('logit_comparisons',[]).append({
                'position':offset,'max_abs':float(np.max(np.abs(error))),
                'relative_l2':float(np.linalg.norm(error)/max(np.linalg.norm(ref),1e-12)),
                'finite':bool(np.isfinite(values).all())})
    _counts[req.uid] = offset+1
    return torch.tensor([target],device=logits.device,dtype=torch.int64)


def complete(uid):
    _counts.pop(uid,None)
    row = _results.pop(uid,None)
    if row is None:return
    from freetoken.engine import telemetry
    telemetry.get_writer().write('teacher.jsonl',row)
