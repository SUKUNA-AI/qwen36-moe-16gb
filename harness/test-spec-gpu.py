"""Small exact cases and distribution checks for the actual GPU verifier."""
import json,random,pathlib
import torch
from freetoken.engine.speculative import verify_gpu,pick_gpu,final_probs,pick
from freetoken.core import SamplingParams

torch.manual_seed(481)
rng=random.Random(411)
assert pick_gpu(torch.tensor([0.,1.,0.],device='cuda'),0.)==1
near_one=[]
for values in ([.2,.8,0.],[1.,0.,0.],[0.,.3,.7,0.]):
    for u in (1.-2**-53,1.-2**-30,0.):
        index=pick_gpu(torch.tensor(values,device='cuda'),u)
        assert values[index]>0,(values,u,index)
        near_one.append({'uniform':u,'selected_probability_positive':True})
try:pick_gpu(torch.tensor([1.,0.],device='cuda'),1.)
except ValueError:pass
else:raise AssertionError('Uniform outside [0,1) was accepted')
checks=[]
for probs,draft_probs in [([.1,.2,.7],[.7,.2,.1]),([1.,0.,0.],[0.,1.,0.]),([.3,.7],[.3,.7])]:
    p=torch.tensor(probs,device='cuda');q=torch.tensor(draft_probs,device='cuda');counts=[0]*len(probs)
    for _ in range(5000):
        draft=pick(draft_probs,rng.random())
        result,_=verify_gpu(p,draft,q,rng.random(),rng.random());counts[result]+=1
    error=max(abs(x/5000-y) for x,y in zip(counts,probs));assert error<.03
    checks.append({'maximum_probability_error':error,'samples':5000})
logits=torch.tensor([[1.,2.,-3.,4.]],device='cuda')
for k,top_p in [(-1,1.),(2,1.),(2,.8),(-1,.8)]:
    p=final_probs(logits,SamplingParams(temperature=.7,top_k=k,top_p=top_p))
    assert abs(p.sum().item()-1)<1e-6 and (p>=0).all()
    if k==2:assert int((p>0).sum())<=2
report={'pass':True,'known_distributions':checks,'zero_probability_boundary':True,'sampling_transform_smoke':True,
        'near_one':near_one,'scheduler_qualification':False}
path=pathlib.Path('/home/USER/qwen36-api/optimization/reports/spec-gpu.json');path.write_text(json.dumps(report,indent=2));print(json.dumps(report))
