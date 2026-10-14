import math,random
from freetoken.engine.speculative import verify,pick,PromptLookup,CostController
import torch
from freetoken.engine.speculative import pick_gpu,verify_gpu,uniform

def test_known_distributions():
    rng=random.Random(1962)
    for p,q in [([.1,.2,.7],[.7,.2,.1]),([1.,0.,0.],[0.,1.,0.]),
                ([.3,.7],[.3,.7]),([0.,.4,.6],[.2,.3,.5])]:
        counts=[0]*len(p);n=100000
        for _ in range(n):
            d=pick(q,rng.random());out,_=verify(p,d,q,rng.random(),rng.random());counts[out]+=1
        assert max(abs(c/n-x) for c,x in zip(counts,p))<.008

def test_point_mass_and_zero_support():
    rng=random.Random(815)
    p=[.05,.8,.15];counts=[0]*3
    for _ in range(100000):
        out,_=verify(p,0,None,rng.random(),rng.random());counts[out]+=1
    assert max(abs(c/100000-x) for c,x in zip(counts,p))<.008
    assert verify([0.,1.],0,None,0.,.5)==(1,False)
    assert verify([1.,0.],0,None,.999,.5)==(0,True)

def test_lookup_without_self_match():
    lookup=PromptLookup([1,2,3,4,5,6,7,8,99,1,2,3])
    assert lookup.propose([1,2,3,4,5,6,7,8,99,1,2,3,4],3)==[5,6,7]
    assert PromptLookup([1,2,3]).propose([1,2,3,4])==[]

def test_cost_disable():
    c=CostController();c.baseline(.01)
    for _ in range(3):c.observe(3,1,2,.001,.04,.01)
    assert c.disabled and c.accepted==3 and c.emitted==6


def test_tensor_cdf_boundaries_and_zero_intervals():
    near_one=math.nextafter(1.,0.)
    cases=[([1.,0.],near_one,0),([.25,.75,0.,0.],near_one,1),
           ([0.,.25,0.,.75,0.],0.,1),([0.,.25,0.,.75,0.],.25,3),
           ([0.,.25,0.,.75,0.],math.nextafter(.25,0.),1),
           ([0.,.25,0.,.75,0.],near_one,3),
           ([.1,.2,.6999999,0.],near_one,2)]
    for values,u,expected in cases:
        p=torch.tensor(values,dtype=torch.float32)
        assert pick_gpu(p,u)==expected,(values,u,expected)
        assert pick(values,u)==expected
    # This is an actual float32 CDF gap, not just rounded uniform input.
    assert torch.tensor(cases[-1][0]).cumsum(0)[-1] < 1


def test_tensor_rejection_uses_positive_residual_support():
    near_one=math.nextafter(1.,0.)
    p=torch.tensor([.25,.75,0.,0.]);q=torch.tensor([1.,0.,0.,0.])
    assert verify_gpu(p,0,q,near_one,near_one)==(1,False)
    assert verify_gpu(p,0,None,.25,near_one)==(1,False)
    assert verify_gpu(p,0,None,math.nextafter(.25,0.),near_one)==(0,True)
    assert verify_gpu(torch.tensor([0.,1.,0.]),0,None,0.,near_one)==(1,False)
    assert verify_gpu(p,1,p,near_one,near_one)==(1,True)


def test_tensor_sampler_exact_stratified_distributions():
    # Exhaustive equal-spaced uniforms isolate each acceptance/rejection branch.
    n=1000
    for values in ([.1,.2,.7,0.],[0.,.25,0.,.75,0.],[1.,0.,0.]):
        p=torch.tensor(values,dtype=torch.float32);counts=[0]*len(values)
        for i in range(n):counts[pick_gpu(p,(i+.5)/n)]+=1
        assert max(abs(c/n-v) for c,v in zip(counts,values))<=1/n
    p=torch.tensor([.25,.5,.25,0.]);q=torch.tensor([.5,.25,.25,0.])
    counts=[0.]*4
    # Integrate draft, acceptance and residual separately; known exact target.
    for draft,qd in enumerate(q.tolist()):
        if not qd:continue
        for ia in range(20):
            for ir in range(20):
                out,_=verify_gpu(p,draft,q,(ia+.5)/20,(ir+.5)/20)
                counts[out]+=qd/400
    assert max(abs(a-b) for a,b in zip(counts,p.tolist()))<1e-12


def test_tensor_sampler_rejects_invalid_inputs():
    for values in ([],[0.,0.],[-.1,1.1],[float('nan'),1.],[float('inf'),0.]):
        try:pick_gpu(torch.tensor(values,dtype=torch.float32),.5)
        except ValueError:pass
        else:raise AssertionError(values)
    for u in (-.1,1.,float('nan'),float('inf')):
        try:pick_gpu(torch.tensor([1.,0.]),u)
        except ValueError:pass
        else:raise AssertionError(u)
        try:verify_gpu(torch.tensor([1.,0.]),0,None,.5,u)
        except ValueError:pass
        else:raise AssertionError(u)
    try:verify_gpu(torch.tensor([.5,.5]),0,torch.tensor([0.,1.]),.2,.3)
    except ValueError:pass
    else:raise AssertionError('zero-probability draft accepted')


def test_request_local_uniform_is_replayable_and_rng_independent():
    rng_state=random.getstate();torch_state=torch.random.get_rng_state().clone()
    expected=[uniform(71,position,channel) for position in range(7,12) for channel in range(4)]
    assert random.getstate()==rng_state
    assert torch.equal(torch.random.get_rng_state(),torch_state)
    random.seed(5)
    for _ in range(100):random.random()
    actual=[uniform(71,position,channel) for position in range(7,12) for channel in range(4)]
    assert actual==expected and len(set(actual))==len(actual)
    assert all(0<=u<1 for u in actual)
    assert [uniform(72,p,0) for p in range(7,12)]!=expected[::4]
