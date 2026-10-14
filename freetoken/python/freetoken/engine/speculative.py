"""Opt-in exact-distribution primitives and request-local prompt lookup.

No vocabulary restriction. The caller supplies final, normalized distributions
after the target sampler's temperature/top-k/top-p transforms.
"""
import math
import random


def validate_probabilities(p):
    if not p or any(not math.isfinite(x) or x < 0 for x in p):
        raise ValueError('Invalid probability vector')
    if abs(sum(p)-1) > 1e-6:raise ValueError('Probabilities must sum to one')


def pick(p,u):
    validate_probabilities(p)
    if not 0 <= u < 1:raise ValueError('Uniform must be in [0,1)')
    u *= math.fsum(p)
    total=0.
    for i,x in enumerate(p):
        total+=x
        if u < total:return i
    return max(i for i,x in enumerate(p) if x>0)


def verify(p,draft,q,u_accept,u_residual):
    """One rejection-sampling row; None means a point mass, never q=0."""
    validate_probabilities(p)
    if not 0 <= draft < len(p):raise ValueError('Draft is outside vocabulary')
    if not 0 <= u_accept < 1:raise ValueError('Invalid acceptance uniform')
    if not 0 <= u_residual < 1:raise ValueError('Invalid residual uniform')
    if q is None:
        q=[0.]*len(p);q[draft]=1.
    validate_probabilities(q)
    if len(q)!=len(p) or q[draft]<=0:raise ValueError('Draft must have positive q probability')
    p_mass,q_mass=math.fsum(p),math.fsum(q)
    p=[x/p_mass for x in p]
    q=[x/q_mass for x in q]
    if u_accept*q[draft] < p[draft]:return draft,True
    residual=[max(0.,a-b) for a,b in zip(p,q)]
    mass=sum(residual)
    if mass<=0:raise ArithmeticError('A zero-mass residual cannot be reached after rejection')
    return pick([x/mass for x in residual],u_residual),False


def uniform(seed,position,channel):
    from .sample import position_seed
    # Disjoint stream keys; no mutable global RNG and no rollback consumption.
    return random.Random(position_seed(seed ^ (channel << 56),position)).random()


class PromptLookup:
    def __init__(self,tokens,ngram=4):
        self.tokens=[];self.ends={};self.ngram=ngram
        self.extend(tokens)

    def extend(self,tokens):
        for token in tokens:
            self.tokens.append(int(token))
            n=len(self.tokens)
            if n>=self.ngram:self.ends[tuple(self.tokens[n-self.ngram:n])]=n

    def propose(self,history,window=3):
        # Query before indexing the newest suffix so it cannot match itself.
        history=list(map(int,history));n=len(history)
        if n<self.ngram:return []
        while len(self.tokens)<n-1:self.extend(history[len(self.tokens):n-1])
        end=self.ends.get(tuple(history[-self.ngram:]))
        result=[] if end is None else self.tokens[end:min(end+window,len(self.tokens))]
        self.extend(history[len(self.tokens):])
        return result


class CostController:
    def __init__(self):
        self.base=[];self.attempts=0;self.disabled=False
        self.draft_s=0.;self.verify_s=0.;self.replay_s=0.
        self.total_s=0.
        self.proposed=0;self.accepted=0;self.emitted=0

    def baseline(self,seconds):
        if len(self.base)<32:self.base.append(seconds)

    def observe(self,proposed,accepted,emitted,draft_s,verify_s,replay_s,total_s=None):
        self.attempts+=1;self.proposed+=proposed;self.accepted+=accepted;self.emitted+=emitted
        self.draft_s+=draft_s;self.verify_s+=verify_s;self.replay_s+=replay_s
        self.total_s+=draft_s+verify_s+replay_s if total_s is None else total_s
        if self.attempts>=3 and self.base:
            baseline=sum(self.base)/len(self.base)
            self.disabled=self.total_s/self.emitted > baseline*1.03

    def report(self):
        return {k:v for k,v in vars(self).items() if k!='base'}


def final_probs(logits,params):
    """Same FlashInfer temperature -> top-k -> top-p chain as Sampler.sample."""
    import torch
    import flashinfer.sampling as sampling
    temperature=max(params.temperature,1e-6)
    p=sampling.softmax(logits.float(),temperature)
    if params.top_k>=1:p=sampling.top_k_renorm_probs(p,params.top_k)
    if params.top_p<1:p=sampling.top_p_renorm_probs(p,max(params.top_p,1e-6))
    return p/p.sum(-1,keepdim=True)


def pick_gpu(p,u):
    """Inverse CDF on represented weights; also executable on CPU for qualification."""
    import torch
    if not 0 <= u < 1:raise ValueError('Uniform must be in [0,1)')
    weights=_tensor_weights(p)
    cdf=weights.cumsum(-1)
    # float32 rounds valid near-one uniforms to 1 and can leave a CDF below 1.
    # Never clamp the resulting index: that can select a zero-probability tail.
    total=cdf[-1]
    threshold=torch.minimum(total*u,torch.nextafter(total,torch.zeros_like(total)))
    return int(torch.searchsorted(cdf,threshold,right=True).item())


def _tensor_weights(p):
    import torch
    if p.ndim!=1 or p.numel()==0 or not p.is_floating_point():
        raise ValueError('Expected a nonempty floating-point probability vector')
    weights=p.to(torch.float64)
    if not bool((torch.isfinite(weights)&(weights>=0)).all().item()):
        raise ValueError('Invalid probability vector')
    mass=float(weights.sum().item())
    if mass<=0 or not math.isfinite(mass):raise ValueError('Probability mass must be positive')
    return weights


def verify_gpu(p,draft,q,u_accept,u_residual):
    if not 0 <= u_accept < 1 or not 0 <= u_residual < 1:
        raise ValueError('Uniforms must be in [0,1)')
    p=_tensor_weights(p);p=p/p.sum()
    if not 0 <= draft < len(p):raise ValueError('Draft is outside vocabulary')
    if q is not None:
        q=_tensor_weights(q)
        if q.shape!=p.shape:raise ValueError('Target and draft vocabularies differ')
        q=q/q.sum()
    pd=float(p[draft].item());qd=1. if q is None else float(q[draft].item())
    if qd<=0:raise ValueError('Invalid zero-probability draft')
    if u_accept*qd<pd:return draft,True
    if q is None:
        residual=p.clone();residual[draft]=0
    else:residual=(p-q).clamp_min(0)
    mass=residual.sum()
    if float(mass.item())<=0:raise ArithmeticError('Empty rejected residual')
    return pick_gpu(residual/mass,u_residual),False


def sampled_enabled(req):
    import os
    return req.sampling_params.is_greedy or os.environ.get('QWEN36_SPEC_SAMPLING')=='1'
