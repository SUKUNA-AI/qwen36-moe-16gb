"""Unquantized native Qwen3.6 MTP reference, isolated behind explicit opt-in.

Architecture checked against vLLM 0fae39a0142c5d296f7938fa753e8f00b5d80171,
qwen3_5_mtp.py. Loads only the 19 mtp tensors from the existing checkpoint;
embedding and output head are borrowed from the target. No target weights change.
"""
import json,pathlib
import torch
import torch.nn.functional as F


class NativeMTP:
    def __init__(self,path,embed,head,context,device='cuda'):
        from safetensors import safe_open
        path=pathlib.Path(path)
        config=json.loads((path/'config.json').read_text())['text_config']
        index=json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
        names=[k for k in index if k.startswith('mtp.')]
        if len(names)!=19:raise ValueError('Unexpected native MTP tensor inventory')
        self.weights={}
        for shard in sorted({index[k] for k in names}):
            with safe_open(path/shard,framework='pt',device='cpu') as f:
                for name in names:
                    if index[name]!=shard:continue
                    t=f.get_tensor(name)
                    if t.dtype!=torch.bfloat16:raise ValueError('Expected native BF16 MTP weights')
                    self.weights[name[4:]]=t.to(device)
        self.embed,self.head=embed,head
        self.eps=config['rms_norm_eps'];self.h=config['hidden_size'];self.d=config['head_dim']
        self.nq=config['num_attention_heads'];self.nkv=config['num_key_value_heads']
        self.topk=config['num_experts_per_tok']
        self.rotary=int(self.d*config['partial_rotary_factor'])
        self.theta=config['rope_parameters']['rope_theta']
        self.k=torch.empty(context,self.nkv,self.d,device=device,dtype=torch.bfloat16)
        self.v=torch.empty_like(self.k);self.length=0
        self.inv_freq=self.theta**(-torch.arange(0,self.rotary,2,device=device,dtype=torch.float32)/self.rotary)
        self.weight_bytes=sum(t.numel()*t.element_size() for t in self.weights.values())

    def norm(self,x,name):
        y=x.float();y=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+self.eps)
        return (y*(1.+self.weights[name+'.weight'].float())).to(x.dtype)

    def linear(self,x,name):return F.linear(x,self.weights[name+'.weight'])

    def rope(self,x,positions):
        phase=positions.float()[:,None]*self.inv_freq[None,:]
        cos,sin=phase.cos()[:,None,:],phase.sin()[:,None,:]
        a,b=x[...,:self.rotary//2].float(),x[...,self.rotary//2:self.rotary].float()
        return torch.cat(((a*cos-b*sin).to(x.dtype),(b*cos+a*sin).to(x.dtype),x[...,self.rotary:]),-1)

    def mlp(self,x):
        prefix='layers.0.mlp.'
        router=self.linear(x,prefix+'gate').float().softmax(-1)
        weight,ids=router.topk(self.topk,dim=-1);weight=weight/weight.sum(-1,keepdim=True)
        gu=self.weights[prefix+'experts.gate_up_proj'];down=self.weights[prefix+'experts.down_proj']
        result=torch.zeros_like(x)
        if len(x)==1:
            selected=ids[0]
            projected=torch.bmm(gu[selected],x.expand(self.topk,-1).unsqueeze(-1)).squeeze(-1)
            gate,up=projected.chunk(2,-1)
            routed=torch.bmm(down[selected],(F.silu(gate)*up).unsqueeze(-1)).squeeze(-1)
            result=(routed.float()*weight[0,:,None]).sum(0,keepdim=True).to(x.dtype)
        else:
            # Reference prefill: no M*K copies of the entire expert banks.
            for expert in range(gu.shape[0]):
                rows,ranks=torch.where(ids==expert)
                if rows.numel()==0:continue
                gate,up=F.linear(x[rows],gu[expert]).chunk(2,-1)
                y=F.linear(F.silu(gate)*up,down[expert])
                result.index_add_(0,rows,(y.float()*weight[rows,ranks,None]).to(x.dtype))
        gate=self.linear(x,prefix+'shared_expert.gate_proj')
        up=self.linear(x,prefix+'shared_expert.up_proj')
        shared=self.linear(F.silu(gate)*up,prefix+'shared_expert.down_proj')
        return result+shared*self.linear(x,prefix+'shared_expert_gate').sigmoid()

    @torch.inference_mode()
    def forward(self,input_ids,hidden,positions,*,commit=True):
        embedding=self.embed(input_ids)
        x=self.linear(torch.cat((self.norm(embedding,'pre_fc_norm_embedding'),
                                 self.norm(hidden,'pre_fc_norm_hidden')),-1),'fc')
        residual=x;x=self.norm(x,'layers.0.input_layernorm')
        p='layers.0.self_attn.'
        qg=self.linear(x,p+'q_proj').view(-1,self.nq,2*self.d)
        q,gate=qg[...,:self.d],qg[...,self.d:]
        k=self.linear(x,p+'k_proj').view(-1,self.nkv,self.d)
        v=self.linear(x,p+'v_proj').view(-1,self.nkv,self.d)
        q=self.rope(self.norm(q,p+'q_norm'),positions)
        k=self.rope(self.norm(k,p+'k_norm'),positions)
        end=self.length+len(x)
        if end>len(self.k):raise ValueError('MTP KV budget exceeded')
        self.k[self.length:end].copy_(k);self.v[self.length:end].copy_(v)
        q=q.transpose(0,1).unsqueeze(0)
        key=self.k[:end].transpose(0,1).unsqueeze(0);value=self.v[:end].transpose(0,1).unsqueeze(0)
        if len(x)==1:
            attn=F.scaled_dot_product_attention(q,key,value,enable_gqa=True)
        else:
            # Explicit offset causal mask: SDPA is_causal aligns at the top left.
            rows=torch.arange(self.length,end,device=x.device)[:,None]
            mask=torch.arange(end,device=x.device)[None,:]<=rows
            attn=F.scaled_dot_product_attention(q,key,value,attn_mask=mask,enable_gqa=True)
        attn=attn.squeeze(0).transpose(0,1)*gate.sigmoid()
        x=residual+self.linear(attn.reshape(-1,self.nq*self.d),p+'o_proj')
        x=x+self.mlp(self.norm(x,'layers.0.post_attention_layernorm'))
        hidden=self.norm(x,'norm')
        if commit:self.length=end
        return self.head(hidden),hidden

    def truncate(self,length):
        if not 0<=length<=self.length:raise ValueError('Invalid MTP rollback length')
        self.length=length


def observe_target(engine,batch):
    from .speculative import sampled_enabled
    req=batch.reqs[0]
    if req.uid<0 or not sampled_enabled(req):return
    if engine._mtp_uid!=req.uid:
        engine._mtp_uid=req.uid;engine._mtp_previous=None;engine.native_mtp.length=0
        # Prefix hits have target GDN/KV snapshots but no MTP snapshot yet. Exact
        # target decoding remains available; this request cannot use MTP drafts.
        engine._mtp_eligible=req.cached_len==0
    if not engine._mtp_eligible or getattr(getattr(req,'spec_cost',None),'disabled',False):return
    hidden=engine.model.mtp_hidden[:len(batch.input_ids)]
    if engine._mtp_previous is None:
        ids=batch.input_ids[1:];prev=hidden[:-1];positions=batch.positions[1:]
    else:
        ids=batch.input_ids;prev=torch.cat((engine._mtp_previous,hidden[:-1]),0);positions=batch.positions
    if len(ids):engine.native_mtp.forward(ids,prev,positions)
    engine._mtp_previous=hidden[-1:].clone()
