"""CPU identity audit of margin snapshots, carried marks, radix donation and COW.

Synthetic GDN states encode SHA-256(prefix tokens) and prefix length as exactly
representable byte values. This tests metadata/ownership, not GDN arithmetic.
"""
import hashlib
import struct
from types import SimpleNamespace
from unittest.mock import patch

import torch

from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch, Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq


def fingerprint(ids):
    values=ids.tolist() if isinstance(ids,torch.Tensor) else list(ids)
    digest=hashlib.sha256(struct.pack('<'+'i'*len(values),*values)).digest()
    return torch.tensor(list(digest+struct.pack('<I',len(values))),dtype=torch.float32)


def stamp(tensor,ids):
    tensor.zero_();flat=tensor.flatten();mark=fingerprint(ids)
    flat[:len(mark)]=mark.to(flat.dtype)


def assert_stamp(tensor,ids):
    assert torch.equal(tensor.flatten()[:36].float(),fingerprint(ids)), \
        f'snapshot does not represent its {len(ids)}-token radix prefix'


def pool(slots=16):
    group=LinearGatedDeltaGroupConfig(name='linear',layer_ids=(0,),num_key_heads=2,
        num_value_heads=4,key_head_dim=16,value_head_dim=16,conv_kernel_dim=4,output_gate='silu')
    return LinearStatePool(group,slots,torch.bfloat16,torch.device('cpu'),tp_size=1)


def fake_forward(batch,p):
    """Feed real snapshot-copy code states at each chunk START, as FLA h stores them."""
    batch.padded_reqs=batch.reqs
    with patch('freetoken.core.get_global_ctx',return_value=SimpleNamespace(linear_state_pool=p)):
        fla=build_fla_metadata(batch,torch.device('cpu'))
    lengths=[r.extend_len for r in batch.reqs]
    chunks=[(n+63)//64 for n in lengths]
    h=torch.empty((1,sum(chunks),4,16,16),dtype=torch.bfloat16)
    conv_dim=p.conv_states.shape[2]
    conv=[];offset=0
    for req,nchunks in zip(batch.reqs,chunks):
        ids=req.input_ids
        if req.mamba_restore_src is not None:
            p.copy_from(req.mamba_restore_src,req.linear_slot_idx)
            assert_stamp(p.recurrent_states[0,req.linear_slot_idx],ids[:req.cached_len])
            req.mamba_restore_src=None
        for j in range(nchunks):stamp(h[0,offset+j],ids[:req.cached_len+j*64])
        offset+=nchunks
        features=(ids[req.cached_len:req.device_len]%251).to(torch.bfloat16)
        conv.append(features[:,None].expand(-1,conv_dim))
        stamp(p.recurrent_states[0,req.linear_slot_idx],ids[:req.device_len])
        last=(ids[:req.device_len][-3:]%251).to(torch.bfloat16)
        p.conv_states[0,req.linear_slot_idx].zero_()
        p.conv_states[0,req.linear_slot_idx,:,-len(last):]=last
    if fla.track_dst is not None:
        Qwen3_5GatedDeltaNet._write_track_snapshot(None,p,0,torch.cat(conv),h,fla)
        for req in batch.reqs:
            if req.mamba_last_track_seqlen is None:continue
            length=req.mamba_last_track_seqlen
            slot=req.mamba_ping_pong[1-req.mamba_next_track_idx]
            assert_stamp(p.recurrent_states[0,slot],req.input_ids[:length])
            expected=(req.input_ids[length-3:length]%251).to(torch.bfloat16)
            assert torch.equal(p.conv_states[0,slot,0],expected)
    return fla


class History:
    def __init__(self,slots=16):
        self.pool=pool(slots)
        self.pages=torch.zeros((2,18000),dtype=torch.int32)
        self.cm=CacheManager(40000,1,self.pages,'hybrid_radix',linear_state_pool=self.pool)
        self.tm=TableManager(2,self.pages)
        self.dm=DecodeManager(1)
        self.pm=PrefillManager(self.cm,self.tm,self.dm)
        self.uid=0

    def audit_tree(self):
        for node in self.cm.prefix_cache._snapshot_nodes():
            keys=[];cursor=node
            while not cursor.is_root():keys.append(cursor._key);cursor=cursor.parent
            ids=torch.cat(list(reversed(keys)))
            assert_stamp(self.pool.recurrent_states[0,node.mamba_value],ids)
            expected=(ids[-3:]%251).to(torch.bfloat16)
            assert torch.equal(self.pool.conv_states[0,node.mamba_value,0,-len(expected):],expected)
        self.cm.check_integrity()

    def prefill(self,ids,budget,margin):
        self.uid+=1
        pending=PendingReq(self.uid,ids,SamplingParams(max_tokens=4))
        self.pm.pending_list.append(pending)
        records=[]
        with patch.dict('os.environ',{'QWEN36_PREFIX_MARGIN':str(margin)}):
            while self.pm.runnable:
                batch=self.pm.schedule_next_batch(budget);assert batch is not None
                req=batch.reqs[0]
                if not records:
                    assert req.mamba_last_track_seqlen is None
                    if req.cached_len:assert req.mamba_restore_src is not None
                self.cm.allocate_paged([req])
                inherited=req.mamba_last_track_seqlen
                fla=fake_forward(batch,self.pool)
                records.append(dict(start=req.cached_len,end=req.device_len,
                    inherited=inherited,tracked=req.mamba_last_track_seqlen,
                    new_snapshot=fla.track_dst is not None))
                req.complete_one()
                if not isinstance(req,ChunkedReq):
                    req.append_host(torch.tensor([90001],dtype=torch.int32))
                    self.cm.cache_req(req,finished=False)
        return req,records

    def decode(self,req,token):
        self.cm.snapshot_toolcall_anchor([req])
        self.cm.allocate_paged([req])
        stamp(self.pool.recurrent_states[0,req.linear_slot_idx],req.input_ids[:req.device_len])
        self.pool.conv_states[0,req.linear_slot_idx]=(req.input_ids[req.device_len-3:req.device_len]%251).to(torch.bfloat16)
        req.complete_one();req.append_host(torch.tensor([token],dtype=torch.int32))

    def finish(self,req):
        index=req.table_idx
        self.cm.cache_req(req,finished=True)
        self.tm.free(index);req.table_idx=-1
        self.audit_tree()


def test_margin_carry_through_short_final_chunk_has_exact_prefix_identity():
    for margin in (1,8):
        for budget in (128,2048,4096,8192):
            for tail in (1,7,8,63,64,65,71,72):
                history=History()
                ids=torch.arange(budget+tail,dtype=torch.int32)+31
                req,records=history.prefill(ids,budget,margin)
                assert records[0]['tracked']==budget-64
                if tail<64+margin:
                    assert not records[-1]['new_snapshot']
                    assert records[-1]['inherited']==(budget-64 if margin>1 else None)
                history.finish(req)


def test_unrelated_requests_dedup_recycling_and_real_prefix_hits():
    history=History(slots=9)
    base=torch.arange(4099,dtype=torch.int32)+20
    first,records=history.prefill(base,4096,8)
    assert records[-1]['tracked']==4032
    history.finish(first)
    # A new request sharing the frozen prefix restores the exact fingerprint.
    follow=base.clone();follow[4032:]+=100000
    second,records=history.prefill(follow,4096,8)
    assert records[0]['start']==4032
    history.finish(second)
    # Force slot reuse/eviction with mutually unrelated roots and short prompts.
    for i in range(20):
        ids=torch.arange(53+i*3,dtype=torch.int32)+1000000*(i+1)
        req,records=history.prefill(ids,128,8)
        assert records[0]['start']==0 and records[0]['inherited'] is None
        history.finish(req)


def test_tool_anchor_donation_tracks_consumed_token_not_later_state():
    history=History();ids=torch.arange(259,dtype=torch.int32)+10
    req,_=history.prefill(ids,256,8)
    assert req.mamba_last_track_seqlen is None
    req.toolcall_anchor_len=req.input_ids.numel()
    anchor=req.toolcall_anchor_len
    history.decode(req,90002)
    assert req.mamba_last_track_seqlen is None
    history.decode(req,90003)
    assert req.mamba_last_track_seqlen==anchor
    anchor_ids=req.input_ids[:anchor].clone()
    history.finish(req)
    rewritten=torch.cat((anchor_ids,torch.tensor([123456],dtype=torch.int32)))
    match=history.cm.match_req(SimpleNamespace(input_ids=rewritten,input_len=len(rewritten)))
    assert match.cuda_handle.cached_len==anchor
    assert_stamp(history.pool.recurrent_states[0,match.mamba_value],anchor_ids)


def test_batch_chunk_offsets_and_nonzero_cached_positions():
    p=pool();reqs=[]
    for uid,(cached,extend) in enumerate(((17,137),(91,204))):
        req=Req(torch.arange(cached+extend,dtype=torch.int32)+uid*10000,
                uid,cached,4,uid,SamplingParams(),None)
        req.linear_slot_idx=p.alloc(1)[0];req.mamba_ping_pong=tuple(p.alloc(2))
        reqs.append(req)
    with patch.dict('os.environ',{'QWEN36_PREFIX_MARGIN':'8'}):
        fla=fake_forward(Batch(reqs,'prefill'),p)
    assert [r.mamba_last_track_seqlen for r in reqs]==[145,283]
    assert fla.track_h_row.tolist()==[2,6]
    assert fla.track_conv_src.tolist()==[[125,126,127],[326,327,328]]


def test_bf16_intermediate_does_not_recover_precision_when_donated_to_fp32():
    p=pool();dst=p.alloc(1)[0]
    reference=torch.linspace(-1.01,1.01,4*16*16,dtype=torch.float32).reshape(1,1,4,16,16)
    h=reference.to(torch.bfloat16)
    fla=SimpleNamespace(track_dst=torch.tensor([dst]),track_h_row=torch.tensor([0]),
                        track_conv_src=torch.tensor([[0,1,2]]))
    conv_in=torch.zeros((3,p.conv_states.shape[2]),dtype=torch.bfloat16)
    Qwen3_5GatedDeltaNet._write_track_snapshot(None,p,0,conv_in,h,fla)
    actual=p.recurrent_states[0,dst]
    assert actual.dtype==torch.float32
    assert torch.equal(actual,h[0,0].float())
    assert not torch.equal(actual,reference[0,0])


def test_direct_fp32_store_is_preserved_while_conv_snapshot_is_copied():
    p=pool();dst=p.alloc(1)[0]
    reference=torch.linspace(-1.01,1.01,4*16*16,dtype=torch.float32).reshape(4,16,16)
    p.recurrent_states[0,dst].copy_(reference)
    h=torch.full((1,1,4,16,16),-9.,dtype=torch.bfloat16)
    fla=SimpleNamespace(track_dst=torch.tensor([dst]),track_h_row=torch.tensor([0]),
                        track_conv_src=torch.tensor([[0,1,2]]))
    conv_in=torch.arange(3,dtype=torch.bfloat16)[:,None].expand(-1,p.conv_states.shape[2])
    Qwen3_5GatedDeltaNet._write_track_snapshot(None,p,0,conv_in,h,fla,recurrent_already_stored=True)
    assert torch.equal(p.recurrent_states[0,dst],reference)
    assert torch.equal(p.conv_states[0,dst,0],torch.arange(3,dtype=torch.bfloat16))
