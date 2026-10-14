"""Experimental RAM complement, disabled unless QWEN36_RESIDENCY is explicit.

Only native NVFP4/Triton, single-GPU offload is supported. 2E scratch slots are
reserved for prefill; every other GPU slot is an owner, never a redundant copy.
All exchanges run on the compute stream. Prefill readers are fenced with the
existing ready/release events. Cache rebuilding is deliberately rejected.
"""
import math
import os
from types import SimpleNamespace
import torch
import triton
from .offload_cache import OffloadMoeCache,attach_offload_moe_cache


class ComplementCache(OffloadMoeCache):
    def allocate(self,method):
        from .host_banks import HostBank
        self.offset=2*self.num_experts
        self.owner_slots=self.cache_size-self.offset
        if self.owner_slots<self.num_layers:raise ValueError('Complement needs resident slots')
        self.adaptive=os.environ.get('QWEN36_RESIDENCY')=='complement'
        self.policy=os.environ.get('QWEN36_CACHE_POLICY','lru')
        self.frequency=torch.zeros(self.num_layers*self.num_experts,device=self.device,dtype=torch.float32)
        self.frequency_at=torch.zeros(self.num_layers*self.num_experts,device=self.device,dtype=torch.int64)
        self.old_ids=torch.empty(self.owner_slots,device=self.device,dtype=torch.int32)
        self.host_slot=torch.full((self.num_layers*self.num_experts,),-1,dtype=torch.int32)
        resident=[];host=[]
        for layer in range(self.num_layers):
            count=self.owner_slots//self.num_layers+(layer<self.owner_slots%self.num_layers)
            resident.extend(range(layer*self.num_experts,layer*self.num_experts+count))
            host.extend(range(layer*self.num_experts+count,(layer+1)*self.num_experts))
        self._initial_resident=resident;self._initial_host=host
        self.host_slot[host]=torch.arange(len(host),dtype=torch.int32)
        self.host_slot=self.host_slot.to(self.device)
        self.slot_for_id.view(-1)[torch.tensor(resident,device=self.device)]=torch.arange(self.owner_slots,device=self.device,dtype=torch.int32)
        self.id_of_slot[self.offset:]=torch.tensor(resident,device=self.device,dtype=torch.int32)
        self._host_banks={}
        for name,spec in method.layout().items():
            if spec.resident:raise ValueError('Complement only supports native NVFP4 banks')
            self._host_banks[name]=HostBank((len(host),*spec.shape),spec.dtype)
            self.bank_sources[name]=[self._host_banks[name].tensor]
            self.bank_caches[name]=torch.empty((self.cache_size,*spec.shape),device=self.device,dtype=spec.dtype)
        self.banks=[(self.bank_sources[n],self.bank_caches[n]) for n in self.bank_schema]
        self.layer_residency=['pinned']*self.num_layers

    def load(self,method,pieces):
        # Bounded per-piece staging: never build a second complete CPU bank.
        rid={gid:i+self.offset for i,gid in enumerate(self._initial_resident)}
        hid={gid:i for i,gid in enumerate(self._initial_host)}
        written=set()
        for layer,e0,e1,piece in pieces:
            stage={name:torch.empty((e1-e0,*spec.shape),dtype=spec.dtype) for name,spec in method.layout().items()}
            method.pack(piece,stage)
            for e in range(e0,e1):
                gid=layer*self.num_experts+e
                if gid in written:raise ValueError('Duplicate expert during complement load')
                written.add(gid)
                for name,t in stage.items():
                    target=self.bank_caches[name][rid[gid]] if gid in rid else self._host_banks[name].tensor[hid[gid]]
                    target.copy_(t[e-e0])
        if len(written)!=self.num_layers*self.num_experts:raise ValueError('Missing expert during complement load')
        del stage
        for bank in self._host_banks.values():bank.pin()
        from freetoken.kernel.pinned import device_ptr
        self._c_host=torch.tensor([device_ptr(self._host_banks[n].tensor) for n in self.bank_schema],device=self.device,dtype=torch.int64)
        self._c_gpu=torch.tensor([self.bank_caches[n].data_ptr() for n in self.bank_schema],device=self.device,dtype=torch.int64)
        feats=[math.prod(self.bank_caches[n].shape[1:])*self.bank_caches[n].element_size() for n in self.bank_schema]
        self._c_feat=torch.tensor(feats,device=self.device,dtype=torch.int64)
        self._tiles=8
        self.persistent_transport=None
        if os.environ.get('QWEN36_COMPLEMENT_TRANSPORT')=='persistent':
            from .cuda_transport import Transport
            self.persistent_transport=Transport(feats,threads=1024,unroll=1,policy='persistent',cap=8)
        self.cuda_transport=os.environ.get('QWEN36_COMPLEMENT_TRANSPORT')=='cuda'
        if self.cuda_transport:
            if any(n%16 for n in feats):raise ValueError('CUDA transport needs 16-byte expert alignment')
            from .complement_cuda import library
            library()
        self._init_prefill_overlap_buffers()
        self._initial_resident=None;self._initial_host=None
        torch.cuda.synchronize()

    def ensure_experts(self,layer_id,expert_ids):
        self._pending_src_layer=layer_id
        self._plan_capacity=expert_ids.numel()
        if not self.adaptive:
            from .complement_kernels import fixed_plan
            fixed_plan[(1,)](expert_ids,self.slot_for_id,self.src_indices,self.evict_slots,self.num_indices,
                layer_id*self.num_experts,self.offset,expert_ids.numel(),triton.next_power_of_2(expert_ids.numel()))
            return
        from flashlib.kernels.slot_cache import lru_ensure
        self.old_ids.copy_(self.id_of_slot[self.offset:])
        if self.policy=='decayed_frequency':
            from .complement_kernels import frequency_plan
            frequency_plan[(1,)](expert_ids,self.slot_for_id,self.id_of_slot[self.offset:],
                self.frequency,self.frequency_at,self.step,self.src_indices,self.evict_slots,self.num_indices,
                self.lru_stats[layer_id],layer_id*self.num_experts,self.owner_slots,expert_ids.numel(),
                float(os.environ.get('QWEN36_CACHE_HALFLIFE','4096')),triton.next_power_of_2(self.owner_slots))
        else:
            lru_ensure(expert_ids,self.slot_for_id.view(-1),self.id_of_slot[self.offset:],
                self.usage[self.offset:],self.step,expert_ids,self.src_indices,self.evict_slots,self.num_indices,
                stats=self.lru_stats[layer_id] if self.collect_stats else None,id_base=layer_id*self.num_experts)
        expert_ids.add_(self.offset)

    def copy_missing(self):
        from .complement_kernels import exchange,commit,fixed_copy
        layer=self._pending_src_layer
        grid=(len(self.bank_schema),self._plan_capacity,self._tiles)
        if self.persistent_transport is not None:
            self.persistent_transport.exchange(self._c_host,self._c_gpu,self._c_feat,self.host_slot,
                self.src_indices,self.evict_slots,self.num_indices,base=layer*self.num_experts,
                offset=self.offset if self.adaptive else 0,exchange=self.adaptive)
            if self.adaptive:
                commit[(1,)](self.host_slot,self.old_ids,self.src_indices,self.evict_slots,self.num_indices,
                    layer*self.num_experts,triton.next_power_of_2(self._plan_capacity))
            return
        if self.cuda_transport:
            from .complement_cuda import exchange as cuda_exchange
            cuda_exchange(self,layer)
            if self.adaptive:
                commit[(1,)](self.host_slot,self.old_ids,self.src_indices,self.evict_slots,self.num_indices,
                    layer*self.num_experts,triton.next_power_of_2(self._plan_capacity))
            return
        if self.adaptive:
            exchange[grid](self._c_host,self._c_gpu,self._c_feat,self.host_slot,self.old_ids,
                self.src_indices,self.evict_slots,self.num_indices,layer*self.num_experts,self.offset,1024)
            commit[(1,)](self.host_slot,self.old_ids,self.src_indices,self.evict_slots,self.num_indices,
                layer*self.num_experts,triton.next_power_of_2(self._plan_capacity))
        else:
            fixed_copy[grid](self._c_host,self._c_gpu,self._c_feat,self.host_slot,self.src_indices,
                self.evict_slots,self.num_indices,layer*self.num_experts,1024)

    def begin_prefill(self):
        self._prefill_buffer_layer=[None,None];self._prefill_buffer_released=[True,True]
        self.prefill_begin_event.record(torch.cuda.current_stream(self.device))
        self.prefill_copy_stream.wait_event(self.prefill_begin_event)

    def prefetch_prefill_layer(self,layer_id):
        if layer_id>=self.num_layers:return
        b=layer_id%2
        if self._prefill_buffer_layer[b]==layer_id:return
        if self._prefill_buffer_layer[b] is not None and not self._prefill_buffer_released[b]:
            raise RuntimeError('Complement staging buffer still has a reader')
        from .complement_kernels import gather_layer
        with torch.cuda.stream(self.prefill_copy_stream):
            if self._prefill_buffer_has_release_event[b]:self.prefill_copy_stream.wait_event(self.prefill_release_events[b])
            if self.persistent_transport is not None:
                self.persistent_transport.gather(self._c_host,self._c_gpu,self._c_feat,self.host_slot,
                    self.slot_for_id,base=layer_id*self.num_experts,offset=self.offset,
                    destination=b*self.num_experts,experts=self.num_experts)
            elif self.cuda_transport:
                from .complement_cuda import gather
                gather(self,layer_id,b*self.num_experts)
            else:
                gather_layer[(len(self.bank_schema),self.num_experts,self._tiles)](
                    self._c_host,self._c_gpu,self._c_feat,self.host_slot,self.slot_for_id,
                    layer_id*self.num_experts,self.offset,b*self.num_experts,1024)
            self.prefill_ready_events[b].record(self.prefill_copy_stream)
        self._prefill_buffer_layer[b]=layer_id;self._prefill_buffer_released[b]=False

    def reset(self):
        # Ownership is permanent until an exchange commits. Resetting the map would
        # discard the sole copy of an expert; only policy ages may be reset.
        self.usage.zero_();self.step.zero_();self.frequency.zero_();self.frequency_at.zero_()

    def rebuild(self,cache_size):raise RuntimeError('Complement residency requires process restart for resizing')


def build(engine,config,method):
    if method.kernel.name!='triton' or config.model_config.expert_quant!='nvfp4' or config.moe_strategy!='offload':
        raise ValueError('Complement requires native Triton NVFP4 GPU offload')
    layout=method.layout()
    dummy=SimpleNamespace(sources={name:[torch.empty((config.model_config.num_experts,*spec.shape),dtype=spec.dtype,device='meta')]
                                 for name,spec in layout.items() if not spec.resident})
    size,pages,overlap=engine._resolve_auto_moe_cache_size(config,dummy,method)
    if not overlap:raise ValueError('Complement requires prefill overlap')
    object.__setattr__(config,'moe_cache_size',size);object.__setattr__(config,'num_page_override',pages)
    cache=ComplementCache(num_layers=config.model_config.num_moe_layers,num_experts=config.model_config.num_experts,
        cache_size=size,device=engine.device,quant_format='nvfp4',layout=layout,prefill_overlap=True)
    cache.allocate(method)
    from .expert_pieces import iter_expert_pieces
    cache.load(method,iter_expert_pieces(config.model_path,config.model_config,method.kind,parallel=False))
    cache.collect_stats=True
    attach_offload_moe_cache(engine.model,cache)
    engine.ctx.moe_offload_cache=cache;engine.moe_offload_cache=cache
    return cache
