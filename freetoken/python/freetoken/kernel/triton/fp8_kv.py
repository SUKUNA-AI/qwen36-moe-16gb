"""FP8 KV quantization and paged scatter for Qwen's small GQA heads."""
import triton
import triton.language as tl


@triton.jit
def _store(K, V, LOC, KC, VC, OVERFLOW,
           KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
           VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
           HEADS: tl.constexpr, DIM: tl.constexpr,
           K_INV: tl.constexpr, V_INV: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    x = tl.arange(0, BLOCK)
    mask = x < HEADS * DIM
    head, dim = x // DIM, x % DIM
    k = tl.load(K + token*KS0 + head*KS1 + dim*KS2, mask, 0).to(tl.float32) * K_INV
    v = tl.load(V + token*VS0 + head*VS1 + dim*VS2, mask, 0).to(tl.float32) * V_INV
    failures = tl.sum(((tl.abs(k)>448) | (tl.abs(v)>448)).to(tl.int32),0)
    if failures > 0: tl.atomic_add(OVERFLOW, failures)
    destination = tl.load(LOC + token).to(tl.int64) * (HEADS*DIM) + x
    tl.store(KC + destination, tl.minimum(tl.maximum(k,-448),448), mask)
    tl.store(VC + destination, tl.minimum(tl.maximum(v,-448),448), mask)


def store_fp8(k, v, indices, k_cache, v_cache, scales, overflow):
    heads,dim=k_cache.shape[-2:]
    k=k.view(-1,heads,dim)
    v=v.view(-1,heads,dim)
    n, heads, dim = k.shape
    _store[(n,)](k,v,indices,k_cache,v_cache,overflow,
        *k.stride(),*v.stride(),heads,dim,1.0/scales[0],1.0/scales[1],
        triton.next_power_of_2(heads*dim))
