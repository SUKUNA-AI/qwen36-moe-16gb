"""SM120 owner gate: expert slots whose byte offset passes 2**32 (memory_ratio stages near 4096 slots).

Qwen3.6-35B-A3B's gate_up bank row is 1 MiB, so slot 4096 starts at byte 2**32 and slot 2048 at
2**31; the top stage (memory_ratio 0.94 with host embeddings) plans ~5400 slots. Every kernel that
addresses a slot by index must compute that offset in 64 bits: the decode GEMVs that read slots, and
the two gathers that write missed experts into them. (Prefill kernels
index a 256-position layer buffer, never the slot cache.) Each check compares a high slot with the
same bytes at a low slot. Needs ~6.5 GiB of free VRAM; run under the GPU lock.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="owner CUDA/SM120 required")

HIDDEN, INTER = 2048, 512
SLOTS = 5600  # gate_up bank: 5600 MiB, past 2**32 bytes and above every planned stage
HIGH = (SLOTS - 1, 4096, 2049, 2048)
LOW = (3, 4, 5, 6)


@pytest.fixture(scope="module")
def gate_up_bank():
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(7)
    packed = torch.empty((SLOTS, 2 * INTER, HIDDEN // 2), dtype=torch.uint8, device=device)
    scale = torch.empty((SLOTS, 2 * INTER, HIDDEN // 16), dtype=torch.uint8, device=device)
    glob = torch.empty((SLOTS, 2 * INTER), dtype=torch.float32, device=device)
    for hi, lo in zip(HIGH, LOW):
        row = torch.randint(0, 256, (2 * INTER, HIDDEN // 2), dtype=torch.uint8, generator=g)
        # e4m3 scales in a finite, non-tiny range (exponent field 4..11), random mantissa
        srow = (torch.randint(4, 12, (2 * INTER, HIDDEN // 16), generator=g) << 3
                | torch.randint(0, 8, (2 * INTER, HIDDEN // 16), generator=g)).to(torch.uint8)
        grow = torch.rand((2 * INTER,), generator=g) + 0.5
        for slot in (hi, lo):
            packed[slot].copy_(row)
            scale[slot].copy_(srow)
            glob[slot].copy_(grow)
    assert packed[HIGH[0]].data_ptr() - packed.data_ptr() >= 2**32
    yield packed, scale.view(torch.float8_e4m3fn), glob
    del packed, scale, glob
    torch.cuda.empty_cache()


def _bits(t):
    return t.contiguous().view(torch.int16)


@pytest.mark.parametrize("variant", ["marlin", "serial"])
@pytest.mark.parametrize("top_k", [1, 4])
def test_decode_gemv_reads_slots_past_4gib(gate_up_bank, variant, top_k):
    from freetoken.moe.fused_nvfp4 import _decode_gemm, _decode_gemm_marlin

    gemm = _decode_gemm_marlin if variant == "marlin" else _decode_gemm
    packed, scale, glob = gate_up_bank
    device = packed.device
    g = torch.Generator(device="cpu").manual_seed(top_k)
    a = torch.randn((1, HIDDEN), generator=g).to(device=device, dtype=torch.bfloat16)
    weights = torch.ones((1, top_k), dtype=torch.float32, device=device)
    outs = []
    for slots in (HIGH[:top_k], LOW[:top_k]):
        ids = torch.tensor([slots], dtype=torch.int32, device=device)
        c = torch.empty((1, top_k, 2 * INTER), dtype=torch.bfloat16, device=device)
        gemm(a, packed, scale, glob, c, weights, ids, False, False)
        outs.append(c)
    torch.cuda.synchronize()
    assert outs[0].abs().sum() > 0
    assert torch.equal(_bits(outs[0]), _bits(outs[1]))


@pytest.mark.parametrize("kernel", ["decode_gather", "fast_index_copy_multi"])
def test_gathers_write_slots_past_4gib(gate_up_bank, kernel):
    from freetoken.kernel.pinned import device_ptr

    packed = gate_up_bank[0]
    device = packed.device
    row_bytes = packed[0].numel()
    g = torch.Generator(device="cpu").manual_seed(11)
    source = torch.randint(0, 256, (len(HIGH), row_bytes), dtype=torch.uint8, generator=g).pin_memory()
    sentinels = (SLOTS - 2, 4095, 4097, 2047, 2050)
    for slot in HIGH + sentinels:
        packed[slot].fill_(0xA5)
    dst_ptrs = torch.tensor([packed.data_ptr()], dtype=torch.int64, device=device)
    src_ptrs = torch.tensor([device_ptr(source)], dtype=torch.int64, device=device)
    feat = torch.tensor([row_bytes], dtype=torch.int64, device=device)
    dst_idx = torch.tensor(HIGH, dtype=torch.int32, device=device)
    src_idx = torch.arange(len(HIGH), dtype=torch.int32, device=device)
    count = torch.tensor([len(HIGH)], dtype=torch.int64, device=device)
    if kernel == "decode_gather":
        from freetoken.kernel.decode_gather import TUNED_DECODE_GATHER, decode_gather_jit

        decode_gather_jit(dst_ptrs, src_ptrs, feat, dst_idx, src_idx, count, config=TUNED_DECODE_GATHER)
    else:
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        fast_index_copy_multi_jit(dst_ptrs, src_ptrs, feat, dst_idx, src_idx, count)
    torch.cuda.synchronize()
    for i, slot in enumerate(HIGH):
        assert torch.equal(packed[slot].view(-1).cpu(), source[i]), slot
    for slot in sentinels:
        assert bool((packed[slot] == 0xA5).all()), slot
