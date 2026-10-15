"""Byte copy of table rows that stay in pinned host memory (host-resident token embeddings).

The GPU dereferences the mapped table over PCIe (``kernel/pinned.device_ptr``), one program
per requested row. Rows move as whole integer words, never through a float type, so every
in-range row lands bit-for-bit equal to the table row. Ids outside the table store zeros.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_WORD_DTYPES = ((8, torch.int64), (4, torch.int32), (2, torch.int16), (1, torch.uint8))
_BLOCK_WORDS = 512


@triton.jit
def _host_rows_kernel(
    table_ptr,
    ids_ptr,
    out_ptr,
    num_rows,
    ROW_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    in_range = (idx >= 0) & (idx < num_rows)
    idx = tl.where(in_range, idx, 0)
    # the table is a host allocation: rebuild a pointer of the output's word type from the raw address
    src = table_ptr.to(tl.int64).to(out_ptr.dtype) + idx * ROW_WORDS
    dst = out_ptr + row * ROW_WORDS
    for start in tl.static_range(0, ROW_WORDS, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < ROW_WORDS
        words = tl.load(src + offsets, mask=mask & in_range, other=0)
        tl.store(dst + offsets, words, mask=mask)


def host_gather_rows(table_ptr: int, num_rows: int, row_bytes: int, ids: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Copy rows ``ids`` of the host table at ``table_ptr`` into ``out`` byte for byte.

    ``ids`` is a flat int32/int64 device tensor; ``out`` is a contiguous device tensor of
    ``ids.numel()`` rows of ``row_bytes`` bytes each. ``table_ptr`` is the device-visible
    address of a pinned+mapped table (``kernel/pinned.device_ptr``).
    """
    n = ids.numel()
    assert out.is_contiguous() and out.numel() * out.element_size() == n * row_bytes, (out.shape, n, row_bytes)
    word_bytes, word = next(w for w in _WORD_DTYPES if row_bytes % w[0] == 0)
    if n:
        row_words = row_bytes // word_bytes
        _host_rows_kernel[(n,)](
            table_ptr,
            ids,
            out.view(word),
            num_rows,
            ROW_WORDS=row_words,
            BLOCK=min(_BLOCK_WORDS, triton.next_power_of_2(row_words)),
            num_warps=4,
        )
    return out


__all__ = ["host_gather_rows"]
