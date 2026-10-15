"""Default-off: keep the token-embedding table in pinned host memory instead of VRAM.

The BF16 table of Qwen3.6-35B-A3B is 248320 x 2048 = 0.95 GiB. Moving it before the weights are
measured lowers ``weights_bytes``, so --moe-cache-auto turns those bytes into expert slots
inside the same ``memory_ratio`` footprint. Lookups copy rows byte for byte over PCIe
(``VocabParallelEmbedding.move_to_host``).
"""
import os

ENV = "QWEN36_HOST_EMBEDDINGS"


def enabled():
    raw = os.environ.get(ENV, "0")
    if raw not in ("", "0", "1"):
        raise ValueError(f"{ENV}={raw!r}: expected unset, 0 or 1")
    return raw == "1"


def embedding_of(model):
    """The input embedding a host move would take, after refusing a table the LM head also reads."""
    from freetoken.layers.embedding import VocabParallelEmbedding

    embed = getattr(getattr(model, "model", None), "embed_tokens", None)
    if not isinstance(embed, VocabParallelEmbedding):
        raise ValueError(f"{ENV}=1: {type(model).__name__} has no model.embed_tokens table")
    if getattr(getattr(model, "lm_head", None), "tied_embedding", None) is not None:
        # a tied head multiplies by the same table every step; it must stay in VRAM
        raise ValueError(f"{ENV}=1: {type(model).__name__} ties lm_head to embed_tokens")
    return embed


def move_embeddings_to_host(model) -> int:
    """Move the input embedding table to pinned host memory and release its VRAM; returns its bytes."""
    import torch

    nbytes = embedding_of(model).move_to_host()
    # the freed GPU block stays reserved by the caching allocator until emptied, and the caller
    # measures weights from free VRAM right after this
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return nbytes
