from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    seeds: list[int | None] | None = None


def position_seed(seed: int, position: int) -> int:
    """SplitMix64 gives each absolute token position an independent Philox key."""
    mask = (1 << 64) - 1
    x = (seed + 0x9E3779B97F4A7C15 * (position + 1)) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    return (x ^ (x >> 31)) & ((1 << 63) - 1)


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
    seed: int | None = None,
) -> torch.Tensor:
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling

    rng = {} if seed is None else {'seed': seed, 'offset': 0}
    if rng and not is_flashinfer_installed():
        raise RuntimeError('Request-local seed requires the qualified FlashInfer sampler')
    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs, **rng)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k, **rng)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p, **rng)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p, **rng)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        seeds = [None if p.seed is None else position_seed(p.seed, r.device_len)
                 for p, r in zip(params, batch.reqs)]
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p,
                                 seeds=seeds if any(s is not None for s in seeds) else None)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("SamplerKernelDispatch"):
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            if args.seeds is not None:
                return torch.cat([sample_impl(logits[i:i+1].float(), args.temperatures[i:i+1],
                    None if args.top_k is None else args.top_k[i:i+1],
                    None if args.top_p is None else args.top_p[i:i+1], seed)
                    for i, seed in enumerate(args.seeds)])
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
