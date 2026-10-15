"""Diagnostic-only VRAM attribution (QWEN36_MEMORY_HISTORY=<directory>); never set in a profile.

Records the PyTorch allocator history from engine start, before the first CUDA allocation, and
writes a snapshot plus allocator counters after each of the first QWEN36_MEMORY_HISTORY_DUMPS
(default 4) finished requests. Recording python stacks slows every allocation; keep it off for speed runs.
"""
import json
import os
import pathlib

ENV = "QWEN36_MEMORY_HISTORY"
DUMPS_ENV = "QWEN36_MEMORY_HISTORY_DUMPS"

_dumped = 0


def directory():
    raw = os.environ.get(ENV, "")
    return pathlib.Path(raw) if raw else None


def start():
    target = directory()
    if target is None:
        return
    import torch

    target.mkdir(parents=True, exist_ok=True)
    # python stacks only: C++ unwinding makes each snapshot take minutes and stalls the server
    torch.cuda.memory._record_memory_history(
        max_entries=int(os.environ.get(ENV + "_ENTRIES", "100000")), stacks="python"
    )


def dump(label=None) -> None:
    """Write snapshot-<n>.pickle and stats-<n>.json for the next finished request, up to the cap."""
    global _dumped
    target = directory()
    if target is None or _dumped >= int(os.environ.get(DUMPS_ENV, "4")):
        return
    import torch

    _dumped += 1
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    stats = {
        "label": label,
        "mem_get_info": {"free": free, "total": total},
        "memory_allocated": torch.cuda.memory_allocated(),
        "memory_reserved": torch.cuda.memory_reserved(),
        "max_memory_allocated": torch.cuda.max_memory_allocated(),
        "max_memory_reserved": torch.cuda.max_memory_reserved(),
        "memory_stats": {k: v for k, v in torch.cuda.memory_stats().items() if isinstance(v, (int, float))},
    }
    (target / f"stats-{_dumped}.json").write_text(json.dumps(stats, indent=1, default=str))
    torch.cuda.memory._dump_snapshot(str(target / f"snapshot-{_dumped}.pickle"))
