"""FTW index validation: every reader entry point refuses a damaged or foreign index before any shard IO."""

import json
import os

import pytest
import torch

from freetoken.checkpoint import ftw
from freetoken.checkpoint.ftw import ALIGN, FTWIndexError, FTWReader, FTWWriter, ftw_quant_format, ftw_tensor_names
from test_ftw_weights import _write_ftw


def _index_path(path):
    return os.path.join(str(path), ftw.INDEX_NAME)


def _rewrite(path, mutate):
    with open(_index_path(path)) as f:
        index = json.load(f)
    mutate(index)
    with open(_index_path(path), "w") as f:
        json.dump(index, f)


def _multi_shard(path):
    """Two small tensors and one that spans shards, with a 2-block shard limit."""
    writer = FTWWriter(str(path), shard_limit=2 * ALIGN)
    tensors = {"a": torch.arange(16, dtype=torch.int32),
               "big": torch.arange(3 * ALIGN, dtype=torch.uint8),
               "c": torch.full((4, 8), 3.0, dtype=torch.bfloat16)}
    for name, tensor in tensors.items():
        writer.add_tensor(name, tensor)
    writer.finalize({"quant_format": "synthetic"})
    return tensors


def test_valid_multi_shard_index_round_trips(tmp_path):
    tensors = _multi_shard(tmp_path)
    reader = FTWReader(str(tmp_path))
    try:
        assert [sh["file"] for sh in reader.shards] == ["freetoken-00000.ftw", "freetoken-00001.ftw",
                                                         "freetoken-00002.ftw"]
        for name, tensor in tensors.items():
            entry = reader.tensors[name]
            dest = bytearray(-(-entry["nbytes"] // ALIGN) * ALIGN)
            reader.read_into(memoryview(dest), entry)
            assert bytes(dest[:entry["nbytes"]]) == tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    finally:
        reader.close()
    assert ftw_tensor_names(str(tmp_path)) == list(tensors)
    assert ftw_quant_format(str(tmp_path)) == "synthetic"


def _shard(i):
    return lambda index: index["shards"][i]


CORRUPTIONS = {
    "not-an-object": None,
    "format": lambda index: index.update(format="safetensors"),
    "future-version": lambda index: index.update(version=999),
    "string-version": lambda index: index.update(version="1"),
    "bool-version": lambda index: index.update(version=True),
    "missing-version": lambda index: index.pop("version"),
    "align": lambda index: index.update(align=512),
    "shard-limit": lambda index: index.update(shard_limit=ALIGN + 1),
    "total-bytes": lambda index: index.update(total_bytes=index["total_bytes"] + ALIGN),
    "shard-path-escape": lambda index: _shard(0)(index).update(file="../freetoken-00000.ftw"),
    "shard-renamed": lambda index: _shard(1)(index).update(file="freetoken-00007.ftw"),
    "shards-swapped": lambda index: index["shards"].reverse(),
    "shard-gap": lambda index: _shard(1)(index).update(global_off=_shard(1)(index)["global_off"] + ALIGN),
    "shard-overlap": lambda index: _shard(1)(index).update(global_off=_shard(1)(index)["global_off"] - ALIGN),
    "shard-over-limit": lambda index: _shard(0)(index).update(nbytes=4 * ALIGN),
    "shard-missing": lambda index: index["shards"].pop(),
    "duplicate-tensor": lambda index: index["tensors"].append(dict(index["tensors"][0])),
    "duplicate-name-other-range": lambda index: index["tensors"][2].update(name="a"),
    "nbytes-vs-shape": lambda index: index["tensors"][0].update(nbytes=index["tensors"][0]["nbytes"] + 4),
    "nbytes-bool": lambda index: index["tensors"][0].update(nbytes=True, shape=[1], dtype="uint8"),
    "out-of-bounds": lambda index: index["tensors"][2].update(global_off=index["total_bytes"]),
    "unaligned-offset": lambda index: index["tensors"][1].update(global_off=index["tensors"][1]["global_off"] + 64),
    "overlapping-tensors": lambda index: index["tensors"][1].update(global_off=0),
    "unknown-dtype": lambda index: index["tensors"][0].update(dtype="Tensor"),
    "negative-shape": lambda index: index["tensors"][0].update(shape=[-16]),
    "missing-kind": lambda index: index["tensors"][0].pop("kind"),
    "missing-name": lambda index: index["tensors"][0].pop("name"),
    "tensors-not-list": lambda index: index.update(tensors={}),
}


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_corrupt_index_is_refused_before_shard_io(tmp_path, monkeypatch, name):
    _multi_shard(tmp_path)
    if name == "not-an-object":
        with open(_index_path(tmp_path), "w") as f:
            json.dump([], f)
    else:
        _rewrite(tmp_path, CORRUPTIONS[name])
    opened = []
    real_open = os.open
    monkeypatch.setattr(ftw.os, "open", lambda p, *a, **k: opened.append(p) or real_open(p, *a, **k))
    for load in (FTWReader, ftw_tensor_names, ftw_quant_format):
        with pytest.raises(FTWIndexError):
            load(str(tmp_path))
    assert not [p for p in opened if str(p).endswith(".ftw")]


def test_helper_checkpoint_passes_and_duplicates_are_not_silently_replaced(tmp_path):
    _write_ftw(tmp_path, ["model.a.weight", "model.b.weight"])
    assert ftw_tensor_names(str(tmp_path)) == ["model.a.weight", "model.b.weight"]
    _rewrite(tmp_path, lambda index: index["tensors"][1].update(name="model.a.weight"))
    with pytest.raises(FTWIndexError, match="duplicate tensor"):
        FTWReader(str(tmp_path))


def test_empty_checkpoint_is_a_valid_index(tmp_path):
    FTWWriter(str(tmp_path)).finalize({})
    reader = FTWReader(str(tmp_path))
    assert reader.tensors == {} and reader.shards == []
    reader.close()
