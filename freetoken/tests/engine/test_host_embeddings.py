"""CPU checks of the QWEN36_HOST_EMBEDDINGS switch; the byte gate is tests/layers/test_host_embedding_gpu.py."""
from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.engine import host_embeddings
from freetoken.layers.embedding import VocabParallelEmbedding


@pytest.fixture(autouse=True)
def _tp1():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


@pytest.mark.parametrize("raw, expected", [(None, False), ("", False), ("0", False), ("1", True)])
def test_flag_values(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(host_embeddings.ENV, raising=False)
    else:
        monkeypatch.setenv(host_embeddings.ENV, raw)
    assert host_embeddings.enabled() is expected


@pytest.mark.parametrize("raw", ["true", "yes", "2", " 1"])
def test_flag_rejects_typos(monkeypatch, raw):
    monkeypatch.setenv(host_embeddings.ENV, raw)
    with pytest.raises(ValueError, match=host_embeddings.ENV):
        host_embeddings.enabled()


def _model(tied):
    embed = VocabParallelEmbedding(16, 8)
    head = SimpleNamespace(tied_embedding=embed if tied else None)
    return SimpleNamespace(model=SimpleNamespace(embed_tokens=embed), lm_head=head)


def test_untied_model_yields_its_embedding():
    model = _model(tied=False)
    assert host_embeddings.embedding_of(model) is model.model.embed_tokens


def test_tied_lm_head_keeps_the_table_in_vram():
    with pytest.raises(ValueError, match="ties lm_head"):
        host_embeddings.embedding_of(_model(tied=True))


def test_model_without_embedding_table_is_refused():
    with pytest.raises(ValueError, match="no model.embed_tokens"):
        host_embeddings.embedding_of(SimpleNamespace(model=SimpleNamespace(), lm_head=None))


def test_move_refuses_a_cpu_table_and_tensor_parallel():
    embed = VocabParallelEmbedding(16, 8)
    embed.weight = torch.zeros(16, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA table"):
        embed.move_to_host()
    embed.tp_size = 2
    with pytest.raises(ValueError, match="tensor parallel"):
        embed.move_to_host()
    assert embed.compute_device == torch.device("cpu")  # unchanged: still the GPU-path layout
