from __future__ import annotations

from typing import Sequence
import json
import os
import pathlib

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.

    ``layer_ids`` lets the pool back only a *subset* of the model's layers while
    callers keep indexing by their global ``layer_id``. Hybrid models (e.g. the
    Qwen3.5 GatedDeltaNet/full-attention stack) interleave linear-attention layers
    that hold no paged KV; passing the full-attention layer ids here allocates one
    storage slab per KV layer (not per model layer) and remaps the global id to its
    dense slot, avoiding a multiple-x over-allocation of unused slabs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: Sequence[int] | None = None,
    ) -> None:
        self.query_dtype = dtype
        self.fp8 = os.environ.get('QWEN36_KV_DTYPE','bf16') == 'fp8'
        self.calibrate_path = os.environ.get('QWEN36_KV_CALIBRATE')
        self.scales = {}
        if self.fp8:
            path = os.environ.get('QWEN36_KV_SCALES')
            if not path: raise ValueError('FP8 KV requires a calibrated QWEN36_KV_SCALES file')
            doc = json.loads(pathlib.Path(path).read_text())
            if doc.get('checkpoint_revision') != '1355db6a052410cfd62085d94b58866fd0f2c3c5':
                raise ValueError('KV calibration checkpoint revision mismatch')
            self.scales = {int(key):tuple(value) for key,value in doc['scales'].items()}
            for layer in (layer_ids if layer_ids is not None else range(num_layers)):
                values = self.scales.get(layer)
                if values is None or len(values)!=2 or any(not 0 < v < 1e6 for v in values):
                    raise ValueError('Invalid or missing KV scale')
            dtype = torch.float8_e4m3fn
        self.overflow = torch.zeros((),dtype=torch.int64,device=device)
        self.calibration_max = torch.zeros((num_layers,2),dtype=torch.float32,device=device) if self.calibrate_path else None
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
        if layer_ids is None:
            num_storage_layers = num_layers
            self._layer_map: list[int] | None = None
        else:
            num_storage_layers = len(layer_ids)
            layer_map = [-1] * num_layers
            for dense, global_id in enumerate(layer_ids):
                if global_id < 0 or global_id >= num_layers:
                    raise ValueError(f"KV layer id {global_id} outside [0, {num_layers})")
                layer_map[global_id] = dense
            self._layer_map = layer_map
        self._kv_buffer = torch.empty(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        """
        _, num_storage_layers, _old_pages, page_size, local_kv_heads, head_dim = self._kv_buffer.shape
        dtype = self._kv_buffer.dtype
        device = self._device
        self._k_buffer = None
        self._v_buffer = None
        self._kv_buffer = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self._kv_buffer = torch.empty(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        per_token = sum(
            spec_kv_bytes_per_token(spec, config)
            for spec in config.model_config.kv_cache_group_specs()
            if not spec.is_swa
        )
        if os.environ.get('QWEN36_KV_DTYPE','bf16') == 'fp8': per_token //= 2
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        buf = self._kv_buffer
        tokens = int(buf.shape[2]) * int(buf.shape[3])
        return int(buf.numel() * buf.element_size()) // tokens, 0

    def _dense(self, layer_id: int) -> int:
        if self._layer_map is None:
            return layer_id
        dense = self._layer_map[layer_id]
        if dense < 0:
            raise KeyError(f"layer {layer_id} has no paged KV storage")
        return dense

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[self._dense(index)]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[self._dense(index)]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        from freetoken.kernel import store_cache

        dense = self._dense(layer_id)
        if self.calibration_max is not None:
            torch.maximum(self.calibration_max[layer_id,0], k.abs().amax().float(), out=self.calibration_max[layer_id,0])
            torch.maximum(self.calibration_max[layer_id,1], v.abs().amax().float(), out=self.calibration_max[layer_id,1])
        if self.fp8:
            from freetoken.kernel.triton.fp8_kv import store_fp8
            store_fp8(k,v,out_loc,self._k_buffer[dense],self._v_buffer[dense],
                      self.scales[layer_id],self.overflow)
            return
        store_cache(
            k_cache=self._k_buffer[dense].view(self._storage_shape),
            v_cache=self._v_buffer[dense].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    def save_calibration(self):
        if self.calibration_max is None: return
        values = self.calibration_max.cpu().tolist()
        doc = {'checkpoint_revision':'1355db6a052410cfd62085d94b58866fd0f2c3c5',
               'format':'e4m3fn', 'margin':1.2, 'amax':values,
               'scales':{str(i):[max(a*1.2/448,1e-8) for a in pair]
                         for i,pair in enumerate(values) if max(pair)>0}}
        path = pathlib.Path(self.calibrate_path)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(doc,indent=2));temp.replace(path)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
