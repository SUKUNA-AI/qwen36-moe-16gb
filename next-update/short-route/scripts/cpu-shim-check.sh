#!/bin/bash
export CUDA_VISIBLE_DEVICES= QWEN36_SHORTROUTE_SHIM=1 QWEN36_SHORTROUTE_OUT=/tmp/shortroute-cpucheck PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/path/to/qwen36-workspace/work/next-update/short-route/shim:/home/USER/qwen36-api/optimization/python
cd /tmp
/home/USER/qwen36-api/.venv/bin/python - <<'PY'
import os, inspect
import freetoken.engine.layer_major as el, freetoken.scheduler.layer_major as sl, freetoken.moe.offload_cache as oc
import shortroute_shim
print('patched', shortroute_shim.PATCHED)
print('min_tokens', el.min_tokens(), el.engine_reason.__module__, el.engine_reason.__qualname__)
os.environ['QWEN36_LAYER_MAJOR_MIN_TOKENS']='2049'; print('min_tokens', el.min_tokens())
print(sl.LayerMajorJob.maybe_start.__func__.__qualname__)
print(oc.OffloadMoeCache.begin_layer_major_prefill.__qualname__)
# fake cache: begin_layer_major_prefill must call begin_prefill with prefill_hit_d2d False and restore it
class C: pass
c=C(); c.prefill_hit_d2d=True; seen=[]
c.begin_prefill=lambda: seen.append(c.prefill_hit_d2d)
oc.OffloadMoeCache.begin_layer_major_prefill(c)
print('begin_prefill saw hit_d2d', seen, 'restored', c.prefill_hit_d2d, 'active', c.layer_major_active)
PY
ls /tmp/shortroute-cpucheck; rm -rf /tmp/shortroute-cpucheck
