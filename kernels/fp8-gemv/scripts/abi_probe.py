"""CPU-only ABI probe of libqwen_fp8_gemv.so, the shape the release tooling would run.

Runs with CUDA_VISIBLE_DEVICES='' in a fresh process: dlopen + constant entry points only (no CUDA
runtime call), then reports whether the CUDA driver got mapped. Also loads the library from an
anonymous memfd copy (the FreeToken loader path) to prove that loading route works here.
Usage: python -I -S abi_probe.py <library>  -> one JSON line.
"""
import ctypes
import hashlib
import json
import os
import sys

path = sys.argv[1]
result = {}
lib = ctypes.CDLL(path)
for name in ("qwen_fp8_gemv_rowwise", "qwen_fp8_gemv_init", "qwen_fp8_gemv_error"):
    result[name] = hasattr(lib, name)
for name in ("qwen_fp8_gemv_abi_version", "qwen_fp8_gemv_max_m"):
    fn = getattr(lib, name, None)
    if fn is None:
        result[name] = None
    else:
        fn.argtypes = []
        fn.restype = ctypes.c_int
        result[name] = int(fn())
fn = getattr(lib, "qwen_fp8_gemv_source_sha256", None)
if fn is None:
    result["qwen_fp8_gemv_source_sha256"] = None
else:
    fn.argtypes = []
    fn.restype = ctypes.c_char_p
    result["qwen_fp8_gemv_source_sha256"] = fn().decode("ascii")
# memfd route (what freetoken/kernel/cuda_fp8_gemv.py does after hashing the bytes)
data = open(path, "rb").read()
fd = os.memfd_create("qwen_fp8_gemv_probe", os.MFD_CLOEXEC)
view = memoryview(data)
while view:
    view = view[os.write(fd, view):]
lib2 = ctypes.CDLL(f"/proc/self/fd/{fd}")
fn2 = getattr(lib2, "qwen_fp8_gemv_abi_version", None)
if fn2 is not None:
    fn2.restype = ctypes.c_int
result["memfd_abi_version"] = int(fn2()) if fn2 is not None else None
result["sha256"] = hashlib.sha256(data).hexdigest()
with open("/proc/self/maps") as maps:
    text = maps.read()
result["libcuda_loaded"] = "libcuda.so" in text
result["memfd_mapped"] = "memfd:qwen_fp8_gemv_probe" in text
print(json.dumps(result, sort_keys=True))
