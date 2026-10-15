"""Manual production profiles restricted to owner-reviewed configurations.

Fail-closed gate shared by two callers:

* ``optimization/release.py`` imports this file from the staging tree and calls
  ``validate_profiles`` with a resolver that maps release paths strictly to
  staged bytes (no fallback to installed files).
* The systemd start path ``start-inference.sh`` executes this file. ``main``
  validates the installed profiles, the installed qualification receipt, the
  actual installed bytes, the venv runtime environment (package versions/RECORDs
  and the prebuilt FreeToken kernel cache bytes) and the checkpoint directory
  before any FreeToken/torch import, then configures a closed environment and
  only then hands over to ``serve_guarded``.

Kernel provenance (fail-closed, same check at release and start): the receipt binds the
``kernel/kernel_provenance.py`` inventory of exactly the receipt-bound kernel bytes plus the venv
kernel cache (every prebuilt module and every required extension must be ``prebuilt``: manifest
recipe digest == current csrc, .so sha256 == manifest; every module must be bound by the released
manifest, and a switch that runs a JIT module, ``decode_gather``, needs that module prebuilt).
ctypes libraries (``QWEN36_*_LIBRARY``) are bound by sha256 and environment variable; the CUDA
NVFP4 prefill, CUDA GDN and FP8 GEMV libraries also by an ABI probe in a separate CPU-only process.
The CUDA library switches ``cuda_nvfp4_prefill``, ``cuda_gdn`` and ``cuda_fp8_gemv`` are accepted
only with bit-exact evidence bound to the released library in the receipt.

Agreed API (release owner / transport):
    canonical_profile_sha256(profile)
    validate_profiles(profiles, qualification, resolve_source, *, expected_sources=None, environment=None)
    validate_profile(name, profiles, qualification, resolve_source, *, expected_sources=None, environment=None)

Validation is pure stdlib and side-effect free (no package is imported). The kernel provenance
step executes only the receipt-bound stdlib module ``kernel/kernel_provenance.py`` from verified bytes
(never imported, no bytecode written) over a temporary copy; the ABI probe runs in a subprocess.
"""
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
import types

ROOT = pathlib.Path('/home/USER/qwen36-api')
MODEL = '/path/to/LLM/qwen36-api/model'
CHECKPOINT = {'id': 'nvidia/Qwen3.6-35B-A3B-NVFP4',
              'revision': '1355db6a052410cfd62085d94b58866fd0f2c3c5'}
CONTEXTS = {'baseline64': 65536, 'main128': 131072, 'long256': 262144}
CHUNKS = (1024, 2048, 4096, 8192)
# Exclusive ceiling. Owner rule (15.10): Windows keeps 1 GiB and the model may use the rest; the measured
# footprint at ratio r is about r * 14.65 GiB + 0.67 GiB, so 0.95 stays under 15.6 GiB - 1 GiB.
MAX_MEMORY_RATIO = 0.95
ALLOWED_FLAGS = frozenset({'--moe-prefill-hit-d2d', '--enable-special-token-ckpt'})
# Feature switches (exact booleans in a profile). SWITCH_ENV is the complete fixed environment a switch
# exports when true (configure_environment); a switch with a ctypes library also exports its release path
# from NATIVE_LIBRARIES. Nothing a profile does not switch on reaches FreeToken (CLEARED_ENV_PREFIXES).
SWITCH_ENV = {
    'fused_attention_gate': {'QWEN36_FUSED_ATTN_GATE': '1'},
    'fp32_gdn_snapshot': {'QWEN36_FP32_GDN_SNAPSHOT': '1'},
    'retain_chunk_snapshot': {'QWEN36_RETAIN_CHUNK_SNAPSHOT': '1'},
    'cuda_nvfp4_prefill': {'QWEN36_NVFP4_PREFILL': 'cuda'},
    # Layer-major prefill (engine/scheduler/moe layer_major.py): the same tasks in layer-major order.
    'layer_major_prefill': {'QWEN36_LAYER_MAJOR_PREFILL': '1'},
    # Bit-exact CUDA GDN kernels (kernel/cuda_gdn.py); decode variant pinned to the measured 'cluster'.
    'cuda_gdn': {'QWEN36_CUDA_GDN': '1', 'QWEN36_CUDA_GDN_DECODE': 'cluster'},
    # Bit-exact FP8 decode GEMV (kernel/cuda_fp8_gemv.py; the module also pins the library sha256).
    'cuda_fp8_gemv': {'QWEN36_FP8_GEMV': '1'},
    # Bit-exact decode o-projection (kernel/cuda_oproj_gemv.py) and NVFP4 MoE decode GEMV
    # (kernel/cuda_moe_decode.py); both modules also pin their library sha256.
    'cuda_oproj_gemv': {'QWEN36_OPROJ_GEMV': '1'},
    'cuda_moe_decode': {'QWEN36_MOE_DECODE_CUDA': '1'},
    # TMA decode expert gather (kernel/decode_gather.py): a JIT module that must load as a manifest-bound
    # prebuilt (KERNEL_MODULE_SWITCHES). TVM_FFI_CUDA_ARCH_LIST=12.0 is the arch the module is built for and
    # matches the qualification runs; it is inert because FREETOKEN_DISABLE_JIT=1 never lets tvm-ffi build.
    'decode_gather': {'QWEN36_DECODE_GATHER': 'tuned', 'TVM_FFI_CUDA_ARCH_LIST': '12.0'},
    # Optional, easy to drop: deleting the entry also drops its SWITCH_REQUIRES rule and VALUE_FIELDS
    # (both follow SWITCH_ENV); profiles without the key stay valid.
    # Background refill of the slots layer-major prefill borrowed (moe/slot_refill.py).
    'layer_major_refill': {'QWEN36_LAYER_MAJOR_REFILL': '1'},
    'decode_shared_overlap': {'QWEN36_DECODE_SHARED_OVERLAP': '1'},
    # Token-embedding table in pinned host memory (engine/host_embeddings.py): its VRAM goes to expert slots.
    'host_embeddings': {'QWEN36_HOST_EMBEDDINGS': '1'},
    # C0 (moe/offload_cache.py): prefill copy stream at top priority plus one compute-stream invalidation of both
    # double buffers per prefill epoch, so each layer's DMA no longer waits behind the previous expert GEMM.
    'prefill_copy_priority': {'QWEN36_PREFILL_COPY_PRIORITY': '1', 'QWEN36_PREFILL_EPOCH_INVALIDATE': '1'},
    # D3b (moe/pregate.py): layer L+1's router on layer L's output prefetches up to 1 (GDN) / 2 (attention)
    # predicted misses on the T3 side stream during the mixer; ensure_experts still follows the real routing.
    'decode_pregate': {'QWEN36_DECODE_PREGATE': '1', 'QWEN36_DECODE_PREGATE_ATTN': '2'},
    # D5 (moe/shared_epilogue.py): sigmoid gate, shared product and routed sum of the MoE block in one kernel,
    # with torch's roundings (no fp contraction) and a start-up self-check against torch.
    'decode_shared_epilogue': {'QWEN36_DECODE_SHARED_EPILOGUE': '1'},
    # PLACEHOLDER (possible future switch, not in the runtime tree yet).
    'sync_lean': {'QWEN36_SYNC_LEAN': '1'},
}
FEATURE_SWITCHES = tuple(SWITCH_ENV)
# switch -> switch it needs in the same profile (rules of dropped switches vanish with them).
SWITCH_REQUIRES = {switch: needed for switch, needed in {'layer_major_refill': 'layer_major_prefill',
                                                         'decode_pregate': 'decode_shared_overlap'}.items()
                   if switch in SWITCH_ENV}
# Optional value fields: field -> (variable, allowed values, switch the field needs). Exported only with its
# switch; absent = FreeToken's default. Fields of a dropped switch vanish with it.
VALUE_FIELDS = {field: spec for field, spec in {
    # Fused bit-exact CUDA FLA prefill kernels on top of cuda_gdn (models/qwen3_5_moe/gdn.py): 'wu' = recompute_w_u +
    # chunk_h + chunk_o, 'ho' = chunk_h + chunk_o (FreeToken also reads '1' as 'wu'; production exports the name).
    'cuda_gdn_fla': ('QWEN36_CUDA_GDN_FLA', ('wu', 'ho'), 'cuda_gdn'),
    # moe/slot_refill.py policies (FreeToken default: last_chunk).
    'layer_major_refill_policy': ('QWEN36_LAYER_MAJOR_REFILL_POLICY', ('last_chunk', 'restore'), 'layer_major_refill'),
    # Short-prompt route: layer-major only from this many uncached prompt tokens (engine/layer_major.py
    # min_tokens); at most chunk + 1 (validate_profile_schema).
    'layer_major_min_tokens': ('QWEN36_LAYER_MAJOR_MIN_TOKENS', range(0, 2147483648), 'layer_major_prefill'),
    # Optional attention batching of layer-major prefill (QWEN36_LAYER_MAJOR_ATTN_BATCH=<k>, e.g. 2; under test).
    # engine/layer_major.py attn_batch() honours 1-4 only and silently runs k=1 for anything else.
    'layer_major_attn_batch': ('QWEN36_LAYER_MAJOR_ATTN_BATCH', tuple(range(1, 5)), 'layer_major_prefill'),
    # C1 (moe/offload_cache.py): a non-layer-major chunk shorter than this many tokens fetches only the experts its
    # router selected (device-side lists); measured together with C0, break-even ~170 tokens, release value 160.
    'prefill_selected': ('QWEN36_PREFILL_SELECTED', range(1, 512), 'prefill_copy_priority'),
    # D6 (moe/offload_cache.py): after each layer's GEMM of a non-layer-major chunk, D2D this many of the chunk's
    # most-routed experts from the double buffer into LRU decode slots; same bytes, other residency (teacher gate).
    'prefill_warm_decode': ('QWEN36_PREFILL_WARM_DECODE', range(1, 65), 'prefill_copy_priority'),
}.items() if spec[2] in SWITCH_ENV}
# Layer-major + hit-D2D requires a positive min-tokens threshold to separate short chunked
# requests from epochs that borrow decode slots. Layer-major + CUDA NVFP4 remains allowed.
SWITCH_EXCLUDED_FLAGS = {'layer_major_prefill': frozenset({'--moe-prefill-hit-d2d'})}
REQUIRED_FIELDS = frozenset({'context', 'chunk', 'memory_ratio', 'kv_dtype', 'flags', 'qualified'})
OPTIONAL_FIELDS = frozenset({'nvfp4_prefill', 'prefix_margin', *FEATURE_SWITCHES, *VALUE_FIELDS})
# Keep identical to optimization/release.py FILES (checked by CPU tests).
MANAGED_FILES = frozenset({'guard.py', 'ingress.py', 'request_metrics.py', 'serve_guarded.py',
                           'readiness.py', 'placement.py', 'keepalive.py', 'metrics_summary.py',
                           'profile-runtime.py', 'profiles.json', 'start-inference.sh'})
RUNTIME_TREE = 'release/python/freetoken'
RUNTIME_PREFIX = RUNTIME_TREE + '/'
LIBRARY_DIR = 'release/lib'
GATE_LIBRARY = 'release/lib/libqwen_attention_gate.so'
# Bit-exact CUDA NVFP4 MoE prefill (same bits as Triton): only with receipt evidence for the profile.
NVFP4_LIBRARY = 'release/lib/libqwen_nvfp4_bitexact.so'
# Bit-exact CUDA GDN kernels and FP8 decode GEMV: only with receipt evidence for the profile.
GDN_LIBRARY = 'release/lib/libqwen36_gdn.so'
FP8_GEMV_LIBRARY = 'release/lib/libqwen_fp8_gemv.so'
OPROJ_GEMV_LIBRARY = 'release/lib/libqwen_oproj_gemv.so'
MOE_DECODE_LIBRARY = 'release/lib/libqwen_moe_decode.so'
# switch -> (environment variable FreeToken's ctypes loader reads, release path); nothing else ships.
NATIVE_LIBRARIES = {'fused_attention_gate': ('QWEN36_FUSED_ATTN_GATE_LIBRARY', GATE_LIBRARY),
                    'cuda_nvfp4_prefill': ('QWEN36_CUDA_NVFP4_LIBRARY', NVFP4_LIBRARY),
                    'cuda_gdn': ('QWEN36_CUDA_GDN_LIBRARY', GDN_LIBRARY),
                    'cuda_fp8_gemv': ('QWEN36_FP8_GEMV_LIBRARY', FP8_GEMV_LIBRARY),
                    'cuda_oproj_gemv': ('QWEN36_OPROJ_GEMV_LIBRARY', OPROJ_GEMV_LIBRARY),
                    'cuda_moe_decode': ('QWEN36_MOE_DECODE_LIBRARY', MOE_DECODE_LIBRARY)}
LIBRARY_ENV = {path: env for env, path in NATIVE_LIBRARIES.values()}
NATIVE_LIBRARY_KEYS = frozenset({'env', 'sha256', 'abi'})
# ABI records the out-of-process probe must reproduce exactly. The value type selects what is probed:
# True -> the symbol exists, int -> an int f(void) constant, str -> a const char *f(void) constant;
# libcuda_loaded: the probe process must not map the CUDA driver (nothing initializes CUDA there).
# moe/cuda_nvfp4.py CudaPrefill: two entry points plus fp32 NVFP4 globals (libcudart linked, not started).
NVFP4_ABI = {'qwen_nvfp4_prefill': True, 'qwen_nvfp4_cuda_error': True, 'qwen_nvfp4_global_bits': 32,
             'libcuda_loaded': False}
# kernel/cuda_gdn.py GdnLib: qwen_gdn_abi_version() == 2 and every entry point it binds, including the fused FLA
# kernels it binds even with QWEN36_CUDA_GDN_FLA off (static cudart).
GDN_ABI = {'qwen_gdn_abi_version': 2, 'qwen_gdn_conv_qkv': True, 'qwen_gdn_gate_cumsum': True,
           'qwen_gdn_gated_norm': True, 'qwen_gdn_decode_fused': True, 'qwen_gdn_decode_cluster': True,
           'qwen_gdn_silu_eval': True, 'qwen_gdn_silu_points': True, 'qwen_gdn_silu_compare': True,
           'qwen_gdn_gate_eval': True, 'qwen_gdn_fla_chunk_ho': True, 'qwen_gdn_fla_wu_chunk_ho': True,
           'libcuda_loaded': False}
# kernel/cuda_fp8_gemv.py pins ABI 2, MAX_M 8 and the CUDA source sha256 (static cudart);
# record from work/wf-fp8gemv/results/release-tooling.json.
FP8_GEMV_ABI = {'qwen_fp8_gemv_rowwise': True, 'qwen_fp8_gemv_init': True, 'qwen_fp8_gemv_error': True,
                'qwen_fp8_gemv_abi_version': 2, 'qwen_fp8_gemv_max_m': 8,
                'qwen_fp8_gemv_source_sha256': '9844420833b0c33c5b5fb73280f63527597306adb49b979731ac1fe89ac708fc',
                'libcuda_loaded': False}
# kernel/cuda_oproj_gemv.py / cuda_moe_decode.py pin ABI 1 and the CUDA source sha256 (static cudart);
# records from kernels/decode-oproj and kernels/decode-moe build-info.txt.
OPROJ_GEMV_ABI = {'qwen_oproj_tensorwise': True, 'qwen_oproj_init': True, 'qwen_oproj_error': True,
                  'qwen_oproj_abi_version': 1, 'qwen_oproj_max_m': 8, 'qwen_oproj_splits': 3,
                  'qwen_oproj_source_sha256': 'ff47a7d57830fa593d633a0482f26adc55724f03727ec3d80725f415146a7431',
                  'libcuda_loaded': False}
MOE_DECODE_ABI = {'qwen_moe_decode_gemv': True, 'qwen_moe_decode_error': True, 'qwen_moe_decode_abi_version': 1,
                  'qwen_moe_decode_source_sha256': '088bb27525a60b602d9ebc84d1d6810343747fec116fa9c288b031cc5437c95a',
                  'libcuda_loaded': False}
LIBRARY_ABI = {NVFP4_LIBRARY: NVFP4_ABI, GDN_LIBRARY: GDN_ABI, FP8_GEMV_LIBRARY: FP8_GEMV_ABI,
               OPROJ_GEMV_LIBRARY: OPROJ_GEMV_ABI, MOE_DECODE_LIBRARY: MOE_DECODE_ABI}
ABI_PROBE_TIMEOUT = 60
# Same library search path as start-inference.sh (libcudart.so.13 of the pinned toolkit).
ABI_PROBE_LIBRARY_PATH = ('cuda-13.4/lib', 'cuda-13.4/lib64')
ABI_PROBE_SYSTEM_PATH = '/usr/lib/wsl/lib'
QUALIFICATION_RECEIPT = 'release/qualification.json'
SELECTED_PROFILE = 'config/selected-profile'
ACCEPTANCE_STATUSES = ('accepted', 'accepted_with_known_limits')
PROFILES_KEYS = frozenset({'schema', 'default', 'profiles'})
QUALIFICATION_KEYS = frozenset({'schema', 'checkpoint', 'profiles', 'source_files', 'environment',
                                'kernel_provenance', 'native_libraries'})
RECORD_KEYS = frozenset({'configuration_sha256', 'acceptance_status', 'known_limitations', 'evidence'})
# CUDA library switches accepted only with bit-exact evidence bound to the released library bytes; the
# receipt record key is the switch name.
CUDA_EVIDENCE_SWITCHES = ('cuda_nvfp4_prefill', 'cuda_gdn', 'cuda_fp8_gemv', 'cuda_gdn_fla', 'cuda_oproj_gemv',
                          'cuda_moe_decode')
CUDA_EVIDENCE_LIBRARIES = {**{k:v[1] for k,v in NATIVE_LIBRARIES.items()}, 'cuda_gdn_fla':GDN_LIBRARY}

def cuda_evidence_enabled(profile, switch):
    return profile.get(switch) in ('wu', 'ho') if switch == 'cuda_gdn_fla' else profile.get(switch) is True

# Experimental switches a qualified profile may enable only once listed in QUALIFIABLE_EXPERIMENTAL, and then
# only with these receipt evidence lists (each a non-empty set of listed evidence files): the GPU component
# report, the run receipts and schema-2 teacher reports of its byte-exact gate, and the paired A/B report.
EXPERIMENTAL_EVIDENCE = {'decode_shared_overlap': ('component_reports', 'run_receipts', 'bitexact_reports', 'ab_reports')}
# T3 is qualifiable only with that evidence: listing it here admits the profile schema, but release and start
# refuse a receipt without all four lists (_experimental_evidence_valid), and make-qualification.py, the only
# receipt producer, checks their contents (test_decode_overlap_gpu.py 4/4 without skips, run receipts bound to
# the staged bytes, a 125/125 max_abs 0 teacher report bound to those runs, a compare-ab PASS; GPU_VALIDATION.md
# section 5). Without that evidence a qualified profile still cannot enable it.
QUALIFIABLE_EXPERIMENTAL = frozenset({'decode_shared_overlap'})

CUDA_EVIDENCE_KEY = 'cuda_nvfp4_prefill'
CUDA_EVIDENCE_KEYS = frozenset({'library', 'library_sha256', 'run_receipts', 'bitexact_reports'})
EVIDENCE_KEYS = frozenset({'path', 'sha256'})
JSON_LIMIT = 4 * 1024 * 1024
# Executable code the release tree does not carry. FreeToken loads prebuilt kernels from
# the venv kernel cache by file name before it would compile release sources
# (kernel/utils.py _load_prebuilt); torch/triton/flashinfer/sglang-kernel/tvm-ffi come
# from the venv. start-inference.sh runs $ROOT/.venv/bin/python.
VENV_SITE_PACKAGES = '.venv/lib/python3.12/site-packages'
RUNTIME_PACKAGES = ('torch', 'triton', 'flashinfer-python', 'sglang-kernel', 'transformers',
                    'apache-tvm-ffi', 'freetoken-kernel-cache')
KERNEL_CACHE_PACKAGE = 'freetoken_kernel_cache'
ENVIRONMENT_KEYS = frozenset({'packages', 'package_records', 'kernel_cache'})
KERNEL_CACHE_JIT_DIR = 'jit_cache'
# FreeToken kernel provenance (kernel/kernel_provenance.py, stdlib only) and its manifest, both
# part of the receipt-bound runtime tree.
KERNEL_TREE = RUNTIME_PREFIX + 'kernel/'
KERNEL_PROVENANCE_MODULE = KERNEL_TREE + 'kernel_provenance.py'
KERNEL_PROVENANCE_MANIFEST = KERNEL_TREE + 'kernel_cache_manifest.json'
KERNEL_PROVENANCE_KEYS = frozenset({'required_extensions', 'required_modules', 'inventory_sha256', 'inventory'})
KERNEL_INVENTORY_KEYS = frozenset({'schema', 'manifest_sha256', 'modules', 'extensions'})
KERNEL_MODULE_KEYS = frozenset({'status', 'so_sha256', 'source_sha256'})
KERNEL_EXTENSION_KEYS = frozenset({'status', 'so_name', 'so_sha256', 'source_sha256', 'sources'})
# Release and start accept only verified prebuilts: a private JIT build is not receipt-bound.
KERNEL_MODULE_STATUSES = ('prebuilt',)
# Every module must be bound by the released kernel/kernel_cache_manifest.json, not by a record a cache
# build ships next to the .so files (kernel_provenance lookup origin 'cache-provenance').
KERNEL_MODULE_ORIGIN = 'manifest'
# switch -> JIT kernel-cache modules it executes. A qualified profile that switches it on needs each module
# as a verified prebuilt in the venv kernel cache; without one FreeToken would JIT-build it into a private,
# receipt-unbound directory. aot.default_kernel_specs() lists the module, so a rebuilt release cache ships
# it (and the manifest must bind it) whether or not the switch is on.
KERNEL_MODULE_SWITCHES = {'decode_gather': ('freetoken__decode_gather_tma_8_3_16384',)}
# The same for production flags and value fields. Hit-D2D prefill (moe/offload_cache.py _prefetch_split) gathers
# resident experts with fast_index_copy_multi (64 blocks per bank) and copies misses with batch_memcpy; C1
# (prefill_selected) runs the decode-gather module plus that gather. Without a prebuilt FreeToken logs one
# warning and silently keeps the slower copies, so a qualified profile must find each as a manifest-bound prebuilt.
KERNEL_MODULE_FLAGS = {'--moe-prefill-hit-d2d': ('freetoken__batch_memcpy', 'freetoken__fast_index_copy_multi_1024_64')}
KERNEL_MODULE_FIELDS = {'prefill_selected': ('freetoken__decode_gather_tma_8_3_16384',
                                             'freetoken__fast_index_copy_multi_1024_64')}
# serve_argv/configure_environment run every production profile with --moe-strategy offload;
# _cpu_moe is required only by the cpu/hybrid strategies.
PRODUCTION_STRATEGY = 'offload'
CPU_MOE_STRATEGIES = frozenset({'cpu', 'hybrid'})
_CACHE_SO = re.compile(r'jit_cache/([^/]+)/([^/]+)\.so')
# Served weights (download-checkpoint.py writes both documents after verifying every file).
CHECKPOINT_MANIFEST = 'checkpoint-manifest.json'
CHECKPOINT_DOWNLOAD_RECEIPT = 'download-receipt.json'
CHECKPOINT_RECEIPT_KEYS = frozenset({'file', 'bytes', 'sha256', 'status'})
CHECKPOINT_STATUSES = ('downloaded_verified', 'verified_existing')
# Config/index/tokenizer files are re-hashed at every gate; larger weight shards are bound
# by exact size plus the LFS sha256 of the pinned revision (not re-read, ~23 GB).
CHECKPOINT_HASH_LIMIT = 64 * 1024 * 1024
# Pinned FLA chunk_delta_h kernel configuration (equal to the source defaults
# used by every qualification run); environment cannot retune it in production.
GDN_BASELINE = {'SGLANG_GDN_CHUNK_H_BV': '32',
                'SGLANG_GDN_CHUNK_H_NUM_WARPS': '4',
                'SGLANG_GDN_CHUNK_H_NUM_STAGES': '2'}
# Production never JIT-compiles a kernel module at serve time. STRICT_PROVENANCE enforces the
# manifest/recipe/binary binding before loading; DISABLE_JIT alone only forbids builds. The loader loads prebuilts from the venv kernel cache (no
# tvm-ffi build, so no CUDA arch detection) and raises instead of compiling. Inherited TVM_FFI_* build knobs
# are cleared; only decode_gather pins TVM_FFI_CUDA_ARCH_LIST (SWITCH_ENV, inert here). Every module a
# profile runs must therefore be a verified prebuilt (kernel provenance; KERNEL_MODULE_SWITCHES).
KERNEL_LOADER_ENV = {'FREETOKEN_DISABLE_JIT': '1', 'FREETOKEN_STRICT_PROVENANCE': '1'}
# The profile is the complete behavior configuration. Inherited experiment,
# kernel and allocator knobs are removed; only the unit's persistent key stays.
PRESERVED_ENV = frozenset({'QWEN36_API_KEY'})
CLEARED_ENV_PREFIXES = ('QWEN36_', 'FREETOKEN_', 'SGLANG_', 'FLA_', 'LIGHTLLM_', 'DSV4_', 'TVM_FFI_')
CLEARED_ENV_NAMES = frozenset({'CPATH', 'GDN_RECOMPUTE_SUPPRESS_LEVEL',
                               'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF'})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_profile_sha256(profile):
    """Hash all behavior fields; qualified is an administrative release decision."""
    _require(type(profile) is dict, 'Profile must be an object')
    value = {key: item for key, item in profile.items() if key != 'qualified'}
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def validate_profile_schema(name, profile):
    _require(type(name) is str and name in CONTEXTS and type(profile) is dict,
             'Unknown production profile')
    keys = set(profile)
    _require(REQUIRED_FIELDS <= keys and keys <= REQUIRED_FIELDS | OPTIONAL_FIELDS,
             'Unsupported production profile fields')
    _require(type(profile['qualified']) is bool, 'qualified must be a boolean')
    _require(type(profile['context']) is int and profile['context'] == CONTEXTS[name],
             'Profile context does not match its fixed name')
    _require(type(profile['chunk']) is int and profile['chunk'] in CHUNKS, 'Unapproved prefill chunk')
    ratio = profile['memory_ratio']
    _require(type(ratio) is float and math.isfinite(ratio) and 0 < ratio < MAX_MEMORY_RATIO,
             'Invalid memory ratio')
    _require(type(profile['kv_dtype']) is str and profile['kv_dtype'] == 'bf16',
             'Production supports BF16 KV only')
    if 'nvfp4_prefill' in profile:
        _require(type(profile['nvfp4_prefill']) is str and profile['nvfp4_prefill'] == 'triton',
                 'Production supports qualified Triton MoE only')
    flags = profile['flags']
    _require(type(flags) is list and all(type(flag) is str for flag in flags), 'Invalid production flags')
    _require(len(flags) == len(set(flags)) and set(flags) <= ALLOWED_FLAGS, 'Unsupported production flags')
    for key in FEATURE_SWITCHES:
        if key in profile:
            _require(type(profile[key]) is bool, 'Feature switches must be booleans')
    _require(not (profile.get('cuda_nvfp4_prefill') is True and 'nvfp4_prefill' in profile),
             'cuda_nvfp4_prefill and nvfp4_prefill are exclusive')
    for switch, needed in SWITCH_REQUIRES.items():
        _require(profile.get(switch) is not True or profile.get(needed) is True, switch + ' requires ' + needed)
    for switch, excluded in SWITCH_EXCLUDED_FLAGS.items():
        separated = switch == 'layer_major_prefill' and type(profile.get('layer_major_min_tokens')) is int and profile['layer_major_min_tokens'] > 0
        _require(profile.get(switch) is not True or not excluded & set(flags) or separated,
                 switch + ' cannot take effect with ' + ' '.join(sorted(excluded & set(flags))))
    for field, (_, allowed, switch) in VALUE_FIELDS.items():
        if field in profile:
            value = profile[field]
            _require(type(value) in (str, int) and type(value) is type(allowed[0]) and value in allowed,
                     'Invalid ' + field)
            _require(profile.get(switch) is True, field + ' requires ' + switch)
    # A threshold above chunk + 1 would send multi-chunk prompts down chunked prefill (measured slower).
    _require(profile.get('layer_major_min_tokens', 0) <= profile['chunk'] + 1,
             'layer_major_min_tokens must not exceed chunk + 1')
    # C1 fetches the routed experts through the prebuilt TMA decode-gather module (the same instantiation as
    # decode_gather; FREETOKEN_DISABLE_JIT=1): without it C1 logs one warning and silently keeps the old path.
    _require('prefill_selected' not in profile or profile.get('decode_gather') is True,
             'prefill_selected requires decode_gather')
    _require(not (profile.get('decode_shared_overlap') is True and profile.get('layer_major_refill') is True),
             'decode_shared_overlap does not support layer_major_refill')
    for switch in EXPERIMENTAL_EVIDENCE:
        _require(not (profile.get(switch) is True and profile['qualified']) or switch in QUALIFIABLE_EXPERIMENTAL,
                 switch + ' is experimental and cannot be qualified')
    if 'prefix_margin' in profile:
        _require(type(profile['prefix_margin']) is int and 1 <= profile['prefix_margin'] <= 64,
                 'Invalid prefix margin')
    return profile


def _profiles_schema(profiles):
    _require(type(profiles) is dict and set(profiles) == PROFILES_KEYS, 'Invalid profiles document')
    _require(type(profiles['schema']) is int and profiles['schema'] == 1, 'Unsupported profiles schema')
    _require(type(profiles['profiles']) is dict and set(profiles['profiles']) == set(CONTEXTS),
             'Production needs baseline64, main128 and long256 profiles')
    _require(type(profiles['default']) is str and profiles['default'] in CONTEXTS, 'Invalid default profile')
    for name, profile in profiles['profiles'].items():
        validate_profile_schema(name, profile)


def _hash_valid(value):
    return (type(value) is str and len(value) == 64
            and all(char in '0123456789abcdef' for char in value))


def _relative_valid(value):
    return (type(value) is str and bool(value) and '\\' not in value
            and not any(ord(char) < 32 or ord(char) == 127 for char in value)
            and not pathlib.PurePosixPath(value).is_absolute()
            and all(part not in ('', '.', '..') for part in value.split('/')))


def _qualified_names(profiles):
    return {name for name, profile in profiles['profiles'].items() if profile['qualified'] is True}


def required_libraries(profiles):
    """Native ctypes libraries a release may carry: only those a qualified profile switches on."""
    qualified = _qualified_names(profiles)
    return {path for switch, (_, path) in NATIVE_LIBRARIES.items()
            if any(profiles['profiles'][name].get(switch) is True for name in qualified)}


def profile_strategy(profile):
    """MoE strategy a production profile runs (fixed; see serve_argv)."""
    return PRODUCTION_STRATEGY


def required_kernel_extensions(profiles):
    """FreeToken extensions the qualified profiles execute: _pinned_tensor always, _cpu_moe for cpu/hybrid."""
    required = {'_pinned_tensor'}
    if any(profile_strategy(profiles['profiles'][name]) in CPU_MOE_STRATEGIES for name in _qualified_names(profiles)):
        required.add('_cpu_moe')
    return sorted(required)


def required_kernel_modules(profiles):
    """JIT kernel-cache modules a qualified profile switches on (KERNEL_MODULE_SWITCHES, KERNEL_MODULE_FLAGS,
    KERNEL_MODULE_FIELDS): each must be a manifest-bound prebuilt in the venv cache, never a private JIT build."""
    selected = [profiles['profiles'][name] for name in _qualified_names(profiles)]
    modules = set()
    for profile in selected:
        for switch, names in KERNEL_MODULE_SWITCHES.items():
            if profile.get(switch) is True:
                modules.update(names)
        for flag, names in KERNEL_MODULE_FLAGS.items():
            if flag in profile['flags']:
                modules.update(names)
        for field, names in KERNEL_MODULE_FIELDS.items():
            if field in profile:
                modules.update(names)
    return sorted(modules)


def sha256_regular_file(path):
    """Hash one regular file without following a final symlink."""
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(os.fspath(path), flags)
    try:
        _require(stat.S_ISREG(os.fstat(descriptor).st_mode), 'Qualification source is not a regular file')
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                return digest.hexdigest()
            digest.update(block)
    finally:
        os.close(descriptor)


def _environment_valid(value):
    return (type(value) is dict and set(value) == ENVIRONMENT_KEYS
            and type(value['packages']) is dict and set(value['packages']) == set(RUNTIME_PACKAGES)
            and all(type(item) is str and bool(item) for item in value['packages'].values())
            and type(value['package_records']) is dict and set(value['package_records']) == set(RUNTIME_PACKAGES)
            and all(_hash_valid(item) for item in value['package_records'].values())
            and type(value['kernel_cache']) is dict and bool(value['kernel_cache'])
            and all(_relative_valid(path) and _hash_valid(digest) for path, digest in value['kernel_cache'].items()))


def _validate_qualification(profiles, qualification, resolve_source, expected_sources, environment):
    _require(type(qualification) is dict and set(qualification) == QUALIFICATION_KEYS,
             'Release qualification receipt is required')
    _require(type(qualification['schema']) is int and qualification['schema'] == 1,
             'Unsupported qualification schema')
    _require(qualification['checkpoint'] == CHECKPOINT, 'Checkpoint receipt mismatch')
    _require(_environment_valid(qualification['environment']), 'Invalid runtime environment in receipt')
    if environment is not None:
        _require(qualification['environment'] == environment, 'Runtime environment differs from qualification')
    accepted = _qualified_names(profiles)
    records = qualification['profiles']
    _require(type(records) is dict and set(records) == accepted,
             'Qualification must cover exactly the enabled profiles')
    for name in sorted(accepted):
        record = records[name]
        _require(type(record) is dict
                 and RECORD_KEYS <= set(record) <= RECORD_KEYS | set(CUDA_EVIDENCE_SWITCHES) | set(EXPERIMENTAL_EVIDENCE),
                 'Invalid qualification record')
        _require(record['configuration_sha256'] == canonical_profile_sha256(profiles['profiles'][name]),
                 'Profile differs from qualification receipt')
        status, limits = record['acceptance_status'], record['known_limitations']
        _require(status in ACCEPTANCE_STATUSES and type(limits) is list
                 and all(type(item) is str and bool(item.strip()) for item in limits),
                 'Invalid acceptance limitations')
        _require((status == 'accepted_with_known_limits') == bool(limits),
                 'Acceptance status must disclose known limitations')
        evidence = record['evidence']
        _require(type(evidence) is list and bool(evidence), 'Qualification needs evidence references')
        _require(all(type(row) is dict and set(row) == EVIDENCE_KEYS and _relative_valid(row['path'])
                     and _hash_valid(row['sha256']) for row in evidence), 'Invalid evidence references')
        _require(len({row['path'] for row in evidence}) == len(evidence), 'Duplicate evidence references')
    manifest = qualification['source_files']
    _require(type(manifest) is dict, 'Missing source manifest')
    for name in sorted(accepted):
        _cuda_evidence_valid(name, profiles['profiles'][name], records[name], manifest)
        _experimental_evidence_valid(name, profiles['profiles'][name], records[name])
    _require(type(manifest) is dict and bool(manifest), 'Missing source manifest')
    _require(all(_relative_valid(path) and _hash_valid(digest) for path, digest in manifest.items()),
             'Invalid source manifest')
    _require(QUALIFICATION_RECEIPT not in manifest, 'Receipt cannot hash itself')
    _require(MANAGED_FILES <= set(manifest), 'Managed source manifest is incomplete')
    _require(all(path in MANAGED_FILES or path.startswith(RUNTIME_PREFIX) or path.startswith(LIBRARY_DIR + '/')
                 for path in manifest), 'Source manifest contains unmanaged paths')
    _require(any(path.startswith(RUNTIME_PREFIX) and path.endswith('.py') for path in manifest),
             'Runtime source manifest is missing')
    _require({path for path in manifest if path.startswith(LIBRARY_DIR + '/')} == required_libraries(profiles),
             'Only selected qualified native libraries may be released')
    _native_libraries_valid(qualification['native_libraries'], profiles, manifest)
    _kernel_provenance_valid(qualification['kernel_provenance'], profiles, qualification)
    if expected_sources is not None:
        _require(set(manifest) == set(expected_sources), 'Source manifest coverage differs from release')
    _require(callable(resolve_source), 'Source resolver is required')
    for relative in sorted(manifest):
        resolved = resolve_source(relative)
        _require(resolved is not None, 'Qualification source unavailable')
        path = pathlib.Path(resolved)
        try:
            actual = sha256_regular_file(path)
        except FileNotFoundError:
            raise ValueError('Qualification source unavailable') from None
        _require(actual == manifest[relative], 'Source differs from qualification')


def validate_profiles(profiles, qualification, resolve_source, *, expected_sources=None, environment=None):
    """Side-effect-free release preflight. Resolver must refuse unstaged sources.

    Evidence references are trusted owner-reviewed receipts, not automatic quality
    scores. The exporter/release owner checks report hashes; this binds config and
    source bytes, and (when ``environment`` is given, as release and start do) the
    runtime_environment() fingerprint.
    """
    _profiles_schema(profiles)
    _require(profiles['profiles'][profiles['default']]['qualified'] is True,
             'Default profile has not passed release qualification')
    _validate_qualification(profiles, qualification, resolve_source, expected_sources, environment)
    return profiles


def validate_profile(name, profiles, qualification, resolve_source, *, expected_sources=None, environment=None):
    validate_profiles(profiles, qualification, resolve_source, expected_sources=expected_sources,
                      environment=environment)
    _require(type(name) is str and name in profiles['profiles']
             and profiles['profiles'][name]['qualified'] is True,
             'Profile has not passed release qualification')
    return profiles['profiles'][name]


def _production_source(relative):
    """Installed-tree resolver: ROOT only, no symlink in any component below ROOT."""
    _require(_relative_valid(relative), 'Invalid production source path')
    base = ROOT.resolve()
    path = ROOT / relative
    _require(path.resolve() == base.joinpath(relative), 'Production source escapes service root')
    return path


def regular_files(top, relative_base):
    """Regular files below top (no __pycache__/.pyc); any symlink is refused, never followed.

    Shared by the runtime inventory and the release staging inventory.
    """
    found = set()

    def unreadable(error):
        # os.walk silently skips unreadable directories by default; an incomplete
        # inventory must refuse instead of shrinking the release/receipt coverage.
        raise error

    for directory, dirnames, filenames in os.walk(top, onerror=unreadable, followlinks=False):
        for entry in list(dirnames):
            _require(not os.path.islink(os.path.join(directory, entry)), 'Symlink in production tree')
        dirnames[:] = [entry for entry in dirnames if entry != '__pycache__']
        for entry in filenames:
            full = os.path.join(directory, entry)
            mode = os.lstat(full).st_mode
            _require(not stat.S_ISLNK(mode), 'Symlink in production tree')
            if entry.endswith('.pyc'):
                continue
            _require(stat.S_ISREG(mode), 'Unsupported file type in production tree')
            found.add(pathlib.Path(full).relative_to(relative_base).as_posix())
    return found


def _production_sources(profiles):
    """Exact inventory the installed receipt must cover (receipt itself excluded)."""
    base = ROOT
    result = set(MANAGED_FILES)
    python_dir = base / 'release/python'
    _require(python_dir.is_dir() and not python_dir.is_symlink(), 'Installed runtime tree is missing')
    # PYTHONPATH points at release/python: nothing else may shadow imports there.
    _require({entry.name for entry in python_dir.iterdir()} <= {'freetoken', '__pycache__'},
             'Unexpected entries next to the installed runtime package')
    runtime = base / RUNTIME_TREE
    _require(runtime.is_dir() and not runtime.is_symlink(), 'Installed runtime tree is missing')
    result |= regular_files(runtime, base)
    libraries = base / LIBRARY_DIR
    if libraries.exists() or libraries.is_symlink():
        _require(libraries.is_dir() and not libraries.is_symlink(), 'Invalid native library directory')
        # Every present library must be covered: stale/unselected native code refuses start.
        result |= regular_files(libraries, base)
    result |= required_libraries(profiles)
    return result


def _strict_json(data):
    def pairs(items):
        keys = [key for key, _ in items]
        _require(len(keys) == len(set(keys)), 'Duplicate JSON keys')
        return dict(items)

    def constant(value):
        raise ValueError('Non-finite JSON number')

    _require(len(data) <= JSON_LIMIT, 'JSON document too large')
    return json.loads(data.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)


def _read_regular(path):
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(os.fspath(path), flags)
    try:
        _require(stat.S_ISREG(os.fstat(descriptor).st_mode), 'Not a regular file')
        chunks = []
        size = 0
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                return b''.join(chunks)
            size += len(block)
            _require(size <= JSON_LIMIT, 'File too large')
            chunks.append(block)
    finally:
        os.close(descriptor)


def read_json(path):
    """Strict JSON (no duplicate keys, no NaN/Infinity) from a regular non-symlink file."""
    return _strict_json(_read_regular(path))


validate_profiles_document = _profiles_schema


def _distribution_name(value):
    return re.sub(r'[-_.]+', '-', value).lower()


def runtime_environment(site_packages=None):
    """Fingerprint of the venv code the release tree does not carry; nothing is imported.

    packages: installed versions (same representation as experiment run receipts);
    package_records: sha256 of each dist-info RECORD (a rebuilt/reinstalled wheel differs);
    kernel_cache: sha256 of every file of the prebuilt FreeToken kernel cache package.
    """
    base = ROOT / VENV_SITE_PACKAGES if site_packages is None else pathlib.Path(site_packages)
    _require(base.is_dir() and not base.is_symlink(), 'Runtime site-packages are missing')
    infos = {}
    for entry in os.listdir(base):
        if entry.endswith('.dist-info'):
            infos.setdefault(_distribution_name(entry[:-len('.dist-info')].partition('-')[0]), []).append(base / entry)
    packages, records = {}, {}
    for name in RUNTIME_PACKAGES:
        found = infos.get(name, [])
        _require(len(found) == 1, 'Runtime package must be installed exactly once: ' + name)
        info = found[0]
        _require(info.is_dir() and not info.is_symlink(), 'Invalid runtime package metadata: ' + name)
        metadata = importlib.metadata.PathDistribution(info).metadata
        version = metadata.get('Version')
        _require(_distribution_name(metadata.get('Name') or '') == name and type(version) is str and bool(version),
                 'Runtime package metadata differs from its directory: ' + name)
        packages[name] = version
        records[name] = sha256_regular_file(info / 'RECORD')
    cache = base / KERNEL_CACHE_PACKAGE
    _require(cache.is_dir() and not cache.is_symlink(), 'Prebuilt kernel cache package is missing')
    kernel_cache = {relative: sha256_regular_file(cache / relative) for relative in sorted(regular_files(cache, cache))}
    _require(bool(kernel_cache), 'Prebuilt kernel cache package is empty')
    return {'packages': packages, 'package_records': records, 'kernel_cache': kernel_cache}

# ---------------------------------------------------------------------------- native libraries


def _cuda_evidence_valid(name, profile, record, manifest):
    """Each CUDA library switch (CUDA_EVIDENCE_SWITCHES) needs bit-exact evidence bound to its released
    library (else refusal); evidence for a switch the profile does not enable is refused too."""
    listed = {row['path'] for row in record['evidence']}
    for switch in CUDA_EVIDENCE_SWITCHES:
        library = CUDA_EVIDENCE_LIBRARIES[switch]
        if not cuda_evidence_enabled(profile, switch):
            _require(switch not in record, 'Bit-exact evidence for a switch the profile does not enable: ' + switch)
            continue
        evidence = record.get(switch)
        keys = CUDA_EVIDENCE_KEYS | ({'variant','component_reports'} if switch == 'cuda_gdn_fla' else set())
        _require(type(evidence) is dict and set(evidence) == keys,
                 switch + ' needs bit-exact evidence for its library in the receipt: ' + name)
        _require(evidence['library'] == library and _hash_valid(evidence['library_sha256'])
                 and evidence['library_sha256'] == manifest.get(library),
                 switch + ' bit-exact evidence is not bound to the released library: ' + name)
        if switch == 'cuda_gdn_fla':
            _require(evidence['variant'] == profile['cuda_gdn_fla'], 'FLA evidence variant mismatch')
        for key in ('run_receipts', 'bitexact_reports') + (('component_reports',) if switch == 'cuda_gdn_fla' else ()):
            items = evidence[key]
            _require(type(items) is list and bool(items) and len(set(map(str, items))) == len(items)
                     and all(type(item) is str and item in listed for item in items),
                     switch + ' bit-exact evidence must reference listed evidence files: ' + key)


def _experimental_evidence_valid(name, profile, record):
    """A qualified experimental switch (EXPERIMENTAL_EVIDENCE) needs its evidence lists in the record, each
    referencing listed evidence files; evidence for a switch the profile does not enable is refused."""
    listed = {row['path'] for row in record['evidence']}
    for switch, keys in EXPERIMENTAL_EVIDENCE.items():
        if profile.get(switch) is not True:
            _require(switch not in record, 'Evidence for a switch the profile does not enable: ' + switch)
            continue
        _require(switch in QUALIFIABLE_EXPERIMENTAL, switch + ' is experimental and cannot be qualified')
        evidence = record.get(switch)
        _require(type(evidence) is dict and set(evidence) == set(keys),
                 switch + ' needs component, teacher and A/B evidence in the receipt: ' + name)
        for key in keys:
            items = evidence[key]
            _require(type(items) is list and bool(items) and len(set(map(str, items))) == len(items)
                     and all(type(item) is str and item in listed for item in items),
                     switch + ' evidence must reference listed evidence files: ' + key)


def _native_libraries_valid(value, profiles, manifest):
    required = required_libraries(profiles)
    _require(type(value) is dict and set(value) == required,
             'Native library receipt must cover exactly the released libraries')
    for relative in sorted(required):
        row = value[relative]
        _require(type(row) is dict and set(row) == NATIVE_LIBRARY_KEYS and row['env'] == LIBRARY_ENV[relative]
                 and _hash_valid(row['sha256']) and row['sha256'] == manifest.get(relative)
                 and row['abi'] == LIBRARY_ABI.get(relative, {}),
                 'Native library binding differs from the receipt: ' + relative)


def _read_bound(path, expected, label):
    """Bytes of one regular non-symlink file that must hash to the receipt value."""
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(os.fspath(path), flags)
    except FileNotFoundError:
        raise ValueError('Qualification source unavailable: ' + label) from None
    try:
        _require(stat.S_ISREG(os.fstat(descriptor).st_mode), 'Qualification source is not a regular file')
        chunks = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(descriptor)
    data = b''.join(chunks)
    _require(hashlib.sha256(data).hexdigest() == expected, 'Source differs from qualification: ' + label)
    return data


_ABI_PROBE = """
import ctypes, json, sys
spec = json.loads(sys.argv[2])
lib = ctypes.CDLL(sys.argv[1])
result = {name: hasattr(lib, name) for name in spec['symbols']}
for names, restype in ((spec['ints'], ctypes.c_int), (spec['strings'], ctypes.c_char_p)):
    for name in names:
        fn = getattr(lib, name, None)
        if fn is None:
            result[name] = None
            continue
        fn.argtypes = []
        fn.restype = restype
        value = fn()
        result[name] = int(value) if restype is ctypes.c_int else (None if value is None else value.decode('ascii', 'replace'))
with open('/proc/self/maps') as maps:
    result['libcuda_loaded'] = any('libcuda.so' in line for line in maps)
print(json.dumps(result, sort_keys=True))
"""


def abi_probe_spec(expected):
    """What the probe calls for an ABI record (LIBRARY_ABI): symbols, int and string constant functions."""
    spec = {'symbols': [], 'ints': [], 'strings': []}
    for name, value in sorted(expected.items()):
        if name == 'libcuda_loaded':
            continue
        _require(type(name) is str and name.isidentifier(), 'Invalid ABI record field')
        kind = 'symbols' if type(value) is bool else 'ints' if type(value) is int else 'strings' if type(value) is str else None
        _require(kind is not None, 'Invalid ABI record value: ' + name)
        spec[kind].append(name)
    return spec


def probe_library_abi(path, expected):
    """ABI of a ctypes library (the fields of its ``expected`` record), probed in a separate CPU-only process.

    Only constant entry points are called (no CUDA call); dlopen may map libcudart but nothing initializes
    it, and the probe reports whether the driver (libcuda) got mapped. The serving process itself never
    loads a library before FreeToken does.
    """
    library_path = [str(ROOT / item) for item in ABI_PROBE_LIBRARY_PATH] + [ABI_PROBE_SYSTEM_PATH]
    env = {'CUDA_VISIBLE_DEVICES': '', 'PATH': '/usr/bin:/bin', 'LC_ALL': 'C',
           'LD_LIBRARY_PATH': ':'.join(library_path)}
    spec = json.dumps(abi_probe_spec(expected), sort_keys=True)
    try:
        done = subprocess.run([sys.executable, '-I', '-S', '-c', _ABI_PROBE, os.fspath(path), spec], env=env,
                              capture_output=True, text=True, timeout=ABI_PROBE_TIMEOUT, cwd='/')
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError('ABI probe could not run: ' + type(error).__name__) from None
    _require(done.returncode == 0, 'Native library could not be loaded for the ABI probe')
    try:
        result = json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise ValueError('Invalid ABI probe output') from None
    _require(type(result) is dict, 'Invalid ABI probe output')
    return result


def probe_native_library(relative, path):
    """Dispatch: the ABI probe of the released library ``relative`` (LIBRARY_ABI) run on ``path``."""
    _require(relative in LIBRARY_ABI, 'No ABI record for native library: ' + str(relative)[:80])
    return probe_library_abi(path, LIBRARY_ABI[relative])


def probe_nvfp4_library(path):
    """ABI of the CUDA NVFP4 ctypes library (qwen_nvfp4_global_bits() is a constant, no CUDA call)."""
    return probe_native_library(NVFP4_LIBRARY, path)


def native_library_record(profiles, source_files, resolve_source):
    """Receipt record of the released ctypes libraries: env variable, sha256, ABI probe result."""
    record = {}
    for relative in sorted(required_libraries(profiles)):
        _require(relative in source_files, 'Native library is not in the source manifest: ' + relative)
        abi = {}
        if relative in LIBRARY_ABI:
            resolved = resolve_source(relative)
            _require(resolved is not None, 'Qualification source unavailable: ' + relative)
            # Bracket the probe with the bound hash: the probed bytes are the receipt bytes.
            _read_bound(pathlib.Path(resolved), source_files[relative], relative)
            abi = probe_native_library(relative, pathlib.Path(resolved))
            _read_bound(pathlib.Path(resolved), source_files[relative], relative)
            _require(abi == LIBRARY_ABI[relative], 'Native library ABI differs (' + relative + '): '
                     + json.dumps(abi, sort_keys=True)[:200])
        record[relative] = {'env': LIBRARY_ENV[relative], 'sha256': source_files[relative], 'abi': abi}
    return record


def verify_native_libraries(profiles, qualification, resolve_source):
    """Release/start: live ABI probe of the bound libraries must equal the receipt."""
    record = native_library_record(profiles, qualification['source_files'], resolve_source)
    _require(record == qualification['native_libraries'], 'Native libraries differ from qualification')
    return record


# ---------------------------------------------------------------------------- kernel provenance


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                                     allow_nan=False).encode()).hexdigest()


def kernel_cache_dir(site_packages=None):
    base = ROOT / VENV_SITE_PACKAGES if site_packages is None else pathlib.Path(site_packages)
    return base / KERNEL_CACHE_PACKAGE / KERNEL_CACHE_JIT_DIR


def kernel_inventory_summary(inventory):
    """Path-independent projection of kernel_provenance.inventory(): the bytes/status the receipt binds."""
    manifest = inventory['manifest']
    return {'schema': 1,
            'manifest_sha256': manifest['sha256'] if manifest else None,
            'modules': {row['name']: {'status': row['status'], 'so_sha256': row.get('so_sha256'),
                                      'source_sha256': row.get('source_sha256')} for row in inventory['modules']},
            'extensions': {row['name']: {'status': row['status'],
                                         'so_name': pathlib.PurePath(row['so_path']).name if row.get('so_path') else None,
                                         'so_sha256': row.get('so_sha256'), 'source_sha256': row.get('source_sha256'),
                                         'sources': row.get('sources')} for row in inventory['extensions']}}


def kernel_inventory_sha256(summary):
    return _canonical_sha256(summary)


def _load_kernel_provenance(source, path):
    """Execute verified kernel_provenance.py bytes as a private module (no import, no bytecode)."""
    module = types.ModuleType('qwen36_kernel_provenance')
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def _kernel_inventory(module, kernel_dir, cache_dir):
    return module.inventory(kernel_dir=kernel_dir, cache_dir=pathlib.Path(cache_dir),
                            manifest_path=pathlib.Path(kernel_dir) / pathlib.PurePosixPath(KERNEL_PROVENANCE_MANIFEST).name,
                            use_default_cache=False)


def kernel_inventory_for_directory(kernel_dir, cache_dir):
    """Inventory summary (no check) of an on-disk kernel dir, e.g. the native qualification tree;
    run receipts may carry kernel_inventory_sha256(...) of it."""
    kernel_dir = pathlib.Path(kernel_dir)
    path = kernel_dir / pathlib.PurePosixPath(KERNEL_PROVENANCE_MODULE).name
    module = _load_kernel_provenance(_read_regular_unbounded(path), path)
    return kernel_inventory_summary(_kernel_inventory(module, kernel_dir, cache_dir))


def _read_regular_unbounded(path):
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(os.fspath(path), flags)
    try:
        _require(stat.S_ISREG(os.fstat(descriptor).st_mode), 'Not a regular file')
        chunks = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                return b''.join(chunks)
            chunks.append(block)
    finally:
        os.close(descriptor)


def kernel_provenance_record(profiles, source_files, resolve_source, cache_dir):
    """kernel_provenance inventory()+check() over exactly the receipt-bound kernel bytes and the venv cache.

    The release kernel subtree is copied (each file hash-verified against ``source_files``) into a
    temporary directory; the verified kernel_provenance.py runs there. Fail-closed: every cache module
    and every required extension must be ``prebuilt``, every module bound by the released manifest, and
    every module a qualified profile switches on (``required_kernel_modules``) present as such a prebuilt.
    Returns the receipt record.
    """
    _require(type(source_files) is dict, 'Missing source manifest')
    kernel = {path[len(KERNEL_TREE):]: digest for path, digest in source_files.items() if path.startswith(KERNEL_TREE)}
    module_name = pathlib.PurePosixPath(KERNEL_PROVENANCE_MODULE).name
    _require(module_name in kernel and KERNEL_PROVENANCE_MANIFEST in source_files,
             'Kernel provenance module or manifest is missing from the release')
    required = required_kernel_extensions(profiles)
    required_modules = required_kernel_modules(profiles)
    with tempfile.TemporaryDirectory(prefix='qwen36-kernel-provenance-') as temp:
        kernel_dir = pathlib.Path(temp) / 'kernel'
        source = None
        for relative in sorted(kernel):
            _require(_relative_valid(relative), 'Invalid kernel path in the source manifest')
            resolved = resolve_source(KERNEL_TREE + relative)
            _require(resolved is not None, 'Qualification source unavailable: ' + KERNEL_TREE + relative)
            data = _read_bound(pathlib.Path(resolved), kernel[relative], KERNEL_TREE + relative)
            target = kernel_dir.joinpath(*relative.split('/'))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            if relative == module_name:
                source = data
        module = _load_kernel_provenance(source, kernel_dir / module_name)
        try:
            inventory = _kernel_inventory(module, kernel_dir, cache_dir)
            module.check(inventory, required_extensions=required, allow_module_statuses=KERNEL_MODULE_STATUSES,
                         allow_unknown_modules=False)
            rows = {row['name']: row for row in inventory['modules']}
            unbound = sorted(str(name)[:80] for name, row in rows.items() if row.get('origin') != KERNEL_MODULE_ORIGIN)
            missing = [name for name in required_modules if name not in rows or rows[name].get('status') != 'prebuilt']
        except module.ProvenanceError as error:
            raise ValueError('Kernel provenance refused: ' + str(error)[:300]) from None
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError('Kernel provenance refused: invalid manifest (' + type(error).__name__ + ')') from None
    _require(not unbound, 'Kernel provenance refused: module not bound by the released kernel manifest: '
             + ', '.join(unbound[:5]))
    _require(not missing, 'Kernel provenance refused: switched-on kernel module is not a manifest-bound prebuilt: '
             + ', '.join(missing))
    summary = kernel_inventory_summary(inventory)
    return {'required_extensions': required, 'required_modules': required_modules,
            'inventory_sha256': kernel_inventory_sha256(summary), 'inventory': summary}


def verify_kernel_provenance(profiles, qualification, resolve_source, cache_dir):
    """Release/start: the live kernel inventory must pass check() and equal the receipt."""
    record = kernel_provenance_record(profiles, qualification['source_files'], resolve_source, cache_dir)
    _require(record == qualification['kernel_provenance'], 'Kernel provenance differs from qualification')
    return record


def _cache_modules(environment):
    names = set()
    for path in environment['kernel_cache']:
        match = _CACHE_SO.fullmatch(path)
        if match and match.group(1) == match.group(2):
            names.add(match.group(1))
    return names


def _kernel_provenance_valid(value, profiles, qualification):
    """Structure + cross-binding: receipt inventory vs released bytes and the environment fingerprint."""
    _require(type(value) is dict and set(value) == KERNEL_PROVENANCE_KEYS, 'Invalid kernel provenance record')
    _require(value['required_extensions'] == required_kernel_extensions(profiles),
             'Kernel provenance required extensions differ from the profiles')
    _require(value['required_modules'] == required_kernel_modules(profiles),
             'Kernel provenance required modules differ from the profiles')
    inventory = value['inventory']
    _require(type(inventory) is dict and set(inventory) == KERNEL_INVENTORY_KEYS
             and type(inventory['schema']) is int and inventory['schema'] == 1, 'Invalid kernel inventory')
    _require(_hash_valid(value['inventory_sha256']) and value['inventory_sha256'] == kernel_inventory_sha256(inventory),
             'Kernel inventory digest mismatch')
    manifest = qualification['source_files']
    _require(KERNEL_PROVENANCE_MODULE in manifest and _hash_valid(inventory['manifest_sha256'])
             and inventory['manifest_sha256'] == manifest.get(KERNEL_PROVENANCE_MANIFEST),
             'Kernel provenance manifest is not the released one')
    modules, cache = inventory['modules'], qualification['environment']['kernel_cache']
    _require(type(modules) is dict and bool(modules) and set(modules) == _cache_modules(qualification['environment']),
             'Kernel inventory does not cover exactly the prebuilt kernel cache')
    for name, row in modules.items():
        _require(type(row) is dict and set(row) == KERNEL_MODULE_KEYS and row['status'] in KERNEL_MODULE_STATUSES
                 and _hash_valid(row['source_sha256'])
                 and row['so_sha256'] == cache.get(KERNEL_CACHE_JIT_DIR + '/' + name + '/' + name + '.so'),
                 'Kernel module is not a verified prebuilt: ' + str(name)[:80])
    _require(all(name in modules for name in value['required_modules']),
             'Kernel inventory misses a switched-on prebuilt module')
    extensions = inventory['extensions']
    _require(type(extensions) is dict and set(value['required_extensions']) <= set(extensions),
             'Kernel inventory misses required extensions')
    for name, row in extensions.items():
        _require(type(row) is dict and set(row) == KERNEL_EXTENSION_KEYS and type(row['status']) is str
                 and type(row['sources']) is dict, 'Invalid kernel extension row')
        if row['so_name'] is None:
            _require(row['so_sha256'] is None, 'Invalid kernel extension row')
        else:
            _require(type(row['so_name']) is str and '/' not in row['so_name'] and _relative_valid(row['so_name'])
                     and row['so_sha256'] == manifest.get(KERNEL_TREE + row['so_name']),
                     'Kernel extension is not the released one: ' + str(name)[:40])
    for name in value['required_extensions']:
        _require(extensions[name]['status'] == 'prebuilt', 'Required kernel extension is not prebuilt: ' + name)


def verify_checkpoint(model_dir=None):
    """Bind the served weights directory to CHECKPOINT (the receipt only names the constant).

    checkpoint-manifest.json must name CHECKPOINT; download-receipt.json must cover exactly
    the manifest files with the manifest size (and the LFS sha256 where the manifest has
    one); every file must exist as a regular file of that size; files up to
    CHECKPOINT_HASH_LIMIT and every non-LFS file are re-hashed against the receipt.
    """
    base = pathlib.Path(MODEL if model_dir is None else model_dir)
    manifest = read_json(base / CHECKPOINT_MANIFEST)
    _require(type(manifest) is dict and manifest.get('repo') == CHECKPOINT['id']
             and manifest.get('revision') == CHECKPOINT['revision'],
             'Checkpoint manifest differs from the pinned checkpoint')
    files = manifest.get('files')
    _require(type(files) is list and bool(files), 'Checkpoint manifest lists no files')
    expected = {}
    for row in files:
        _require(type(row) is dict and _relative_valid(row.get('path')) and type(row.get('size')) is int
                 and row['size'] >= 0 and row['path'] not in expected, 'Invalid checkpoint manifest entry')
        lfs = row.get('lfs')
        _require(lfs is None or (type(lfs) is dict and _hash_valid(lfs.get('oid'))),
                 'Invalid checkpoint LFS object id')
        expected[row['path']] = (row['size'], None if lfs is None else lfs['oid'])
    downloaded = read_json(base / CHECKPOINT_DOWNLOAD_RECEIPT)
    _require(type(downloaded) is list, 'Invalid checkpoint download receipt')
    resolved = base.resolve()
    seen = set()
    for row in downloaded:
        _require(type(row) is dict and set(row) == CHECKPOINT_RECEIPT_KEYS and type(row['file']) is str
                 and row['file'] in expected
                 and row['file'] not in seen and row['status'] in CHECKPOINT_STATUSES
                 and _hash_valid(row['sha256']) and type(row['bytes']) is int,
                 'Invalid checkpoint download receipt entry')
        name = row['file']
        seen.add(name)
        size, oid = expected[name]
        _require(row['bytes'] == size and (oid is None or row['sha256'] == oid),
                 'Checkpoint download receipt differs from the pinned manifest: ' + name)
        path = base / name
        _require(path.resolve() == resolved.joinpath(name), 'Checkpoint file crosses a symlink: ' + name)
        info = os.lstat(path)
        _require(stat.S_ISREG(info.st_mode) and info.st_size == size, 'Checkpoint file size differs: ' + name)
        if oid is None or size <= CHECKPOINT_HASH_LIMIT:
            _require(sha256_regular_file(path) == row['sha256'], 'Checkpoint file bytes differ: ' + name)
    _require(seen == set(expected), 'Checkpoint download receipt is incomplete')


def load_installed():
    profiles = _strict_json(_read_regular(_production_source('profiles.json')))
    _profiles_schema(profiles)
    qualification = _strict_json(_read_regular(_production_source(QUALIFICATION_RECEIPT)))
    return profiles, qualification


def configure_environment(name, profile, environ=None):
    environ = os.environ if environ is None else environ
    for key in list(environ):
        if key in PRESERVED_ENV:
            continue
        if key.startswith(CLEARED_ENV_PREFIXES) or key in CLEARED_ENV_NAMES:
            environ.pop(key, None)
    environ.update(GDN_BASELINE)
    environ.update(KERNEL_LOADER_ENV)
    environ.update(QWEN36_PROFILE=name, QWEN36_CONTEXT=str(profile['context']),
                   QWEN36_PREFILL_CHUNK=str(profile['chunk']), QWEN36_MEMORY_RATIO=str(profile['memory_ratio']),
                   QWEN36_KV_DTYPE='bf16', QWEN36_STRATEGY=profile_strategy(profile), QWEN36_DATA_ROOT=str(ROOT),
                   QWEN36_STALL_SECONDS='300')
    if 'prefix_margin' in profile:
        environ['QWEN36_PREFIX_MARGIN'] = str(profile['prefix_margin'])
    # Exactly what each switched-on feature needs (SWITCH_ENV), plus the release path of its library.
    for switch in FEATURE_SWITCHES:
        if profile.get(switch) is True:
            environ.update(SWITCH_ENV[switch])
            if switch in NATIVE_LIBRARIES:
                env, path = NATIVE_LIBRARIES[switch]
                environ[env] = str(ROOT / path)
    for field, (env, _, switch) in VALUE_FIELDS.items():
        if field in profile and profile.get(switch) is True:
            environ[env] = str(profile[field])
    return environ


def serve_argv(profile):
    return ['ft', 'serve', '--model', MODEL, '--served-model-name', 'qwen36-35b',
            '--host', '127.0.0.1', '--port', '1919', '--text-model-only', '--cors-origins', '',
            '--max-running-requests', '1', '--max-seq-len-override', str(profile['context']),
            '--max-output-tokens', '8192', '--max-prefill-length', str(profile['chunk']),
            '--memory-ratio', str(profile['memory_ratio']), '--kv-reserve-tokens', str(profile['context']),
            '--moe-cache-auto', '--moe-strategy', profile_strategy(profile), '--moe-cpu-threads', '16', '--expert-load', 'serial',
            '--cache-type', 'radix', '--enable-cache-report'] + list(profile['flags'])


def _selected_name(profiles):
    selected = ROOT / SELECTED_PROFILE
    if not os.path.lexists(selected):
        return profiles['default']
    name = _read_regular(_production_source(SELECTED_PROFILE)).decode('ascii').strip()
    _require(name in CONTEXTS, 'Invalid selected profile')
    return name


def _write_selected(name):
    selected = _production_source(SELECTED_PROFILE)
    temp = selected.with_name(selected.name + '.tmp')
    if os.path.lexists(temp):
        _require(not os.path.islink(temp), 'Refusing a symlinked selection file')
        os.unlink(temp)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(temp, flags, 0o600)
    try:
        os.write(descriptor, (name + '\n').encode('ascii'))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(temp, 0o600)
    os.replace(temp, selected)


def prepare(argv):
    """Validate everything for 'select NAME' or the default start; no side effects."""
    _require(argv == [] or (len(argv) == 2 and argv[0] == 'select'), 'Usage: profile-runtime.py [select NAME]')
    profiles, qualification = load_installed()
    name = argv[1] if argv else _selected_name(profiles)
    profile = validate_profile(name, profiles, qualification, _production_source,
                               expected_sources=_production_sources(profiles),
                               environment=runtime_environment())
    # Before any FreeToken import: prebuilt kernels/extensions provably match their sources,
    # ctypes libraries match the receipt bytes and ABI.
    verify_kernel_provenance(profiles, qualification, _production_source, kernel_cache_dir())
    verify_native_libraries(profiles, qualification, _production_source)
    verify_checkpoint()
    return ('select' if argv else 'start'), name, profile


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        action, name, profile = prepare(argv)
        if action == 'select':
            _write_selected(name)
        else:
            _require(bool(os.environ.get('QWEN36_API_KEY')), 'Protected API key environment is missing')
    except (ValueError, KeyError, TypeError, OSError, UnicodeError) as error:
        detail = str(error) if isinstance(error, ValueError) and not isinstance(error, UnicodeError) else ''
        raise SystemExit('Production profile validation failed: ' + type(error).__name__
                         + (': ' + detail[:200] if detail else '')) from None
    if action == 'select':
        print(json.dumps({'selected_profile': name, 'context': profile['context'], 'kv_dtype': profile['kv_dtype']}))
        return None
    # Only after every check: closed environment, fixed argv, then the server.
    configure_environment(name, profile)
    sys.argv = serve_argv(profile)
    from serve_guarded import main as serve
    return serve()


if __name__ == '__main__':
    main()
