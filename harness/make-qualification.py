"""Build a schema-1 release qualification receipt from an explicit owner decision.

Offline and CPU-only: never starts a model, service or release. The decision file
names the accepted profiles; the tool hashes the staged sources (the exact
``release.py`` mapping) and the referenced evidence files, optionally cross-checks
experiment run receipts against the staged bytes/configuration, then validates the
result with the staged ``profile-runtime.py`` before writing it (never overwrites).

Decision file (schema 1)::

    {"schema": 1,
     "profiles": {"main128": {"acceptance_status": "accepted" | "accepted_with_known_limits",
                              "known_limitations": ["..."],
                              "evidence": ["reports/relative/path.json", ...],
                              "run_receipts": ["relative/path/listed/in/evidence.json"],
                              "cuda_nvfp4_evidence": {"run_receipts": ["runs/x.json"],
                                                      "bitexact_reports": ["reports/y.json"]},
                              "cuda_gdn_evidence": {...same shape...},
                              "cuda_fp8_gemv_evidence": {...same shape...}}}}

Evidence paths are relative to ``release.EVIDENCE_ROOT`` (/home/USER/qwen36-api/optimization);
release.py re-verifies these hashes at install. ``run_receipts`` is optional; each entry
must also be listed in ``evidence``.
The receipt also records the venv runtime environment (``profile-runtime.runtime_environment``:
package versions/RECORD hashes and the prebuilt kernel cache bytes); release and start
refuse when it changes, and run receipts must carry the same package versions. The
checkpoint directory must match the pinned checkpoint before a receipt is written.
The set of decided profiles must equal the set with ``qualified: true`` in the
staged profiles document. This tool does not decide acceptance.

Kernel provenance: the receipt records ``kernel_provenance`` (profile-runtime
``kernel_provenance_record``: kernel_provenance inventory()/check() over the staged kernel bytes and
the venv kernel cache, fail-closed) and ``native_libraries`` (ctypes library env/sha256/ABI probe).
A run receipt that carries ``kernel_inventory_sha256`` must match the receipt inventory digest.
``cuda_nvfp4_evidence`` is required exactly for profiles with ``cuda_nvfp4_prefill: true``: run
receipts (bound to the staged bit-exact library and the CUDA configuration) and logit-reference
comparison reports showing bit-identical logits (compare-logit-reference.py, or the compare-quality.py
report of a qualify.py teacher run: logit_comparisons expected == observed > 0, max_abs == max_relative_l2 == 0,
quality_gate_pass, baseline teacher-bf128); all must also be listed in ``evidence``. The same holds for
``cuda_gdn_evidence`` (``cuda_gdn: true``) and ``cuda_fp8_gemv_evidence`` (``cuda_fp8_gemv: true``): the
staged validator's CUDA_EVIDENCE_SWITCHES decide which switches need it.
Run receipts map their env onto the staged validator's switches (RUN_SWITCH_ENV); for the switches of the
production bundle (layer-major prefill, CUDA GDN, FP8 GEMV, decode gather, layer-major refill) the run
must carry exactly the value production exports, and a run that enables a switch the staged validator
does not know is refused; value fields (validator VALUE_FIELDS, e.g. the refill policy) must carry an
allowed value. Production never JIT-compiles kernel modules (validator KERNEL_LOADER_ENV,
FREETOKEN_DISABLE_JIT=1, accepted in runs with that value); a run that sets any TVM_FFI_* knob other than the
decode_gather pin TVM_FFI_CUDA_ARCH_LIST=12.0 is refused.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
DECISION_KEYS = frozenset({'schema', 'profiles'})
DECISION_PROFILE_KEYS = frozenset({'acceptance_status', 'known_limitations', 'evidence'})
# CUDA library switch -> decision key of its bit-exact evidence (receipt record key: the switch name).
CUDA_DECISION_EVIDENCE = {'cuda_nvfp4_prefill': 'cuda_nvfp4_evidence', 'cuda_gdn': 'cuda_gdn_evidence',
                          'cuda_fp8_gemv': 'cuda_fp8_gemv_evidence', 'cuda_gdn_fla':'cuda_gdn_fla_evidence'}
DECISION_OPTIONAL_KEYS = frozenset({'run_receipts', *CUDA_DECISION_EVIDENCE.values()})
CUDA_DECISION_KEYS = frozenset({'run_receipts', 'bitexact_reports'})
# Same representation as optimization/experiment.py and bench.py source receipts.
RUN_SOURCE_SUFFIXES = ('.py', '.cu', '.h', '.cuh', '.cpp')
RUN_SERVICE_FILES = ('serve_guarded.py', 'guard.py', 'ingress.py', 'request_metrics.py')
# experiment.py records exactly these versions as run receipt 'packages'.
RUN_PACKAGES = ('torch', 'triton', 'flashinfer-python', 'sglang-kernel', 'transformers')
RUN_FEATURE_ENV = frozenset({'QWEN36_PREFIX_MARGIN', 'QWEN36_FUSED_ATTN_GATE', 'QWEN36_FUSED_ATTN_GATE_LIBRARY',
                             'QWEN36_FP32_GDN_SNAPSHOT', 'QWEN36_RETAIN_CHUNK_SNAPSHOT', 'QWEN36_KV_DTYPE',
                             'QWEN36_NVFP4_PREFILL', 'QWEN36_CUDA_NVFP4_LIBRARY',
                             'QWEN36_LAYER_MAJOR_PREFILL', 'QWEN36_CUDA_GDN', 'QWEN36_CUDA_GDN_LIBRARY',
                             'QWEN36_CUDA_GDN_DECODE', 'QWEN36_FP8_GEMV', 'QWEN36_FP8_GEMV_LIBRARY',
                             'QWEN36_DECODE_GATHER', 'TVM_FFI_CUDA_ARCH_LIST', 'QWEN36_LAYER_MAJOR_REFILL',
                             'QWEN36_LAYER_MAJOR_REFILL_POLICY', 'QWEN36_SYNC_LEAN', 'QWEN36_LAYER_MAJOR_ATTN_BATCH',
                             'QWEN36_CUDA_GDN_FLA', 'QWEN36_LAYER_MAJOR_MIN_TOKENS', 'QWEN36_DECODE_SHARED_OVERLAP'})
# Profile switch -> (run environment variable, value that switches it on).
RUN_SWITCH_ENV = {'fused_attention_gate': ('QWEN36_FUSED_ATTN_GATE', '1'),
                  'fp32_gdn_snapshot': ('QWEN36_FP32_GDN_SNAPSHOT', '1'),
                  'retain_chunk_snapshot': ('QWEN36_RETAIN_CHUNK_SNAPSHOT', '1'),
                  'cuda_nvfp4_prefill': ('QWEN36_NVFP4_PREFILL', 'cuda'),
                  'layer_major_prefill': ('QWEN36_LAYER_MAJOR_PREFILL', '1'),
                  'cuda_gdn': ('QWEN36_CUDA_GDN', '1'),
                  'cuda_fp8_gemv': ('QWEN36_FP8_GEMV', '1'),
                  'decode_gather': ('QWEN36_DECODE_GATHER', 'tuned'),
                  'layer_major_refill': ('QWEN36_LAYER_MAJOR_REFILL', '1'),
                  'decode_shared_overlap': ('QWEN36_DECODE_SHARED_OVERLAP', '1'),
                  'sync_lean': ('QWEN36_SYNC_LEAN', '1')}
# Profile value field -> run environment variable (validator VALUE_FIELDS give the allowed values).
RUN_VALUE_ENV = {'cuda_gdn_fla': 'QWEN36_CUDA_GDN_FLA', 'layer_major_refill_policy': 'QWEN36_LAYER_MAJOR_REFILL_POLICY',
                 'layer_major_attn_batch': 'QWEN36_LAYER_MAJOR_ATTN_BATCH',
                 'layer_major_min_tokens': 'QWEN36_LAYER_MAJOR_MIN_TOKENS'}
# Production bundle switches: the run must carry exactly what configure_environment exports, or nothing.
# Another spelling FreeToken might also accept ('true', 'on', 'cta', ...) is not the qualified configuration.
RUN_EXACT_ENV = {'QWEN36_DECODE_SHARED_OVERLAP': '1', 'QWEN36_LAYER_MAJOR_PREFILL': '1', 'QWEN36_CUDA_GDN': '1', 'QWEN36_CUDA_GDN_DECODE': 'cluster',
                 'QWEN36_FP8_GEMV': '1', 'QWEN36_DECODE_GATHER': 'tuned', 'TVM_FFI_CUDA_ARCH_LIST': '12.0',
                 'QWEN36_LAYER_MAJOR_REFILL': '1', 'QWEN36_SYNC_LEAN': '1'}
# Variables production exports only together with a switch (variable -> switch variable).
RUN_COMPANION_ENV = {'QWEN36_CUDA_GDN_DECODE': 'QWEN36_CUDA_GDN', 'TVM_FFI_CUDA_ARCH_LIST': 'QWEN36_DECODE_GATHER',
                     'QWEN36_LAYER_MAJOR_REFILL_POLICY': 'QWEN36_LAYER_MAJOR_REFILL',
                     'QWEN36_LAYER_MAJOR_MIN_TOKENS': 'QWEN36_LAYER_MAJOR_PREFILL',
                     'QWEN36_LAYER_MAJOR_ATTN_BATCH': 'QWEN36_LAYER_MAJOR_PREFILL', 'QWEN36_CUDA_GDN_FLA': 'QWEN36_CUDA_GDN'}
# Library switches whose run must name its library explicitly (switch variable -> library variable).
RUN_LIBRARY_ENV = {'QWEN36_CUDA_GDN': 'QWEN36_CUDA_GDN_LIBRARY', 'QWEN36_FP8_GEMV': 'QWEN36_FP8_GEMV_LIBRARY'}
# Teacher fixtures only record logits for the quality comparison; anything else differs.
RUN_DIAGNOSTIC_ENV = frozenset({'QWEN36_TEACHER_FIXTURE', 'QWEN36_LOGIT_REFERENCE', 'QWEN36_TEACHER_REFERENCE_REPORT'})
EXPERIMENT_DEFAULT_MEMORY_RATIO = 0.75


class QualificationError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise QualificationError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence_file(evidence_root, relative, validator):
    """Resolve a decision evidence path strictly below evidence_root, no symlinks anywhere."""
    _require(validator._relative_valid(relative), 'Invalid evidence path: must be relative POSIX without ./..')
    base = pathlib.Path(evidence_root).resolve()
    path = base.joinpath(relative)
    _require(path.resolve() == path, 'Evidence path crosses a symlink or escapes the evidence root')
    try:
        return path, validator.sha256_regular_file(path)
    except FileNotFoundError:
        raise QualificationError('Evidence file is missing: ' + relative) from None
    except (OSError, ValueError):
        raise QualificationError('Evidence file is not a regular non-symlink file: ' + relative) from None


def run_equivalent_profile(configuration, switches=None, pins=None, value_fields=None):
    """Map an experiment configuration to the production profile semantics.

    ``switches``: the staged validator's FEATURE_SWITCHES (default: every switch known here). A switch the
    validator does not know must be off in the run and is left out of the result. ``pins``: the staged
    validator's KERNEL_LOADER_ENV; a run may carry those variables, only with the production value.
    ``value_fields``: the staged validator's VALUE_FIELDS (allowed values; unknown fields must be unset)."""
    _require(type(configuration) is dict and {'context', 'chunk', 'flags', 'env'} <= set(configuration),
             'Run receipt configuration is incomplete')
    _require(set(configuration) <= {'context', 'chunk', 'memory_ratio', 'flags', 'env', 'qualified'},
             'Run receipt configuration has unsupported fields')
    env = configuration['env']
    _require(type(env) is dict, 'Run receipt env is invalid')
    pins = dict(pins or {})
    jit = sorted(str(key)[:40] for key in env if str(key).startswith('TVM_FFI_') and key not in RUN_FEATURE_ENV)
    _require(not jit, 'Run receipt sets ' + ', '.join(jit) + ': tvm-ffi build knobs outside the production profile; '
             'production loads only manifest-bound prebuilts (FREETOKEN_DISABLE_JIT=1)')
    _require(set(env) <= RUN_FEATURE_ENV | RUN_DIAGNOSTIC_ENV | set(pins),
             'Run receipt env contains switches outside the production profile')
    for key, value in pins.items():
        _require(env.get(key, value) == value, 'Run receipt ' + key + ' must be unset or ' + value)
    ratio = configuration.get('memory_ratio', EXPERIMENT_DEFAULT_MEMORY_RATIO)
    # Only the bit-exact CUDA path with an explicit library (the loader default is another .so).
    _require(env.get('QWEN36_NVFP4_PREFILL', 'cuda') == 'cuda'
             and (env.get('QWEN36_NVFP4_PREFILL') != 'cuda' or bool(env.get('QWEN36_CUDA_NVFP4_LIBRARY'))),
             'Run receipt NVFP4 prefill must be unset or cuda with an explicit library')
    for key, value in RUN_EXACT_ENV.items():
        _require(env.get(key, value) == value, 'Run receipt ' + key + ' must be unset or ' + value)
    for key, switch_key in RUN_COMPANION_ENV.items():
        _require(key not in env or env.get(switch_key) == RUN_EXACT_ENV[switch_key],
                 'Run receipt ' + key + ' is set without ' + switch_key)
    for switch_key, library_key in RUN_LIBRARY_ENV.items():
        _require(switch_key not in env or bool(env.get(library_key)),
                 'Run receipt ' + switch_key + ' needs an explicit ' + library_key)
    result = {'context': configuration['context'], 'chunk': configuration['chunk'],
              'memory_ratio': float(ratio) if type(ratio) in (int, float) else ratio,
              'kv_dtype': env.get('QWEN36_KV_DTYPE', 'bf16'), 'flags': configuration['flags'],
              'prefix_margin': int(env.get('QWEN36_PREFIX_MARGIN', '1')),
              **{switch: env.get(key) == value for switch, (key, value) in RUN_SWITCH_ENV.items()}}
    for field, key in RUN_VALUE_ENV.items():
        raw = env.get(key)
        if value_fields is None:
            result[field] = raw
            continue
        if field not in value_fields:
            _require(raw is None, 'Run receipt sets ' + key + ', which the staged validator does not support')
            continue
        allowed = value_fields[field][1]
        value = int(raw) if raw is not None and type(allowed[0]) is int and raw.isdigit() else raw
        _require(raw is None or value in allowed, 'Run receipt ' + key + ' is not an allowed value')
        result[field] = value
    if switches is not None:
        known = set(switches)
        unknown = sorted(switch for switch in RUN_SWITCH_ENV if switch not in known and result[switch])
        _require(not unknown, 'Run receipt enables switches the staged validator does not support: '
                 + ', '.join(unknown))
        result = {key: value for key, value in result.items() if key not in RUN_SWITCH_ENV or key in known}
    return result


def production_semantics(profile, validator):
    return {'context': profile['context'], 'chunk': profile['chunk'], 'memory_ratio': profile['memory_ratio'],
            'kv_dtype': profile['kv_dtype'], 'flags': profile['flags'],
            'prefix_margin': profile.get('prefix_margin', 1),
            **{key: profile.get(key, False) for key in validator.FEATURE_SWITCHES},
            **{key: profile.get(key) for key in validator.VALUE_FIELDS}}


QUALITY_REPORT_KEYS = frozenset({'baseline', 'candidate', 'logit_comparisons', 'forced_targets_complete',
                                 'quality_gate_pass', 'new_functional_failures'})


def quality_evidence_module():
    source = HERE/'python/freetoken/quality_evidence.py'
    if not source.is_file():
        fork = pathlib.Path(os.environ.get('QWEN36_FREETOKEN_SOURCE', HERE.parent/'freetoken'))
        source = fork/'python/freetoken/quality_evidence.py'
    return load_module('quality_evidence_checks', source)


def check_quality_bitexact_report(report):
    """compare-quality.py output of a qualify.py teacher run: every in-engine logit comparison against the
    teacher-bf128 logit reference (engine/qualification.py, QWEN36_LOGIT_REFERENCE) must be exactly zero."""
    helpers = quality_evidence_module()
    TEACHER_CASES, unique_rows, hash_valid = helpers.TEACHER_CASES, helpers.unique_rows, helpers.hash_valid
    rows = unique_rows(report.get('rows'), TEACHER_CASES, 'quality report')
    _require(all(type(r.get('tokens')) is int and r['tokens'] > 0 for r in rows.values()), 'quality token counts missing')
    case_runs = report.get('case_runs')
    _require(type(case_runs) is dict and set(case_runs) == {'baseline','candidate'} and
             all(type(v) is dict and set(v) == set(TEACHER_CASES) and all(type(run) is str and run for run in v.values())
                 for v in case_runs.values()), 'quality case->run bindings incomplete')
    hashes = report.get('reports_sha256')
    _require(type(hashes) is dict and set(hashes) == {'baseline','candidate','functional_baseline','functional_candidate'}
             and all(hash_valid(v) for v in hashes.values()) and hash_valid(report.get('fixture_sha256')),
             'quality source/fixture identities missing')
    _require(type(report.get('tokens')) is int and report['tokens'] == sum(r['tokens'] for r in rows.values()),
             'quality total token count mismatch')
    logits = report['logit_comparisons']
    _require(type(logits) is dict and report['baseline'] == 'teacher-bf128'
             and type(report['candidate']) is str and report['candidate'].startswith('teacher-')
             and report['candidate'] != 'teacher-bf128'
             and type(logits.get('expected')) is int and logits['expected'] == sum((r['tokens']+7)//8 for r in rows.values())
             and type(logits.get('observed')) is int and logits['observed'] == logits['expected']
             and logits.get('finite_and_complete') is True and logits.get('all_byte_equal') is True
             and all(type(logits.get(key)) in (int, float) and logits.get(key) == 0
                     for key in ('max_abs', 'max_relative_l2'))
             and report.get('schema') == 2 and report.get('evidence_valid') is True
             and report.get('bit_exact_pass') is True
             and report['forced_targets_complete'] is True and report['quality_gate_pass'] is True
             and report['new_functional_failures'] == [],
             'Teacher-forcing quality report does not show identical logits vs teacher-bf128')


def check_fla_component_report(report, variant, library_sha256):
    rows = report.get('results') if type(report) is dict else None
    _require(type(report) is dict and report.get('all_pass') is True and report.get('library_sha256') == library_sha256
             and report.get('fla_fused_candidate') is True and report.get('fla_variant') == variant
             and type(report.get('fla_fused_calls')) is int and report['fla_fused_calls'] > 0
             and type(rows) is list and rows and type(report.get('cases')) is int and report['cases'] == len(rows)
             and all(type(r) is dict and r.get('pass') is True for r in rows),
             'FLA component evidence incomplete, unbound, wrong variant or non-exact')


def check_bitexact_report(report):
    """compare-logit-reference.py output must show every compared logit file bit-identical; a compare-quality.py
    teacher report (no ``kind``) is accepted when every logit comparison is exactly zero."""
    if type(report) is dict and 'kind' not in report and QUALITY_REPORT_KEYS <= set(report):
        check_quality_bitexact_report(report)
        return
    logits = report.get('logits') if type(report) is dict else None
    _require(type(report) is dict and report.get('kind') == 'logit-reference-comparison' and type(logits) is dict,
             'Bit-exact evidence must be a logit-reference comparison report')
    compared, max_abs = logits.get('compared'), logits.get('max_abs')
    _require(type(compared) is int and compared > 0 and logits.get('bit_identical') == compared
             and type(logits.get('bit_identical')) is int
             and all(type(logits.get(key)) is int and logits.get(key) == 0 for key in ('shape_mismatch', 'only_old', 'only_new'))
             and type(max_abs) in (int, float) and max_abs == 0 and logits.get('all_finite') is True,
             'Bit-exact evidence does not show identical logits')


def check_run_receipt(receipt, profile, source_hashes, validator, environment, kernel_provenance=None):
    """Bind an experiment run receipt to the staged bytes, the venv packages and the profile semantics.

    Returns True when the run receipt also carried the kernel inventory digest (and it matched)."""
    _require(type(receipt) is dict, 'Run receipt must be an object')
    _require(receipt.get('checkpoint') == validator.CHECKPOINT['id']
             and receipt.get('checkpoint_revision') == validator.CHECKPOINT['revision'],
             'Run receipt checkpoint mismatch')
    packages = receipt.get('packages')
    _require(type(packages) is dict and set(packages) == set(RUN_PACKAGES)
             and all(packages[name] == environment['packages'][name] for name in RUN_PACKAGES),
             'Run receipt packages differ from the runtime environment')
    configuration = receipt.get('configuration')
    # The loader pins (FREETOKEN_DISABLE_JIT=1) are accepted with their production value and recommended for
    # decode_gather runs (the run then proves the module loaded as a manifest-bound prebuilt); not required.
    _require(run_equivalent_profile(configuration, validator.FEATURE_SWITCHES, validator.KERNEL_LOADER_ENV,
                                    validator.VALUE_FIELDS)
             == production_semantics(profile, validator), 'Run receipt configuration differs from the profile')
    staged_runtime = {path[len(validator.RUNTIME_PREFIX):]: digest for path, digest in source_hashes.items()
                      if path.startswith(validator.RUNTIME_PREFIX) and path.endswith(RUN_SOURCE_SUFFIXES)}
    _require(receipt.get('source_files') == staged_runtime, 'Run receipt runtime sources differ from staged bytes')
    service = receipt.get('service_source_files')
    _require(type(service) is dict and all(service.get(name) == source_hashes[name] for name in RUN_SERVICE_FILES),
             'Run receipt service sources differ from staged bytes')
    libraries = receipt.get('native_libraries')
    _require(type(libraries) is dict, 'Run receipt native libraries are invalid')
    expected = {env: path for switch, (env, path) in validator.NATIVE_LIBRARIES.items() if profile.get(switch) is True}
    if expected:
        _require(set(libraries) == set(expected)
                 and all(type(libraries[env]) is dict and libraries[env].get('sha256') == source_hashes.get(path)
                         for env, path in expected.items()),
                 'Run receipt native libraries differ from the staged libraries')
    else:
        _require(not libraries, 'Run receipt used a native library the profile does not enable')
    if 'kernel_inventory_sha256' not in receipt:
        return False
    _require(kernel_provenance is not None and receipt['kernel_inventory_sha256'] == kernel_provenance['inventory_sha256'],
             'Run receipt kernel inventory differs from the staged kernel provenance')
    return True


def build_receipt(validator, profiles, decision, evidence_root, sources, environment, *, kernel_cache_dir):
    """Return a validated schema-1 receipt. sources: {production relative path: staged file};
    environment: validator.runtime_environment() of the service venv; kernel_cache_dir: its jit_cache."""
    validator.validate_profiles_document(profiles)
    _require(validator._environment_valid(environment), 'Invalid runtime environment fingerprint')
    _require(set(validator.CUDA_EVIDENCE_SWITCHES) <= set(CUDA_DECISION_EVIDENCE),
             'The staged validator requires bit-exact evidence for a switch this tool has no decision key for')
    qualified = {name for name, profile in profiles['profiles'].items() if profile['qualified'] is True}
    _require(profiles['profiles'][profiles['default']]['qualified'] is True,
             'Default profile is not marked qualified in the staged profiles document')
    _require(type(decision) is dict and set(decision) == DECISION_KEYS and type(decision['schema']) is int
             and decision['schema'] == 1, 'Invalid decision document')
    decided = decision['profiles']
    _require(type(decided) is dict and set(decided) == qualified,
             'Decision must list exactly the profiles marked qualified')
    _require(validator.QUALIFICATION_RECEIPT not in sources, 'Receipt cannot hash itself')
    source_hashes = {}
    for relative in sorted(sources):
        try:
            source_hashes[relative] = validator.sha256_regular_file(pathlib.Path(sources[relative]))
        except (OSError, ValueError):
            raise QualificationError('Staged source unavailable or not a regular file: ' + relative) from None

    def resolve(relative):
        if relative not in sources:
            raise ValueError('Source is not staged')
        return sources[relative]

    # Fail-closed before any record: kernel provenance of the staged bytes, ctypes library ABI.
    kernel = validator.kernel_provenance_record(profiles, source_hashes, resolve, kernel_cache_dir)
    native = validator.native_library_record(profiles, source_hashes, resolve)
    records = {}
    for name in sorted(decided):
        entry = decided[name]
        _require(type(entry) is dict and DECISION_PROFILE_KEYS <= set(entry)
                 and set(entry) <= DECISION_PROFILE_KEYS | DECISION_OPTIONAL_KEYS, 'Invalid decision entry: ' + name)
        status, limits = entry['acceptance_status'], entry['known_limitations']
        _require(status in validator.ACCEPTANCE_STATUSES, 'Unsupported acceptance status: ' + name)
        _require(type(limits) is list and all(type(item) is str and item.strip() for item in limits),
                 'Known limitations must be non-blank strings: ' + name)
        _require((status == 'accepted_with_known_limits') == bool(limits),
                 'accepted_with_known_limits requires limitations; accepted forbids them: ' + name)
        evidence = entry['evidence']
        _require(type(evidence) is list and evidence and len(set(map(str, evidence))) == len(evidence),
                 'Evidence must be a non-empty list without duplicates: ' + name)
        rows, paths = [], {}
        for relative in evidence:
            path, digest = evidence_file(evidence_root, relative, validator)
            rows.append({'path': relative, 'sha256': digest})
            paths[relative] = path
        profile = profiles['profiles'][name]
        run_receipts = entry.get('run_receipts', [])
        _require(type(run_receipts) is list, 'run_receipts must be a list: ' + name)
        for relative in run_receipts:
            _require(relative in paths, 'Run receipt must also be listed as evidence: ' + name)
            check_run_receipt(read_evidence(validator, paths, relative), profile, source_hashes, validator,
                              environment, kernel)
        records[name] = {'configuration_sha256': validator.canonical_profile_sha256(profile),
                         'acceptance_status': status, 'known_limitations': list(limits), 'evidence': rows}
        for switch, decision_key in sorted(CUDA_DECISION_EVIDENCE.items()):
            cuda = entry.get(decision_key)
            if switch not in validator.CUDA_EVIDENCE_SWITCHES or not validator.cuda_evidence_enabled(profile, switch):
                _require(cuda is None, decision_key + ' given for a profile without ' + switch + ': ' + name)
                continue
            decision_keys = CUDA_DECISION_KEYS | ({'component_reports'} if switch == 'cuda_gdn_fla' else set())
            _require(type(cuda) is dict and set(cuda) == decision_keys,
                     switch + ' requires ' + decision_key + ' (run_receipts, bitexact_reports): ' + name)
            for key in sorted(decision_keys):
                items = cuda[key]
                _require(type(items) is list and items and len(set(map(str, items))) == len(items)
                         and all(item in paths for item in items),
                         switch + ' bit-exact evidence must be non-empty and listed in evidence: ' + key)
            for relative in cuda['run_receipts']:
                check_run_receipt(read_evidence(validator, paths, relative), profile, source_hashes, validator,
                                  environment, kernel)
            for relative in cuda['bitexact_reports']:
                check_bitexact_report(read_evidence(validator, paths, relative))
            library = validator.CUDA_EVIDENCE_LIBRARIES[switch]
            if switch == 'cuda_gdn_fla':
                for relative in cuda['component_reports']:
                    check_fla_component_report(read_evidence(validator, paths, relative), profile['cuda_gdn_fla'], source_hashes[library])
                runs = {read_evidence(validator, paths, rel)['run_id'] for rel in cuda['run_receipts']}
                for relative in cuda['bitexact_reports']:
                    quality = read_evidence(validator, paths, relative)
                    _require(quality.get('schema') == 2 and set(quality.get('case_runs',{}).get('candidate',{}).values()) == runs,
                             'FLA teacher report is not bound to the listed candidate runs')
            records[name][switch] = {
                'library': library, 'library_sha256': source_hashes[library],
                'run_receipts': list(cuda['run_receipts']), 'bitexact_reports': list(cuda['bitexact_reports'])}
            if switch == 'cuda_gdn_fla':
                records[name][switch].update(variant=profile['cuda_gdn_fla'], component_reports=list(cuda['component_reports']))
    receipt = {'schema': 1, 'checkpoint': dict(validator.CHECKPOINT), 'profiles': records,
               'source_files': source_hashes, 'environment': environment,
               'kernel_provenance': kernel, 'native_libraries': native}
    validator.validate_profiles(profiles, receipt, resolve, expected_sources=set(sources), environment=environment)
    return receipt


def read_evidence(validator, paths, relative):
    try:
        return validator.read_json(paths[relative])
    except (OSError, ValueError):
        raise QualificationError('Evidence is not strict JSON: ' + relative) from None


def write_new(path, data):
    path = pathlib.Path(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv=None, release=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--decision', required=True, help='Owner decision JSON (schema 1)')
    parser.add_argument('--output', help='New receipt path; refused if it exists')
    parser.add_argument('--dry-run', action='store_true', help='Validate and print a summary; write nothing')
    args = parser.parse_args(argv)
    if not args.dry_run and not args.output:
        parser.error('--output is required unless --dry-run')
    if release is None:  # tests inject a release module whose paths point at a synthetic staging tree
        release = load_module('qwen36_release_for_qualification', HERE / 'release.py')
    try:
        validator = release.load_validator()
        profiles = validator.read_json(release.OUTPUT / 'profiles.json')
        validator.validate_profiles_document(profiles)
        decision = validator.read_json(pathlib.Path(args.decision))
        sources = release.staged_sources(validator, profiles)
        release.verify_checkpoint(validator)
        receipt = build_receipt(validator, profiles, decision, release.EVIDENCE_ROOT, sources,
                                release.runtime_environment(validator),
                                kernel_cache_dir=release.kernel_cache_dir(validator))
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit('Qualification refused: ' + type(error).__name__ + ': ' + str(error)[:300]) from None
    data = (json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + '\n').encode()
    summary = {'profiles': sorted(receipt['profiles']), 'source_files': len(receipt['source_files']),
               'kernel_inventory_sha256': receipt['kernel_provenance']['inventory_sha256'],
               'kernel_required_modules': receipt['kernel_provenance']['required_modules'],
               'native_libraries': sorted(receipt['native_libraries']),
               'receipt_sha256': hashlib.sha256(data).hexdigest(), 'written': False}
    if not args.dry_run:
        try:
            write_new(args.output, data)
        except FileExistsError:
            raise SystemExit('Qualification refused: output exists; receipts are never overwritten') from None
        summary.update(written=True, output=str(args.output))
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    sys.exit(main())
