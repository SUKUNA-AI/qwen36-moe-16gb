"""CPU-only release qualification tests on a synthetic staging tree in a temp directory.

release.py paths (ROOT/PROJECT/OUTPUT/runtime source/gate library) are patched to
temporary directories; systemctl is mocked; the real staged profile-runtime.py is
copied byte-for-byte into the synthetic OUTPUT so the real validator is exercised.
Also covers make-qualification.py. Never touches /home/USER/qwen36-api.
"""
import contextlib
import copy
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
SERVICE = HERE.parent
REAL_VALIDATOR = SERVICE / 'profile-runtime.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = load('release_under_qualification_test', HERE / 'release.py')
maker = load('make_qualification_under_test', HERE / 'make-qualification.py')
constants = load('validator_constants_for_fixture', REAL_VALIDATOR)

RUNTIME_FILES = {
    '__init__.py': b'# package\n',
    'engine/engine.py': b'# engine\n',
    'kernel/cuda_attention_gate.py': b'# loader\n',
    'kernel/csrc/qwen36_attention_gate.cu': b'// gate\n',
    'kernel/csrc/qwen36_nvfp4_prefill.cu': b'// rejected kernel source still ships as source\n',
    'moe/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_H100_80GB_HBM3.json': b'{}\n',
}
BUILD_LIBRARIES = ('libqwen_attention_gate.so', 'libqwen_nvfp4_prefill.so', 'libqwen_nvfp4_matched.so',
                   'libqwen36_kv.so', 'libqwen36_complement.so')
PACKAGE_VERSIONS = {'torch': '2.11.0', 'triton': '3.6.0', 'flashinfer-python': '0.6.18.post1',
                    'sglang-kernel': '0.4.5', 'transformers': '5.16.1', 'apache-tvm-ffi': '0.1.13.post3',
                    'freetoken-kernel-cache': '0.1.3+cu130'}
KERNEL_SO = 'jit_cache/freetoken__batch_memcpy/freetoken__batch_memcpy.so'
MODEL_FILES = {'config.json': b'{"model_type": "synthetic"}\n',
               'model-00001-of-00001.safetensors': b'synthetic shard bytes, long'}
# Kernel provenance fixture: real (read-only) FreeToken kernel_provenance.py + synthetic csrc; the
# manifest binds the synthetic native extensions and the synthetic venv kernel cache module.
KP_PATH = release.RUNTIME_SOURCE / 'kernel/kernel_provenance.py'
KP_BYTES = KP_PATH.read_bytes()
KP = constants._load_kernel_provenance(KP_BYTES, KP_PATH)
KERNEL_SOURCES = {'kernel/csrc/pinned_tensor.cpp': b'// pinned tensor\n',
                  'kernel/csrc/cpu_moe/cpu_moe_ext.cpp': b'// cpu moe\n',
                  'kernel/csrc/ple_store/ple_store_ext.cpp': b'// ple store\n',
                  'kernel/csrc/include/freetoken/common.cuh': b'// shared header\n',
                  'kernel/csrc/jit/batch_memcpy.cu': b'// batch memcpy\n',
                  'kernel/kernel_provenance.py': KP_BYTES}
CACHE_MODULES = {'freetoken__batch_memcpy': ('jit/batch_memcpy.cu',)}
NVFP4 = 'release/lib/libqwen_nvfp4_bitexact.so'
BITEXACT_REPORT = {'kind': 'logit-reference-comparison', 'gate': False,
                   'logits': {'compared': 12, 'shape_mismatch': 0, 'only_old': 0, 'only_new': 0, 'bit_identical': 12,
                              'max_abs': 0.0, 'max_relative_l2': 0.0, 'mean_relative_l2': 0.0, 'top1_agreement': 1.0,
                              'all_finite': True}}
STUB_C = b"""
#ifndef BITS
#define BITS 32
#endif
int qwen_nvfp4_global_bits(void) { return BITS; }
int qwen_nvfp4_prefill(const void *args, int bm, int bn, int bk, int fused, void *stream) { return 0; }
const char *qwen_nvfp4_cuda_error(int code) { return "synthetic"; }
"""
# libqwen36_gdn.so / libqwen_fp8_gemv.so as kernel/cuda_gdn.py and kernel/cuda_fp8_gemv.py bind them (no CUDA).
GDN_STUB_C = b"""
#ifndef ABI
#define ABI 2
#endif
int qwen_gdn_fla_chunk_ho(void) { return 0; }
int qwen_gdn_fla_wu_chunk_ho(void) { return 0; }
int qwen_gdn_abi_version(void) { return ABI; }
int qwen_gdn_conv_qkv(void) { return 0; }
int qwen_gdn_gate_cumsum(void) { return 0; }
int qwen_gdn_gated_norm(void) { return 0; }
int qwen_gdn_decode_fused(void) { return 0; }
int qwen_gdn_decode_cluster(void) { return 0; }
int qwen_gdn_silu_eval(void) { return 0; }
int qwen_gdn_silu_points(void) { return 0; }
int qwen_gdn_silu_compare(void) { return 0; }
int qwen_gdn_gate_eval(void) { return 0; }
"""
FP8_STUB_C = b"""
#ifndef ABI
#define ABI 2
#endif
int qwen_fp8_gemv_abi_version(void) { return ABI; }
int qwen_fp8_gemv_max_m(void) { return 8; }
const char *qwen_fp8_gemv_source_sha256(void) { return "9844420833b0c33c5b5fb73280f63527597306adb49b979731ac1fe89ac708fc"; }
int qwen_fp8_gemv_rowwise(void) { return 0; }
int qwen_fp8_gemv_init(void) { return 0; }
const char *qwen_fp8_gemv_error(int code) { return "synthetic"; }
"""
STUB_SOURCES = {'good': (STUB_C, []), 'bits16': (STUB_C, ['-DBITS=16']), 'gdn': (GDN_STUB_C, []),
                'gdn_abi1': (GDN_STUB_C, ['-DABI=1']), 'fp8': (FP8_STUB_C, []), 'fp8_abi1': (FP8_STUB_C, ['-DABI=1'])}
_STUB_DIR = tempfile.TemporaryDirectory(prefix='qwen-nvfp4-stub-')
_STUBS = {}
GDN = 'release/lib/libqwen36_gdn.so'
FP8 = 'release/lib/libqwen_fp8_gemv.so'
GATE = 'release/lib/libqwen_attention_gate.so'
GATHER = 'freetoken__decode_gather_tma_8_3_16384'
GATHER_CSRC = {'kernel/csrc/jit/decode_gather.cuh': b'// decode gather (TMA ring)\n'}
GATHER_MODULES = dict(CACHE_MODULES, **{GATHER: ('jit/decode_gather.cuh',)})
BUNDLE = ('cuda_nvfp4_prefill', 'layer_major_prefill', 'cuda_gdn', 'cuda_fp8_gemv', 'decode_gather')
BUNDLE_EVIDENCE = ('cuda_nvfp4_evidence', 'cuda_gdn_evidence', 'cuda_fp8_gemv_evidence')
# Placeholder switch (background expert refill): profile-runtime.py may drop it alone; make-qualification keeps
# knowing its variable and refuses runs that enable it once the validator does not support it.
PLACEHOLDER = 'layer_major_refill'


def stub(name='good'):
    """Real ELF ctypes library (NVFP4 / GDN / FP8 GEMV entry points, no CUDA); None without a C compiler."""
    if name not in _STUBS:
        compiler = shutil.which('gcc') or shutil.which('cc')
        if compiler is None:
            _STUBS[name] = None
        else:
            data, defines = STUB_SOURCES[name]
            source = pathlib.Path(_STUB_DIR.name) / (name + '.c')
            source.write_bytes(data)
            target = pathlib.Path(_STUB_DIR.name) / (name + '.so')
            subprocess.run([compiler, '-shared', '-fPIC', '-O0', *defines, '-o', str(target), str(source)],
                           check=True, capture_output=True, timeout=120)
            _STUBS[name] = target
    return _STUBS[name]


def kernel_manifest(kernel_dir, cache_dir, cache_modules=None):
    cache_modules = CACHE_MODULES if cache_modules is None else cache_modules
    csrc = pathlib.Path(kernel_dir) / 'csrc'
    upstream = {path.relative_to(csrc).as_posix(): KP.text_sha256(path) for path in sorted(csrc.rglob('*')) if path.is_file()}
    recipes = {name: KP.make_recipe(name, 'synthetic', [csrc / rel for rel in rels], csrc=csrc)
               for name, rels in cache_modules.items()}
    data = KP.generate_manifest(recipes=recipes, cache_dir=pathlib.Path(cache_dir), upstream_csrc=upstream,
                                upstream_ref='synthetic', basis={'kind': 'synthetic-test'}, kernel_dir=pathlib.Path(kernel_dir))
    assert set(data['modules']) == set(cache_modules) and len(data['extensions']) == 3, data['skipped']
    return (json.dumps(data, indent=1, sort_keys=True) + '\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dist_info(base, name):
    return pathlib.Path(base) / (name.replace('-', '_') + '-' + PACKAGE_VERSIONS[name] + '.dist-info')


def make_site_packages(root):
    base = pathlib.Path(root) / constants.VENV_SITE_PACKAGES
    for name, version in PACKAGE_VERSIONS.items():
        info = dist_info(base, name)
        info.mkdir(parents=True, exist_ok=True)
        (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: %s\nVersion: %s\n' % (name, version))
        (info / 'RECORD').write_text('%s/__init__.py,sha256=synthetic,1\n' % name.replace('-', '_'))
    for relative, data in (('__init__.py', b'# cache\n'), (KERNEL_SO, b'\x7fELF batch memcpy')):
        path = base / constants.KERNEL_CACHE_PACKAGE / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return base


def make_model(path):
    path.mkdir(parents=True)
    files, receipt = [], []
    for name, data in MODEL_FILES.items():
        row = {'type': 'file', 'oid': 'f' * 40, 'path': name, 'size': len(data)}
        if name.endswith('.safetensors'):
            row['lfs'] = {'oid': sha(data), 'size': len(data), 'pointerSize': 130}
        files.append(row)
        (path / name).write_bytes(data)
        receipt.append({'file': name, 'bytes': len(data), 'sha256': sha(data), 'status': 'downloaded_verified'})
    (path / constants.CHECKPOINT_MANIFEST).write_text(json.dumps(
        {'repo': constants.CHECKPOINT['id'], 'revision': constants.CHECKPOINT['revision'], 'files': files}))
    (path / constants.CHECKPOINT_DOWNLOAD_RECEIPT).write_text(json.dumps(receipt))
    return path


def profiles_doc(qualified=('main128',), gate=True):
    profiles = {
        'baseline64': {'context': 65536, 'chunk': 2048, 'memory_ratio': 0.75, 'kv_dtype': 'bf16', 'flags': [],
                       'fused_attention_gate': False, 'fp32_gdn_snapshot': False, 'retain_chunk_snapshot': False,
                       'qualified': False},
        'main128': {'context': 131072, 'chunk': 4096, 'memory_ratio': 0.73, 'kv_dtype': 'bf16', 'flags': [],
                    'prefix_margin': 8, 'fused_attention_gate': gate, 'fp32_gdn_snapshot': True,
                    'retain_chunk_snapshot': True, 'qualified': False},
        'long256': {'context': 262144, 'chunk': 4096, 'memory_ratio': 0.72, 'kv_dtype': 'bf16', 'flags': [],
                    'prefix_margin': 8, 'fused_attention_gate': gate, 'fp32_gdn_snapshot': True,
                    'retain_chunk_snapshot': False, 'qualified': False},
    }
    for name in qualified:
        profiles[name]['qualified'] = True
    return {'schema': 1, 'default': 'main128', 'profiles': profiles}


class Staging(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='qwen-release-qual-cpu-')
        base = pathlib.Path(self.tmp.name)
        self.root = base / 'root'
        self.project = base / 'project'
        self.output = self.project / 'outputs/qwen36-api'
        self.runtime_source = self.project / 'work/FreeToken-opt/python/freetoken'
        self.evidence_root = base / 'evidence'
        self.output.mkdir(parents=True)
        self.root.mkdir()
        for name in release.FILES:
            data = REAL_VALIDATOR.read_bytes() if name == 'profile-runtime.py' else ('# staged ' + name + '\n').encode()
            (self.output / name).write_bytes(data)
        self.write_profiles(profiles_doc())
        for relative, data in RUNTIME_FILES.items():
            self.put(self.runtime_source / relative, data)
        self.put(self.runtime_source / '__pycache__/__init__.cpython-312.pyc', b'pyc')
        self.put(self.runtime_source / 'engine/stale.pyc', b'pyc')
        # Qualified native tree = the source tree + compiled FreeToken extensions (native-only).
        self.qualified_runtime = self.root / 'optimization/python/freetoken'
        for relative, data in RUNTIME_FILES.items():
            self.put(self.qualified_runtime / relative, data)
        for relative in release.NATIVE_EXTENSIONS:
            self.put(self.qualified_runtime / relative, b'\x7fELF ' + relative.encode())
        self.put(self.qualified_runtime / 'kernel/__pycache__/pinned.cpython-312.pyc', b'pyc')
        self.site_packages = make_site_packages(self.root)
        self.kernel_files = dict(KERNEL_SOURCES)
        for relative, data in KERNEL_SOURCES.items():
            self.put(self.qualified_runtime / relative, data)
        self.kernel_files['kernel/kernel_cache_manifest.json'] = kernel_manifest(
            self.qualified_runtime / 'kernel', self.site_packages / constants.KERNEL_CACHE_PACKAGE / 'jit_cache')
        for relative, data in self.kernel_files.items():
            for top in (self.runtime_source, self.qualified_runtime):
                self.put(top / relative, data)
        self.nvfp4_source = self.root / 'optimization/private-libraries/libqwen_nvfp4_bitexact.so'
        self.gdn_source = self.root / 'optimization/private-cuda-gdn/pinned/libqwen36_gdn-a8a806eda3a90a65.so'
        self.fp8_source = self.root / 'optimization/private-fp8gemv/libqwen_fp8_gemv-685c78cf89e60d02.so'
        self.model = make_model(base / 'model')
        for library in BUILD_LIBRARIES:
            self.put(self.root / 'optimization/build' / library, ('build ' + library).encode())
        self.gate_source = self.put(self.root / 'optimization/private-libraries/libqwen_attention_gate.so',
                                    b'\x7fELF qualified gate')
        self.put(self.evidence_root / 'reports/main128-quality.json', b'{"status":"PASS"}')
        self.put(self.evidence_root / 'reports/main128-speed.json', b'{"status":"PASS"}')
        self.put(self.evidence_root / 'reports/long256-quality.json', b'{"status":"PASS"}')
        patches = {'ROOT': self.root, 'PROJECT': self.project, 'OUTPUT': self.output,
                   'RUNTIME_SOURCE': self.runtime_source, 'GATE_LIBRARY_SOURCE': self.gate_source,
                   'QUALIFICATION_SOURCE': self.output / 'qualification.json', 'EVIDENCE_ROOT': self.evidence_root,
                   'QUALIFIED_RUNTIME': self.qualified_runtime, 'MODEL_DIR': self.model,
                   'NVFP4_LIBRARY_SOURCE': self.nvfp4_source, 'GDN_LIBRARY_SOURCE': self.gdn_source,
                   'FP8_GEMV_LIBRARY_SOURCE': self.fp8_source}
        self.patchers = [mock.patch.object(release, key, value) for key, value in patches.items()]
        for patcher in self.patchers:
            patcher.start()
        self.stopped = mock.patch.object(release, 'stopped')
        self.stopped_mock = self.stopped.start()

    def tearDown(self):
        self.stopped.stop()
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.tmp.cleanup()

    @staticmethod
    def put(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def write_profiles(self, profiles):
        self.profiles = profiles
        (self.output / 'profiles.json').write_text(json.dumps(profiles, indent=2))

    def decision(self, names=None, statuses=None):
        names = sorted(n for n, p in self.profiles['profiles'].items() if p['qualified'] is True) if names is None else names
        statuses = statuses or {}
        result = {}
        for name in names:
            status, limits = statuses.get(name, ('accepted', []))
            evidence = ['reports/main128-quality.json', 'reports/main128-speed.json'] if name == 'main128' \
                else ['reports/long256-quality.json']
            result[name] = {'acceptance_status': status, 'known_limitations': limits, 'evidence': evidence}
        return {'schema': 1, 'profiles': result}

    def build(self, decision=None):
        validator = release.load_validator()
        sources = release.staged_sources(validator, self.profiles)
        return maker.build_receipt(validator, self.profiles, decision or self.decision(), self.evidence_root, sources,
                                   release.runtime_environment(validator),
                                   kernel_cache_dir=release.kernel_cache_dir(validator))

    def installed_runtime(self):
        installed = load('installed_profile_runtime', self.root / 'profile-runtime.py')
        installed.ROOT = self.root
        installed.MODEL = str(self.model)
        return installed

    def only_manifest(self):
        manifests = sorted((self.root / 'backups').glob('release-*/manifest.json'))
        self.assertEqual(len(manifests), 1)
        return manifests[0]

    def qualify(self, receipt=None):
        receipt = self.build() if receipt is None else receipt
        (self.output / 'qualification.json').write_text(json.dumps(receipt, indent=2, sort_keys=True))
        return receipt

    def install(self, check):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            release.install(check)
        return json.loads(out.getvalue().strip().splitlines()[-1])

    def snapshot(self):
        return {path.relative_to(self.root).as_posix(): path.read_bytes() for path in sorted(self.root.rglob('*'))
                if path.is_file() and not path.is_symlink() and path.relative_to(self.root).parts[0] != 'backups'}

    def refused(self, check=True):
        before = self.snapshot()
        with self.assertRaises(RuntimeError) as raised:
            self.install(check)
        self.stopped_mock.assert_not_called()
        self.assertEqual(self.snapshot(), before)
        return str(raised.exception)


class ReleaseQualificationTests(Staging):
    def test_check_passes_with_receipt_and_ships_only_gate(self):
        self.qualify()
        before = self.snapshot()
        result = self.install(True)
        self.assertEqual(result['check'], 'PASS')
        self.assertEqual(result['qualified_profiles'], ['main128'])
        self.assertEqual(result['native_libraries'], ['release/lib/libqwen_attention_gate.so'])
        self.assertEqual(result['files'], len(release.FILES) + len(RUNTIME_FILES) + len(self.kernel_files)
                         + len(release.NATIVE_EXTENSIONS) + 2)
        self.assertEqual(result['start_profile'], 'main128')
        self.stopped_mock.assert_called_once()
        self.assertEqual(self.snapshot(), before)

    def test_old_truthiness_qualified_refused(self):
        for value in (1, 'true', 'True', [True], 1.0):
            profiles = profiles_doc()
            profiles['profiles']['main128']['qualified'] = value
            self.write_profiles(profiles)
            with self.subTest(value=value):
                self.assertIn('Staged profiles refused', self.refused())

    def test_current_all_false_profiles_refused(self):
        self.write_profiles(profiles_doc(qualified=()))
        self.refused()
        self.write_profiles(profiles_doc(qualified=('long256',)))
        self.refused()

    def test_missing_or_malformed_receipt_refused(self):
        self.assertIn('qualification receipt is missing', self.refused())
        for data in (b'', b'{', b'{"schema":1,"schema":1}', b'{"x": NaN}', b'[]', b'{}'):
            (self.output / 'qualification.json').write_bytes(data)
            with self.subTest(data=data):
                self.refused()

    def test_staged_source_drift_refused(self):
        self.qualify()
        # Runtime changes reach both trees (source edit synced to native) after the receipt.
        for paths, data in (((self.runtime_source / 'engine/engine.py', self.qualified_runtime / 'engine/engine.py'), b'# changed\n'),
                            ((self.qualified_runtime / release.NATIVE_EXTENSIONS[1],), b'\x7fELF rebuilt extension'),
                            ((self.output / 'guard.py',), b'# changed guard\n'),
                            ((self.gate_source,), b'\x7fELF rebuilt gate')):
            originals = [path.read_bytes() for path in paths]
            for path in paths:
                path.write_bytes(data)
            with self.subTest(path=paths[0].name):
                self.assertIn('Source differs from qualification', self.refused())
            for path, original in zip(paths, originals):
                path.write_bytes(original)
        self.install(True)

    def test_new_or_removed_staged_file_refused(self):
        self.qualify()
        extras = [self.put(top / 'engine/new_module.py', b'# new\n') for top in (self.runtime_source, self.qualified_runtime)]
        self.assertIn('coverage', self.refused())
        for extra in extras:
            extra.unlink()
        removed = [top / 'kernel/cuda_attention_gate.py' for top in (self.runtime_source, self.qualified_runtime)]
        data = removed[0].read_bytes()
        for path in removed:
            path.unlink()
        self.refused()
        for path in removed:
            path.write_bytes(data)
        self.install(True)

    def test_configuration_drift_refused(self):
        self.qualify()
        profiles = copy.deepcopy(self.profiles)
        profiles['profiles']['main128']['chunk'] = 2048
        self.write_profiles(profiles)
        self.refused()

    def test_library_allowlist_replaces_glob(self):
        self.qualify()
        validator = release.load_validator()
        sources = release.staged_sources(validator, self.profiles)
        self.assertEqual({k for k in sources if k.startswith('release/lib/')}, {'release/lib/libqwen_attention_gate.so'})
        self.assertEqual(sources['release/lib/libqwen_attention_gate.so'], self.gate_source)
        no_gate = profiles_doc(gate=False)
        self.assertEqual({k for k in release.staged_sources(validator, no_gate) if k.startswith('release/lib/')}, set())
        # A receipt that lists any other library is refused even if the file exists in build/.
        receipt = self.build()
        receipt['source_files']['release/lib/libqwen_nvfp4_prefill.so'] = sha(b'build libqwen_nvfp4_prefill.so')
        self.qualify(receipt)
        self.refused()
        # Gate profile qualified but private library missing: refused, no fallback to build/.
        self.qualify()
        self.gate_source.unlink()
        self.refused()

    def test_strict_resolver_never_reads_installed_files(self):
        self.put(self.root / 'guard.py', b'# installed guard\n')
        self.put(self.root / 'release/python/freetoken/engine/installed_only.py', b'# old install\n')
        validator = release.load_validator()
        sources = release.staged_sources(validator, self.profiles)
        resolve = release.strict_resolver(sources)
        self.assertEqual(resolve('guard.py'), self.output / 'guard.py')
        for relative in ('release/python/freetoken/engine/installed_only.py', 'release/qualification.json',
                         'release/lib/libqwen_nvfp4_prefill.so', 'config/runtime.env'):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                resolve(relative)
        # Even without the coverage cross-check, an unstaged receipt entry is refused by the resolver.
        receipt = self.build()
        receipt['source_files']['release/python/freetoken/engine/installed_only.py'] = sha(b'# old install\n')
        with self.assertRaises(ValueError):
            validator.validate_profiles(self.profiles, receipt, resolve, expected_sources=None)

    def test_resolver_has_no_installed_fallback(self):
        receipt = self.build()
        installed = self.put(self.root / 'release/python/freetoken/engine/installed_only.py', b'# old install\n')
        receipt['source_files']['release/python/freetoken/engine/installed_only.py'] = sha(installed.read_bytes())
        self.qualify(receipt)
        self.refused()

    def test_symlink_in_staging_refused(self):
        self.qualify()
        link = self.runtime_source / 'engine/link.py'
        link.symlink_to(self.output / 'guard.py')
        self.assertIn('Symlink', self.refused())
        link.unlink()
        guard = self.output / 'guard.py'
        target = self.put(pathlib.Path(self.tmp.name) / 'elsewhere/guard.py', guard.read_bytes())
        guard.unlink()
        guard.symlink_to(target)
        self.refused()

    def test_unreadable_staging_directory_refused(self):
        self.qualify()
        hidden = self.runtime_source / 'hidden'
        hidden.mkdir()
        (hidden / 'module.py').write_bytes(b'# would silently vanish from the release')
        hidden.chmod(0)
        try:
            if os.access(hidden, os.R_OK):
                self.skipTest('running with privileges that bypass directory permissions')
            self.assertIn('Staged runtime tree refused', self.refused())
        finally:
            hidden.chmod(0o700)

    def test_validation_precedes_systemctl_and_active_service_blocks(self):
        self.qualify()
        self.stopped_mock.side_effect = RuntimeError('qwen36-inference.service is not verified stopped')
        before = self.snapshot()
        for check in (True, False):
            with self.assertRaises(RuntimeError):
                self.install(check)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.stopped_mock.call_count, 2)

    def test_release_files_and_validator_inventory_agree(self):
        validator = release.load_validator()
        self.assertEqual(set(release.FILES), set(validator.MANAGED_FILES))
        with mock.patch.object(release, 'FILES', release.FILES[:-1]), self.assertRaises(RuntimeError):
            release.load_validator()

    def test_unexpected_installed_python_entry_refused(self):
        self.qualify()
        self.put(self.root / 'release/python/json.py', b'# would shadow stdlib')
        self.assertIn('Unexpected installed entries', self.refused())

    def test_full_install_runtime_gate_and_rollback(self):
        # Previous installed state: an older managed file, a removed runtime module, a stale CUDA library.
        old = {'guard.py': b'# installed old guard\n', 'start-inference.sh': b'#!/bin/sh\nold\n',
               'release/python/freetoken/engine/removed.py': b'# removed upstream\n',
               'release/python/freetoken/engine/engine.py': b'# old engine\n',
               'release/lib/libqwen_nvfp4_prefill.so': b'old rejected kernel',
               'release/qualification.json': b'{"old": true}\n',
               'config/selected-profile': b'main128\n'}
        for relative, data in old.items():
            self.put(self.root / relative, data)
        self.put(self.root / 'release/python/freetoken/engine/__pycache__/engine.cpython-312.pyc', b'pyc')
        before = self.snapshot()
        self.qualify()
        result = self.install(False)
        self.assertEqual(result['stale_files_removed'], 2)
        self.assertFalse((self.root / 'release/lib/libqwen_nvfp4_prefill.so').exists())
        self.assertFalse((self.root / 'release/python/freetoken/engine/removed.py').exists())
        self.assertEqual((self.root / 'release/lib/libqwen_attention_gate.so').read_bytes(), self.gate_source.read_bytes())
        self.assertEqual((self.root / 'release/qualification.json').read_bytes(),
                         (self.output / 'qualification.json').read_bytes())
        self.assertEqual((self.root / 'release/lib/libqwen_attention_gate.so').stat().st_mode & 0o777, 0o600)
        self.assertEqual((self.root / 'start-inference.sh').stat().st_mode & 0o777, 0o700)
        manifest = pathlib.Path(result['rollback_manifest'])
        receipt = json.loads(manifest.read_text())
        self.assertEqual(receipt['state'], 'installed')
        self.assertEqual(receipt['native_libraries'], ['release/lib/libqwen_attention_gate.so'])
        removed = [row for row in receipt['files'] if row['after'] is None]
        self.assertEqual(sorted(row['path'] for row in removed),
                         ['release/lib/libqwen_nvfp4_prefill.so', 'release/python/freetoken/engine/removed.py'])
        # The installed tree passes the installed runtime gate (start path), without serving.
        installed = self.installed_runtime()
        action, name, profile = installed.prepare([])
        self.assertEqual((action, name, profile['chunk']), ('start', 'main128', 4096))
        self.put(self.root / 'release/lib/libqwen36_kv.so', b'late stale library')
        with self.assertRaises(ValueError):
            installed.prepare([])
        (self.root / 'release/lib/libqwen36_kv.so').unlink()
        # Hash-aware rollback restores the exact previous bytes, including removed stale files.
        with contextlib.redirect_stdout(io.StringIO()):
            release.rollback(manifest, True)
            release.rollback(manifest, False)
        after = self.snapshot()
        self.assertEqual({k: v for k, v in after.items() if k in before}, before)
        self.assertEqual(set(after) - set(before), set())
        with contextlib.redirect_stdout(io.StringIO()):
            release.rollback(manifest, False)

    def test_install_reverifies_evidence_hashes(self):
        self.qualify()
        evidence = self.evidence_root / 'reports/main128-speed.json'
        original = evidence.read_bytes()
        evidence.write_bytes(b'{"status":"FAIL"}')
        self.assertIn('evidence changed', self.refused())
        evidence.unlink()
        self.assertIn('evidence unavailable', self.refused())
        evidence.symlink_to(self.evidence_root / 'reports/main128-quality.json')
        self.refused()
        evidence.unlink()
        evidence.write_bytes(original)
        self.assertEqual(self.install(True)['check'], 'PASS')

    def test_source_changed_after_validation_refused_before_changes(self):
        self.qualify()
        self.put(self.root / 'guard.py', b'# installed guard\n')
        before = self.snapshot()
        real_preflight = release.preflight

        def racing_preflight():
            result = real_preflight()
            (self.runtime_source / 'engine/engine.py').write_bytes(b'# edited during the release window\n')
            return result
        with mock.patch.object(release, 'preflight', racing_preflight):
            with self.assertRaises(RuntimeError) as raised:
                self.install(False)
        self.assertIn('changed after validation', str(raised.exception))
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.root / 'backups').exists() and any((self.root / 'backups').iterdir()))

    def test_rollback_rejects_invalid_removal_rows(self):
        self.qualify()
        result = self.install(False)
        manifest = pathlib.Path(result['rollback_manifest'])
        receipt = json.loads(manifest.read_text())
        receipt['files'].append({'path': 'release/lib/ghost.so', 'before': None, 'after': None})
        manifest.write_text(json.dumps(receipt))
        with self.assertRaises(RuntimeError):
            release.validate_rollback(manifest)
        receipt['files'][-1] = {'path': 'release/lib/ghost.so', 'before': '0' * 64, 'after': None}
        manifest.write_text(json.dumps(receipt))
        with self.assertRaises(RuntimeError):
            release.validate_rollback(manifest)

    # --- review findings (fix-release-gate) -------------------------------------------------

    def test_release_ships_qualified_native_extensions(self):
        # The offload path imports freetoken.kernel._pinned_tensor/_cpu_moe from release/python
        # (PYTHONPATH precedes the venv); the source tree has no compiled extensions.
        receipt = self.qualify()
        for relative in release.NATIVE_EXTENSIONS:
            self.assertFalse((self.runtime_source / relative).exists())
            self.assertEqual(receipt['source_files']['release/python/freetoken/' + relative],
                             sha((self.qualified_runtime / relative).read_bytes()))
        self.install(False)
        kernel = self.root / 'release/python/freetoken/kernel'
        for name in ('_cpu_moe', '_pinned_tensor', '_ple_store'):
            spec = importlib.machinery.PathFinder.find_spec(name, [str(kernel)])
            self.assertIsNotNone(spec, name)
            self.assertEqual(pathlib.Path(spec.origin).read_bytes(),
                             (self.qualified_runtime / ('kernel/' + name + '.cpython-312-x86_64-linux-gnu.so')).read_bytes())
        self.assertEqual(self.installed_runtime().prepare([])[1], 'main128')

    def test_staged_runtime_must_equal_qualified_tree(self):
        self.qualify()
        extension = self.qualified_runtime / release.NATIVE_EXTENSIONS[1]
        data = extension.read_bytes()
        extension.unlink()
        self.assertIn('differs from the qualified runtime tree', self.refused())
        with self.assertRaises(RuntimeError):
            self.build()
        extension.write_bytes(data)
        engine = self.runtime_source / 'engine/engine.py'
        original = engine.read_bytes()
        engine.write_bytes(b'# source edit never synced/qualified\n')
        self.assertIn('Staged runtime bytes differ from the qualified runtime tree: engine/engine.py', self.refused())
        engine.write_bytes(original)
        extra = self.put(self.qualified_runtime / 'engine/native_only.py', b'# only in the qualified tree\n')
        self.assertIn('file set differs', self.refused())
        extra.unlink()
        self.assertEqual(self.install(True)['check'], 'PASS')

    def test_runtime_environment_bound_at_release_and_start(self):
        receipt = self.qualify()
        validator = release.load_validator()
        self.assertEqual(receipt['environment'], release.runtime_environment(validator))
        self.assertEqual(receipt['environment']['packages'], PACKAGE_VERSIONS)
        kernel = self.site_packages / constants.KERNEL_CACHE_PACKAGE / KERNEL_SO
        torch_info = dist_info(self.site_packages, 'torch')
        originals = {path: path.read_bytes() for path in (kernel, torch_info / 'METADATA', torch_info / 'RECORD')}
        for label, path, data in (('prebuilt kernel rebuilt', kernel, b'\x7fELF rebuilt same name'),
                                  ('torch upgraded', torch_info / 'METADATA', b'Name: torch\nVersion: 2.12.0\n'),
                                  ('torch reinstalled', torch_info / 'RECORD', b'torch/other.so,sha256=z,1\n')):
            path.write_bytes(data)
            with self.subTest(label):
                self.assertIn('Runtime environment differs', self.refused())
            path.write_bytes(originals[path])
        self.install(False)
        installed = self.installed_runtime()
        installed.prepare([])
        kernel.write_bytes(b'\x7fELF rebuilt after release')
        with self.assertRaises(ValueError) as raised:
            installed.prepare([])
        self.assertIn('Runtime environment differs', str(raised.exception))
        kernel.write_bytes(originals[kernel])
        installed.prepare([])

    def test_checkpoint_verified_at_release(self):
        self.qualify()
        manifest = self.model / constants.CHECKPOINT_MANIFEST
        good = manifest.read_bytes()
        manifest.write_bytes(good.replace(constants.CHECKPOINT['revision'].encode(), b'0' * 40))
        self.assertIn('Checkpoint refused', self.refused())
        self.put(self.evidence_root / 'decision.json', json.dumps(self.decision()).encode())
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            maker.main(['--decision', str(self.evidence_root / 'decision.json'), '--dry-run'], release=release)
        manifest.write_bytes(good)
        config = self.model / 'config.json'
        config.write_bytes(MODEL_FILES['config.json'].replace(b'synthetic', b'SYNTHETIC'))
        self.assertIn('Checkpoint refused', self.refused())
        config.write_bytes(MODEL_FILES['config.json'])
        shard = self.model / 'model-00001-of-00001.safetensors'
        shard.write_bytes(MODEL_FILES[shard.name][:-1])
        self.assertIn('Checkpoint refused', self.refused())
        shard.write_bytes(MODEL_FILES[shard.name])
        self.assertEqual(self.install(True)['check'], 'PASS')

    def test_saved_selection_must_stay_startable(self):
        self.qualify()
        selected = self.root / 'config/selected-profile'
        self.put(selected, b'long256\n')
        message = self.refused()
        self.assertIn('config/selected-profile', message)
        self.assertIn('not qualified in this release', message)
        self.refused(check=False)
        for data in (b'main256\n', b'\xff\xfe\n', b''):
            selected.write_bytes(data)
            with self.subTest(data=data):
                self.refused()
        selected.unlink()
        self.put(self.root / 'config/target', b'main128\n')
        selected.symlink_to(self.root / 'config/target')
        self.assertIn('symlink', self.refused())
        selected.unlink()
        self.assertEqual(self.install(True)['start_profile'], 'main128')
        selected.write_bytes(b'main128\n')
        self.assertEqual(self.install(True)['start_profile'], 'main128')
        # A release qualifying both profiles keeps a long256 selection startable.
        self.write_profiles(profiles_doc(qualified=('main128', 'long256')))
        self.qualify()
        selected.write_bytes(b'long256\n')
        self.assertEqual(self.install(True)['start_profile'], 'long256')
        self.install(False)
        self.assertEqual(self.installed_runtime().prepare([])[1], 'long256')

    def test_post_install_drift_marks_partial_and_rollback_restores(self):
        self.qualify()
        self.put(self.root / 'guard.py', b'# installed old guard\n')
        before = self.snapshot()
        real_atomic = release.atomic

        def lost_write(src, dest):
            if dest.name != 'guard.py':  # simulated lost replace for one target
                real_atomic(src, dest)
        with mock.patch.object(release, 'atomic', lost_write), self.assertRaises(RuntimeError) as raised:
            self.install(False)
        manifest = self.only_manifest()
        self.assertIn('rollback manifest ' + str(manifest), str(raised.exception))
        self.assertIn('guard.py', str(raised.exception))
        self.assertEqual(json.loads(manifest.read_text())['state'], 'partial')
        with contextlib.redirect_stdout(io.StringIO()):
            release.rollback(manifest, False)
        self.assertEqual(self.snapshot(), before)

    def test_backup_copy_corruption_refused_before_changes(self):
        self.qualify()
        self.put(self.root / 'guard.py', b'# installed old guard\n')
        before = self.snapshot()
        real_copy = shutil.copy2

        def corrupt_copy(src, dst, **kwargs):
            real_copy(src, dst, **kwargs)
            pathlib.Path(dst).write_bytes(b'corrupted backup')
        with mock.patch.object(release.shutil, 'copy2', corrupt_copy), self.assertRaises(RuntimeError) as raised:
            self.install(False)
        self.assertIn('Backup copy differs', str(raised.exception))
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(list((self.root / 'backups').glob('release-*/manifest.json')), [])

    def test_intermediate_symlink_refused(self):
        self.qualify()
        real = self.root / 'real-release'
        real.mkdir()
        (self.root / 'release').symlink_to(real, target_is_directory=True)
        for check in (True, False):
            with self.subTest(check=check):
                self.assertIn('below a symlink', self.refused(check))
        self.assertEqual([files for _, _, files in os.walk(real)], [[]])


class MakeQualificationTests(Staging):
    def test_receipt_schema_and_hashes(self):
        receipt = self.build()
        self.assertEqual(set(receipt), {'schema', 'checkpoint', 'profiles', 'source_files', 'environment',
                                        'kernel_provenance', 'native_libraries'})
        self.assertEqual(receipt['schema'], 1)
        record = receipt['profiles']['main128']
        validator = release.load_validator()
        self.assertEqual(record['configuration_sha256'], validator.canonical_profile_sha256(self.profiles['profiles']['main128']))
        self.assertEqual(record['evidence'][0], {'path': 'reports/main128-quality.json', 'sha256': sha(b'{"status":"PASS"}')})
        self.assertNotIn('release/qualification.json', receipt['source_files'])
        self.assertEqual(receipt['source_files']['profile-runtime.py'], sha(REAL_VALIDATOR.read_bytes()))

    def test_acceptance_statuses(self):
        receipt = self.build(self.decision(statuses={'main128': ('accepted_with_known_limits', ['60K total case fails'])}))
        self.assertEqual(receipt['profiles']['main128']['known_limitations'], ['60K total case fails'])
        for status, limits in (('accepted_with_known_limits', []), ('accepted', ['hidden limit']), ('pass', []),
                               ('accepted', [' ']), ('accepted', None)):
            with self.subTest(status=status, limits=limits), self.assertRaises(ValueError):
                self.build(self.decision(statuses={'main128': (status, limits)}))

    def test_decision_must_match_qualified_profiles(self):
        for names in ([], ['main128', 'long256'], ['long256']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.build(self.decision(names=names))
        decision = self.decision()
        decision['extra'] = 1
        with self.assertRaises(ValueError):
            self.build(decision)
        self.write_profiles(profiles_doc(qualified=('long256',)))
        with self.assertRaises(ValueError):
            self.build(self.decision(names=['long256']))

    def test_evidence_paths_strict(self):
        self.put(pathlib.Path(self.tmp.name) / 'outside.json', b'{}')
        (self.evidence_root / 'reports/link.json').symlink_to(self.evidence_root / 'reports/main128-quality.json')
        for evidence in (['reports/missing.json'], ['../outside.json'], ['/etc/hostname'], ['reports/link.json'],
                         ['reports'], [], ['reports/main128-quality.json', 'reports/main128-quality.json'], [7]):
            decision = self.decision()
            decision['profiles']['main128']['evidence'] = evidence
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                self.build(decision)

    def run_receipt(self, **changes):
        validator = release.load_validator()
        sources = release.staged_sources(validator, self.profiles)
        staged = {path[len(validator.RUNTIME_PREFIX):]: sha(pathlib.Path(file).read_bytes())
                  for path, file in sources.items()
                  if path.startswith(validator.RUNTIME_PREFIX) and path.endswith(maker.RUN_SOURCE_SUFFIXES)}
        receipt = {'profile': 'main-retain', 'run_id': 'a' * 32, 'checkpoint': validator.CHECKPOINT['id'],
                   'checkpoint_revision': validator.CHECKPOINT['revision'],
                   'configuration': {'context': 131072, 'chunk': 4096, 'memory_ratio': 0.73, 'qualified': False,
                                     'flags': [], 'env': {'QWEN36_PREFIX_MARGIN': '8', 'QWEN36_FUSED_ATTN_GATE': '1',
                                     'QWEN36_FUSED_ATTN_GATE_LIBRARY': '/private/libqwen_attention_gate.so',
                                     'QWEN36_FP32_GDN_SNAPSHOT': '1', 'QWEN36_RETAIN_CHUNK_SNAPSHOT': '1'}},
                   'source_files': staged,
                   'service_source_files': {name: sha((self.output / name).read_bytes()) for name in maker.RUN_SERVICE_FILES},
                   'native_libraries': {'QWEN36_FUSED_ATTN_GATE_LIBRARY': {'path': '/private/libqwen_attention_gate.so',
                                                                          'sha256': sha(self.gate_source.read_bytes())}},
                   'packages': {name: PACKAGE_VERSIONS[name] for name in maker.RUN_PACKAGES}}
        for key, value in changes.items():
            receipt[key] = value
        return receipt

    def decision_with_run(self, run):
        self.put(self.evidence_root / 'runs/main-retain.json', json.dumps(run).encode())
        decision = self.decision()
        decision['profiles']['main128']['evidence'].append('runs/main-retain.json')
        decision['profiles']['main128']['run_receipts'] = ['runs/main-retain.json']
        return decision

    def test_run_receipt_binding(self):
        self.build(self.decision_with_run(self.run_receipt()))
        teacher = self.run_receipt()
        teacher['configuration']['env'].update(QWEN36_TEACHER_FIXTURE='/x', QWEN36_LOGIT_REFERENCE='/y')
        self.build(self.decision_with_run(teacher))
        bad = []
        run = self.run_receipt()
        run['source_files']['engine/engine.py'] = '0' * 64
        bad.append(('runtime source drift', run))
        run = self.run_receipt()
        run['service_source_files']['guard.py'] = '0' * 64
        bad.append(('service drift', run))
        run = self.run_receipt()
        run['configuration']['chunk'] = 2048
        bad.append(('chunk differs', run))
        run = self.run_receipt()
        del run['configuration']['memory_ratio']
        bad.append(('default ratio .75 differs from .73', run))
        run = self.run_receipt()
        run['configuration']['env']['QWEN36_SPEC'] = '1'
        bad.append(('spec env', run))
        run = self.run_receipt()
        run['configuration']['env']['QWEN36_NSYS_FORWARDS'] = '96'
        bad.append(('profiler env', run))
        run = self.run_receipt()
        del run['configuration']['env']['QWEN36_RETAIN_CHUNK_SNAPSHOT']
        bad.append(('retain off', run))
        run = self.run_receipt()
        run['native_libraries']['QWEN36_FUSED_ATTN_GATE_LIBRARY']['sha256'] = '0' * 64
        bad.append(('gate library differs', run))
        run = self.run_receipt(checkpoint_revision='0' * 40)
        bad.append(('checkpoint', run))
        for label, run in bad:
            with self.subTest(label), self.assertRaises(ValueError):
                self.build(self.decision_with_run(run))
        decision = self.decision_with_run(self.run_receipt())
        decision['profiles']['main128']['evidence'].remove('runs/main-retain.json')
        with self.assertRaises(ValueError):
            self.build(decision)

    def test_run_receipt_packages_bound_to_environment(self):
        bad = []
        for label, packages in (('torch differs', dict(self.run_receipt()['packages'], torch='2.10.0')),
                                ('package missing', {k: v for k, v in self.run_receipt()['packages'].items() if k != 'triton'}),
                                ('extra package', dict(self.run_receipt()['packages'], vllm='1.0'))):
            bad.append((label, self.run_receipt(packages=packages)))
        run = self.run_receipt()
        del run['packages']
        bad.append(('packages absent', run))
        bad.append(('packages not an object', self.run_receipt(packages=['torch'])))
        for label, run in bad:
            with self.subTest(label), self.assertRaises(ValueError) as raised:
                self.build(self.decision_with_run(run))
            self.assertIn('packages', str(raised.exception))
        self.build(self.decision_with_run(self.run_receipt()))

    def test_nongate_run_receipt_with_library_refused(self):
        self.write_profiles(profiles_doc(gate=False))
        run = self.run_receipt()
        run['configuration']['env'] = {'QWEN36_PREFIX_MARGIN': '8', 'QWEN36_FP32_GDN_SNAPSHOT': '1',
                                       'QWEN36_RETAIN_CHUNK_SNAPSHOT': '1'}
        run['native_libraries'] = {}
        self.assertNotIn('release/lib/libqwen_attention_gate.so', self.build(self.decision_with_run(run))['source_files'])
        run['native_libraries'] = {'QWEN36_FUSED_ATTN_GATE_LIBRARY': {'path': '/private/libqwen_attention_gate.so',
                                                                      'sha256': sha(self.gate_source.read_bytes())}}
        with self.assertRaises(ValueError) as raised:
            self.build(self.decision_with_run(run))
        self.assertIn('does not enable', str(raised.exception))

    def test_run_receipt_kernel_inventory_digest(self):
        digest = self.build()['kernel_provenance']['inventory_sha256']
        self.build(self.decision_with_run(self.run_receipt(kernel_inventory_sha256=digest)))
        for value in ('0' * 64, None, 7):
            with self.subTest(value=value), self.assertRaises(ValueError) as raised:
                self.build(self.decision_with_run(self.run_receipt(kernel_inventory_sha256=value)))
            self.assertIn('kernel inventory', str(raised.exception))
        # A run on the native tree computes the same digest (path independent).
        validator = release.load_validator()
        summary = validator.kernel_inventory_for_directory(self.qualified_runtime / 'kernel', release.kernel_cache_dir(validator))
        self.assertEqual(validator.kernel_inventory_sha256(summary), digest)

    # --- bit-exact CUDA NVFP4 prefill switch -------------------------------------------------

    def cuda_setup(self, name='good'):
        if stub() is None:
            self.skipTest('no C compiler for the synthetic ctypes library')
        self.put(self.nvfp4_source, stub(name).read_bytes())
        profiles = profiles_doc()
        profiles['profiles']['main128']['cuda_nvfp4_prefill'] = True
        self.write_profiles(profiles)

    def cuda_run(self, **changes):
        run = self.run_receipt(**changes)
        run['configuration']['env'].update(QWEN36_NVFP4_PREFILL='cuda',
                                           QWEN36_CUDA_NVFP4_LIBRARY='/private/libqwen_nvfp4_bitexact.so')
        run['native_libraries']['QWEN36_CUDA_NVFP4_LIBRARY'] = {'path': '/private/libqwen_nvfp4_bitexact.so',
                                                                'sha256': sha(self.nvfp4_source.read_bytes())}
        return run

    def cuda_decision(self, run=None, report=None, evidence=True):
        self.put(self.evidence_root / 'runs/cuda.json', json.dumps(self.cuda_run() if run is None else run).encode())
        self.put(self.evidence_root / 'reports/cuda-bitexact.json',
                 json.dumps(BITEXACT_REPORT if report is None else report).encode())
        decision = self.decision()
        entry = decision['profiles']['main128']
        entry['evidence'] += ['runs/cuda.json', 'reports/cuda-bitexact.json']
        if evidence:
            entry['cuda_nvfp4_evidence'] = {'run_receipts': ['runs/cuda.json'],
                                            'bitexact_reports': ['reports/cuda-bitexact.json']}
        return decision

    def test_cuda_switch_without_evidence_refused(self):
        self.cuda_setup()
        with self.assertRaises(ValueError) as raised:
            self.build(self.cuda_decision(evidence=False))
        self.assertIn('cuda_nvfp4_evidence', str(raised.exception))
        # Each decision (re)writes the evidence files, so build it right before its own check.
        bad = []
        for label, logits in (('not all identical', dict(BITEXACT_REPORT['logits'], bit_identical=11)),
                              ('max_abs', dict(BITEXACT_REPORT['logits'], max_abs=1e-3)),
                              ('nothing compared', dict(BITEXACT_REPORT['logits'], compared=0, bit_identical=0)),
                              ('missing files', dict(BITEXACT_REPORT['logits'], only_new=1)),
                              ('non-finite', dict(BITEXACT_REPORT['logits'], all_finite=False))):
            bad.append((label, lambda logits=logits: self.cuda_decision(report=dict(BITEXACT_REPORT, logits=logits))))
        bad.append(('wrong report kind', lambda: self.cuda_decision(report=dict(BITEXACT_REPORT, kind='teacher'))))

        def run_with(mutate):
            run = self.cuda_run()
            mutate(run)
            return lambda: self.cuda_decision(run=run)
        bad.append(('run used another library', run_with(
            lambda run: run['native_libraries']['QWEN36_CUDA_NVFP4_LIBRARY'].update(sha256='0' * 64))))
        bad.append(('run used the loader default library', run_with(
            lambda run: (run['configuration']['env'].pop('QWEN36_CUDA_NVFP4_LIBRARY'),
                         run['native_libraries'].pop('QWEN36_CUDA_NVFP4_LIBRARY')))))
        bad.append(('other backend', run_with(lambda run: run['configuration']['env'].update(QWEN36_NVFP4_PREFILL='b12x'))))
        bad.append(('triton run as cuda evidence', lambda: self.cuda_decision(run=self.run_receipt())))

        def decision_with(mutate):
            def make():
                decision = self.cuda_decision()
                mutate(decision['profiles']['main128'])
                return decision
            return make
        bad.append(('report not listed as evidence', decision_with(lambda entry: entry['evidence'].remove('reports/cuda-bitexact.json'))))
        bad.append(('no run receipt', decision_with(lambda entry: entry['cuda_nvfp4_evidence'].update(run_receipts=[]))))
        bad.append(('no report', decision_with(lambda entry: entry['cuda_nvfp4_evidence'].update(bitexact_reports=[]))))
        bad.append(('extra key', decision_with(lambda entry: entry['cuda_nvfp4_evidence'].update(note=1))))
        for label, make in bad:
            with self.subTest(label), self.assertRaises(ValueError):
                self.build(make())
        self.build(self.cuda_decision())
        # A release receipt without the CUDA evidence record is refused by release.py.
        receipt = self.build(self.cuda_decision())
        del receipt['profiles']['main128']['cuda_nvfp4_prefill']
        self.qualify(receipt)
        self.assertIn('bit-exact', self.refused())
        # Evidence for a profile without the switch is refused too.
        self.write_profiles(profiles_doc())
        with self.assertRaises(ValueError):
            self.build(self.cuda_decision())

    def test_cuda_switch_accepts_quality_teacher_report(self):
        # release-prep GAP-1: qualify.py/compare-quality.py is the only producer of candidate logit evidence.
        self.cuda_setup()
        good = {'baseline': 'teacher-bf128', 'candidate': 'teacher-cuda-bitexact128', 'forced_targets_complete': True,
                'logit_comparisons': {'expected': 125, 'observed': 125, 'finite_and_complete': True,
                                      'max_abs': 0.0, 'max_relative_l2': 0.0},
                'new_functional_failures': [], 'ppl_gate_pass': True, 'quality_gate_pass': True,
                'relative_perplexity_growth': 0.0}
        receipt = self.build(self.cuda_decision(report=good))
        self.assertEqual(receipt['profiles']['main128']['cuda_nvfp4_prefill']['bitexact_reports'],
                         ['reports/cuda-bitexact.json'])
        logits = good['logit_comparisons']
        for label, report in (('max_abs', dict(good, logit_comparisons=dict(logits, max_abs=1e-3))),
                              ('relative', dict(good, logit_comparisons=dict(logits, max_relative_l2=1e-9))),
                              ('incomplete', dict(good, logit_comparisons=dict(logits, observed=124))),
                              ('nothing', dict(good, logit_comparisons=dict(logits, expected=0, observed=0))),
                              ('bool count', dict(good, logit_comparisons=dict(logits, expected=True, observed=True))),
                              ('not finite', dict(good, logit_comparisons=dict(logits, finite_and_complete=False))),
                              ('quality gate', dict(good, quality_gate_pass=False)),
                              ('forced targets', dict(good, forced_targets_complete=False)),
                              ('new failure', dict(good, new_functional_failures=['tool-cycle'])),
                              ('other baseline', dict(good, baseline='teacher-fp128')),
                              ('self reference', dict(good, candidate='teacher-bf128')),
                              ('kind present', dict(good, kind='teacher')),
                              ('missing key', {k: v for k, v in good.items() if k != 'quality_gate_pass'})):
            with self.subTest(label), self.assertRaises(ValueError):
                self.build(self.cuda_decision(report=report))

    def test_cuda_switch_with_evidence_releases_and_starts(self):
        self.cuda_setup()
        receipt = self.qualify(self.build(self.cuda_decision()))
        record = receipt['profiles']['main128']['cuda_nvfp4_prefill']
        self.assertEqual(record, {'library': NVFP4, 'library_sha256': sha(self.nvfp4_source.read_bytes()),
                                  'run_receipts': ['runs/cuda.json'], 'bitexact_reports': ['reports/cuda-bitexact.json']})
        self.assertEqual(receipt['native_libraries'][NVFP4],
                         {'env': 'QWEN36_CUDA_NVFP4_LIBRARY', 'sha256': record['library_sha256'], 'abi': constants.NVFP4_ABI})
        result = self.install(True)
        self.assertEqual(result['native_libraries'], ['release/lib/libqwen_attention_gate.so', NVFP4])
        self.assertEqual(result['provenance']['native_library_sha256'][NVFP4], record['library_sha256'])
        self.install(False)
        installed = self.installed_runtime()
        action, name, profile = installed.prepare([])
        environ = installed.configure_environment(name, profile, {})
        self.assertEqual((environ['QWEN36_NVFP4_PREFILL'], environ['QWEN36_CUDA_NVFP4_LIBRARY']),
                         ('cuda', str(self.root / NVFP4)))
        (self.root / NVFP4).write_bytes(stub('bits16').read_bytes())
        with self.assertRaises(ValueError):
            installed.prepare([])

    def test_cuda_library_replaced_or_wrong_abi_refused(self):
        self.cuda_setup()
        receipt = self.qualify(self.build(self.cuda_decision()))
        good = self.nvfp4_source.read_bytes()
        self.nvfp4_source.write_bytes(stub('bits16').read_bytes())
        self.assertIn('Source differs from qualification', self.refused())
        with self.assertRaises(ValueError) as raised:  # re-qualifying the 16-bit library
            self.build(self.cuda_decision())
        self.assertIn('ABI differs', str(raised.exception))
        digest = sha(self.nvfp4_source.read_bytes())
        forged = copy.deepcopy(receipt)
        forged['source_files'][NVFP4] = digest
        forged['native_libraries'][NVFP4]['sha256'] = digest
        forged['profiles']['main128']['cuda_nvfp4_prefill']['library_sha256'] = digest
        self.qualify(forged)
        message = self.refused()
        self.assertIn('Native library refused', message)
        self.assertIn('ABI differs', message)
        self.nvfp4_source.write_bytes(good)
        self.cuda_decision()  # rewrites the evidence files with the original (deterministic) bytes
        self.qualify(receipt)
        self.assertEqual(self.install(True)['check'], 'PASS')

    def test_nvfp4_library_only_with_cuda_profile(self):
        if stub() is None:
            self.skipTest('no C compiler for the synthetic ctypes library')
        self.put(self.nvfp4_source, stub().read_bytes())
        validator = release.load_validator()
        self.assertNotIn(NVFP4, release.staged_sources(validator, self.profiles))
        unqualified = profiles_doc()
        unqualified['profiles']['long256']['cuda_nvfp4_prefill'] = True
        self.assertNotIn(NVFP4, release.staged_sources(validator, unqualified))
        receipt = self.build()
        self.assertNotIn(NVFP4, receipt['source_files'])
        self.assertNotIn(NVFP4, receipt['native_libraries'])
        receipt['source_files'][NVFP4] = sha(self.nvfp4_source.read_bytes())
        self.qualify(receipt)
        self.refused()

    def test_cli_dry_run_and_never_overwrite(self):
        self.put(self.evidence_root / 'decision.json', json.dumps(self.decision()).encode())
        args = ['--decision', str(self.evidence_root / 'decision.json')]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            maker.main(args + ['--dry-run'], release=release)
        self.assertFalse(json.loads(out.getvalue())['written'])
        target = self.output / 'qualification.json'
        self.assertFalse(target.exists())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            maker.main(args + ['--output', str(target)], release=release)
        summary = json.loads(out.getvalue())
        self.assertTrue(summary['written'])
        self.assertEqual(summary['receipt_sha256'], sha(target.read_bytes()))
        self.assertEqual(target.stat().st_mode & 0o777, 0o600)
        first = target.read_bytes()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            maker.main(args + ['--output', str(target)], release=release)
        self.assertEqual(target.read_bytes(), first)
        # The generated receipt is accepted by the release gate.
        self.assertEqual(self.install(True)['check'], 'PASS')
        self.write_profiles(profiles_doc(qualified=()))
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            maker.main(args + ['--dry-run'], release=release)

    def test_zz_no_cuda_or_torch(self):
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), '', 'run with CUDA_VISIBLE_DEVICES= (empty)')
        for name in ('torch', 'freetoken', 'triton'):
            self.assertNotIn(name, sys.modules)


class ReleaseProvenanceTests(Staging):
    """release.py preflight: kernel_provenance check over the staged tree + venv cache (fail-closed)."""

    def edit_both(self, relative, data):
        """A source edit synced to the native tree (both trees stay byte-identical)."""
        originals = []
        for top in (self.runtime_source, self.qualified_runtime):
            path = top / relative
            originals.append((path, path.read_bytes()))
            path.write_bytes(data)
        return originals

    @staticmethod
    def restore(originals):
        for path, data in originals:
            path.write_bytes(data)

    def forge(self, receipt, relatives):
        """A stale/forged receipt: source hashes follow the new bytes, kernel provenance does not."""
        forged = copy.deepcopy(receipt)
        for relative in relatives:
            forged['source_files']['release/python/freetoken/' + relative] = sha((self.qualified_runtime / relative).read_bytes())
        return forged

    def test_inventory_in_receipt_check_and_release_manifest(self):
        receipt = self.qualify()
        record = receipt['kernel_provenance']
        self.assertEqual(record['required_extensions'], ['_pinned_tensor'])
        self.assertEqual(set(record['inventory']['modules']), set(CACHE_MODULES))
        self.assertEqual(record['inventory']['manifest_sha256'],
                         sha(self.kernel_files['kernel/kernel_cache_manifest.json']))
        validator = release.load_validator()
        self.assertEqual(validator.kernel_inventory_for_directory(self.qualified_runtime / 'kernel',
                                                                  release.kernel_cache_dir(validator)), record['inventory'])
        self.assertEqual(receipt['native_libraries'], {'release/lib/libqwen_attention_gate.so': {
            'env': 'QWEN36_FUSED_ATTN_GATE_LIBRARY', 'sha256': sha(self.gate_source.read_bytes()), 'abi': {}}})
        result = self.install(True)
        self.assertEqual(result['provenance']['kernel_inventory_sha256'], record['inventory_sha256'])
        self.assertEqual(result['provenance']['kernel_module_statuses'], ['prebuilt'])
        self.assertEqual(result['provenance']['kernel_extensions'],
                         {'_cpu_moe': 'prebuilt', '_pinned_tensor': 'prebuilt', '_ple_store': 'prebuilt'})
        result = self.install(False)
        manifest = json.loads(pathlib.Path(result['rollback_manifest']).read_text())
        self.assertEqual(manifest['kernel_provenance'], record)
        self.assertEqual(manifest['native_library_bindings'], receipt['native_libraries'])
        self.assertEqual(self.installed_runtime().prepare([])[1], 'main128')

    def test_edited_kernel_source_refused(self):
        receipt = self.qualify()
        for relative in ('kernel/csrc/jit/batch_memcpy.cu', 'kernel/csrc/include/freetoken/common.cuh',
                         'kernel/csrc/pinned_tensor.cpp'):
            originals = self.edit_both(relative, b'// edited, prebuilt .so not rebuilt\n')
            with self.subTest(relative):
                with self.assertRaises(ValueError) as raised:
                    self.build()
                self.assertIn('Kernel provenance refused', str(raised.exception))
                self.qualify(receipt)
                self.assertIn('Source differs', self.refused())
                self.qualify(self.forge(receipt, [relative]))
                message = self.refused()
                self.assertIn('Kernel provenance refused', message)
                self.assertTrue('jit-pending' in message or 'stale-source' in message, message)
            self.restore(originals)
        self.qualify(receipt)
        self.assertEqual(self.install(True)['check'], 'PASS')

    def test_replaced_extension_so_refused(self):
        receipt = self.qualify()
        relative = release.NATIVE_EXTENSIONS[1]
        path = self.qualified_runtime / relative
        original = path.read_bytes()
        path.write_bytes(b'\x7fELF replaced _pinned_tensor')
        with self.assertRaises(ValueError) as raised:
            self.build()
        self.assertIn('so-mismatch', str(raised.exception))
        self.qualify(self.forge(receipt, [relative]))
        self.assertIn('Kernel extension is not the released one', self.refused())
        # Even an inventory edited to the new bytes fails the live check (manifest binds the old .so).
        forged = self.forge(receipt, [relative])
        inventory = forged['kernel_provenance']['inventory']
        inventory['extensions']['_pinned_tensor']['so_sha256'] = sha(path.read_bytes())
        forged['kernel_provenance']['inventory_sha256'] = constants.kernel_inventory_sha256(inventory)
        self.qualify(forged)
        message = self.refused()
        self.assertIn('Kernel provenance refused', message)
        self.assertIn('so-mismatch', message)
        path.write_bytes(original)
        self.qualify(receipt)
        self.assertEqual(self.install(True)['check'], 'PASS')

    def test_cpu_moe_required_only_for_cpu_or_hybrid(self):
        self.edit_both('kernel/csrc/cpu_moe/cpu_moe_ext.cpp', b'// cpu moe edited, _cpu_moe not rebuilt\n')
        receipt = self.qualify()
        self.assertEqual(receipt['kernel_provenance']['inventory']['extensions']['_cpu_moe']['status'], 'stale-source')
        self.assertEqual(self.install(True)['check'], 'PASS')
        self.stopped_mock.reset_mock()
        real = release.load_validator

        def hybrid():
            validator = real()
            validator.profile_strategy = lambda profile: 'hybrid'
            return validator
        with mock.patch.object(release, 'load_validator', hybrid):
            with self.assertRaises(ValueError) as raised:
                self.build()
            self.assertIn('extension _cpu_moe: stale-source', str(raised.exception))
            self.assertIn('required extensions differ', self.refused())


class BundleReleaseTests(Staging):
    """Production bundle: layer-major prefill, CUDA GDN, FP8 GEMV, decode gather (with CUDA NVFP4)."""

    run_receipt = MakeQualificationTests.run_receipt
    decision_with_run = MakeQualificationTests.decision_with_run

    def add_gather(self, manifest=True, so=True):
        """The decode-gather prebuilt: csrc in both trees, .so in the venv cache, manifest entry merged."""
        for relative, data in GATHER_CSRC.items():
            for top in (self.runtime_source, self.qualified_runtime):
                self.put(top / relative, data)
        self.kernel_files.update(GATHER_CSRC)
        cache = self.site_packages / constants.KERNEL_CACHE_PACKAGE / 'jit_cache'
        so_path = self.put(cache / GATHER / (GATHER + '.so'), b'\x7fELF decode gather prebuilt')
        data = kernel_manifest(self.qualified_runtime / 'kernel', cache, GATHER_MODULES if manifest else CACHE_MODULES)
        if not so:
            so_path.unlink()
            so_path.parent.rmdir()
        self.kernel_files['kernel/kernel_cache_manifest.json'] = data
        for top in (self.runtime_source, self.qualified_runtime):
            self.put(top / 'kernel/kernel_cache_manifest.json', data)
        return so_path

    def bundle_setup(self, switches=BUNDLE):
        if stub() is None:
            self.skipTest('no C compiler for the synthetic ctypes libraries')
        for switch, path, name in (('cuda_nvfp4_prefill', self.nvfp4_source, 'good'), ('cuda_gdn', self.gdn_source, 'gdn'),
                                   ('cuda_fp8_gemv', self.fp8_source, 'fp8')):
            if switch in switches:
                self.put(path, stub(name).read_bytes())
        if 'decode_gather' in switches:
            self.add_gather()
        profiles = profiles_doc()
        profiles['profiles']['main128'].update({key: True for key in switches})
        self.write_profiles(profiles)

    def bundle_run(self, switches=BUNDLE, **changes):
        """A run receipt of the bundle profile, as experiment.py records it (main-combined.json style env)."""
        run = self.run_receipt(**changes)
        env, libraries = run['configuration']['env'], run['native_libraries']
        for switch, key, path, source, extra in (
                ('cuda_nvfp4_prefill', 'QWEN36_CUDA_NVFP4_LIBRARY', '/private/libqwen_nvfp4_bitexact.so', self.nvfp4_source,
                 {'QWEN36_NVFP4_PREFILL': 'cuda'}),
                ('cuda_gdn', 'QWEN36_CUDA_GDN_LIBRARY', '/private/libqwen36_gdn.so', self.gdn_source,
                 {'QWEN36_CUDA_GDN': '1', 'QWEN36_CUDA_GDN_DECODE': 'cluster'}),
                ('cuda_fp8_gemv', 'QWEN36_FP8_GEMV_LIBRARY', '/private/libqwen_fp8_gemv.so', self.fp8_source,
                 {'QWEN36_FP8_GEMV': '1'})):
            if switch in switches:
                env.update(extra, **{key: path})
                libraries[key] = {'path': path, 'sha256': sha(source.read_bytes())}
        if 'layer_major_prefill' in switches:
            env['QWEN36_LAYER_MAJOR_PREFILL'] = '1'
        if 'decode_gather' in switches:
            env.update(QWEN36_DECODE_GATHER='tuned', TVM_FFI_CUDA_ARCH_LIST='12.0', FREETOKEN_DISABLE_JIT='1')
        return run

    def bundle_decision(self, run=None, report=None, evidence=BUNDLE_EVIDENCE):
        self.put(self.evidence_root / 'runs/bundle.json', json.dumps(self.bundle_run() if run is None else run).encode())
        self.put(self.evidence_root / 'reports/bundle-bitexact.json',
                 json.dumps(BITEXACT_REPORT if report is None else report).encode())
        decision = self.decision()
        entry = decision['profiles']['main128']
        entry['evidence'] += ['runs/bundle.json', 'reports/bundle-bitexact.json']
        entry['run_receipts'] = ['runs/bundle.json']
        for key in evidence:
            entry[key] = {'run_receipts': ['runs/bundle.json'], 'bitexact_reports': ['reports/bundle-bitexact.json']}
        return decision

    def test_libraries_staged_only_when_switched_on(self):
        if stub() is None:
            self.skipTest('no C compiler for the synthetic ctypes libraries')
        self.put(self.gdn_source, stub('gdn').read_bytes())
        self.put(self.fp8_source, stub('fp8').read_bytes())
        validator = release.load_validator()

        def libraries(profiles):
            return {k: v for k, v in release.staged_sources(validator, profiles).items() if k.startswith('release/lib/')}
        self.assertEqual(libraries(self.profiles), {GATE: self.gate_source})
        on = profiles_doc()
        on['profiles']['main128'].update(cuda_gdn=True, cuda_fp8_gemv=True)
        self.assertEqual(libraries(on), {GATE: self.gate_source, GDN: self.gdn_source, FP8: self.fp8_source})
        unqualified = profiles_doc()
        unqualified['profiles']['long256'].update(cuda_gdn=True, cuda_fp8_gemv=True)
        self.assertEqual(libraries(unqualified), {GATE: self.gate_source})
        # Switched on but the pinned library is missing: refused, never a fallback.
        self.write_profiles(on)
        self.fp8_source.unlink()
        with self.assertRaises(ValueError) as raised:
            self.build()
        self.assertIn('Staged source unavailable or not a regular file: ' + FP8, str(raised.exception))

    def test_bundle_release_and_start(self):
        self.bundle_setup()
        receipt = self.qualify(self.build(self.bundle_decision()))
        record = receipt['profiles']['main128']
        for switch, library, source in (('cuda_nvfp4_prefill', NVFP4, self.nvfp4_source), ('cuda_gdn', GDN, self.gdn_source),
                                        ('cuda_fp8_gemv', FP8, self.fp8_source)):
            self.assertEqual(record[switch], {'library': library, 'library_sha256': sha(source.read_bytes()),
                                              'run_receipts': ['runs/bundle.json'],
                                              'bitexact_reports': ['reports/bundle-bitexact.json']})
        self.assertEqual(receipt['native_libraries'][GDN], {'env': 'QWEN36_CUDA_GDN_LIBRARY',
                                                            'sha256': sha(self.gdn_source.read_bytes()), 'abi': constants.GDN_ABI})
        self.assertEqual(receipt['native_libraries'][FP8], {'env': 'QWEN36_FP8_GEMV_LIBRARY',
                                                            'sha256': sha(self.fp8_source.read_bytes()),
                                                            'abi': constants.FP8_GEMV_ABI})
        self.assertEqual(receipt['kernel_provenance']['required_modules'], [GATHER])
        result = self.install(True)
        self.assertEqual(result['native_libraries'], sorted([GATE, NVFP4, GDN, FP8]))
        self.assertEqual(result['provenance']['required_modules'], [GATHER])
        result = self.install(False)
        manifest = json.loads(pathlib.Path(result['rollback_manifest']).read_text())
        self.assertEqual(manifest['kernel_provenance']['required_modules'], [GATHER])
        self.assertEqual(manifest['native_library_bindings'], receipt['native_libraries'])
        for library, source in ((GDN, self.gdn_source), (FP8, self.fp8_source)):
            self.assertEqual((self.root / library).read_bytes(), source.read_bytes())
            self.assertEqual((self.root / library).stat().st_mode & 0o777, 0o600)
        installed = self.installed_runtime()
        action, name, profile = installed.prepare([])
        environ = installed.configure_environment(name, profile, {'TVM_FFI_CUDA_ARCH_LIST': '8.9'})
        self.assertEqual({key: environ[key] for key in (
            'QWEN36_LAYER_MAJOR_PREFILL', 'QWEN36_CUDA_GDN', 'QWEN36_CUDA_GDN_DECODE', 'QWEN36_CUDA_GDN_LIBRARY',
            'QWEN36_FP8_GEMV', 'QWEN36_FP8_GEMV_LIBRARY', 'QWEN36_DECODE_GATHER', 'FREETOKEN_DISABLE_JIT')}, {
            'QWEN36_LAYER_MAJOR_PREFILL': '1', 'QWEN36_CUDA_GDN': '1', 'QWEN36_CUDA_GDN_DECODE': 'cluster',
            'QWEN36_CUDA_GDN_LIBRARY': str(self.root / GDN), 'QWEN36_FP8_GEMV': '1',
            'QWEN36_FP8_GEMV_LIBRARY': str(self.root / FP8), 'QWEN36_DECODE_GATHER': 'tuned', 'FREETOKEN_DISABLE_JIT': '1'})
        self.assertEqual(environ['TVM_FFI_CUDA_ARCH_LIST'], '12.0')  # pinned (inert under FREETOKEN_DISABLE_JIT=1)
        self.assertNotIn('QWEN36_LAYER_MAJOR_REFILL', environ)

    def test_bundle_run_receipt_semantics(self):
        self.bundle_setup()
        self.build(self.bundle_decision())

        def changed(**changes):
            run = self.bundle_run()
            run['configuration']['env'].update(changes)
            return run

        def dropped(*keys):
            run = self.bundle_run()
            for key in keys:
                run['configuration']['env'].pop(key)
            return run
        bad = [('gdn decode variant cta', changed(QWEN36_CUDA_GDN_DECODE='cta'), 'must be unset or cluster'),
               ('layer-major spelled true', changed(QWEN36_LAYER_MAJOR_PREFILL='true'), 'must be unset or 1'),
               ('decode gather spelled on', changed(QWEN36_DECODE_GATHER='on'), 'must be unset or tuned'),
               ('fp8 gemv 0', changed(QWEN36_FP8_GEMV='0'), 'must be unset or 1'),
               ('gdn without library', dropped('QWEN36_CUDA_GDN_LIBRARY'), 'needs an explicit QWEN36_CUDA_GDN_LIBRARY'),
               ('fp8 without library', dropped('QWEN36_FP8_GEMV_LIBRARY'), 'needs an explicit QWEN36_FP8_GEMV_LIBRARY'),
               ('decode variant without gdn', dropped('QWEN36_CUDA_GDN', 'QWEN36_CUDA_GDN_LIBRARY'),
                'set without QWEN36_CUDA_GDN'),
               ('layer-major off', dropped('QWEN36_LAYER_MAJOR_PREFILL'), 'differs from the profile'),
               ('decode gather off', dropped('QWEN36_DECODE_GATHER', 'TVM_FFI_CUDA_ARCH_LIST'), 'differs from the profile'),
               ('arch pin without decode gather', dropped('QWEN36_DECODE_GATHER'), 'set without QWEN36_DECODE_GATHER'),
               ('refill on, profile off', changed(QWEN36_LAYER_MAJOR_REFILL='1'),
                'differs from the profile' if PLACEHOLDER in constants.SWITCH_ENV else 'does not support: ' + PLACEHOLDER),
               ('refill policy without refill', changed(QWEN36_LAYER_MAJOR_REFILL_POLICY='restore'),
                'set without QWEN36_LAYER_MAJOR_REFILL'),
               ('sync lean on, profile off', changed(QWEN36_SYNC_LEAN='1'),
                'differs from the profile' if 'sync_lean' in constants.SWITCH_ENV else 'does not support: sync_lean'),
               ('gdn fla on, profile off', changed(QWEN36_CUDA_GDN_FLA='wu'),
                'differs from the profile' if 'cuda_gdn_fla' in constants.VALUE_FIELDS else 'does not support'),
               ('gdn fla spelled 1', changed(QWEN36_CUDA_GDN_FLA='1'),
                'not an allowed value' if 'cuda_gdn_fla' in constants.VALUE_FIELDS else 'does not support'),
               ('other arch list', changed(TVM_FFI_CUDA_ARCH_LIST='8.9 12.0'), 'must be unset or 12.0'),
               ('other tvm-ffi knob', changed(TVM_FFI_CACHE_DIR='/tmp/x'), 'TVM_FFI_CACHE_DIR'),
               ('loader pin spelled 0', changed(FREETOKEN_DISABLE_JIT='0'), 'must be unset or 1')]
        run = dropped('QWEN36_CUDA_GDN', 'QWEN36_CUDA_GDN_DECODE', 'QWEN36_CUDA_GDN_LIBRARY')
        run['configuration']['env']['QWEN36_CUDA_GDN_FLA'] = 'ho'
        bad.append(('gdn fla without gdn', run, 'QWEN36_CUDA_GDN_FLA is set without QWEN36_CUDA_GDN'))
        run = self.bundle_run()
        run['native_libraries']['QWEN36_CUDA_GDN_LIBRARY']['sha256'] = '0' * 64
        bad.append(('gdn library bytes differ', run, 'native libraries differ'))
        run = self.bundle_run()
        del run['native_libraries']['QWEN36_FP8_GEMV_LIBRARY']
        bad.append(('fp8 library not recorded', run, 'native libraries differ'))
        for label, run, text in bad:
            with self.subTest(label), self.assertRaises(ValueError) as raised:
                self.build(self.bundle_decision(run=run))
            self.assertIn(text, str(raised.exception))
        # Accepted equivalents: the loader pin and the (inert) arch pin are production values, not required in runs
        # (the parent's main-combined.json carries the arch pin and no loader pin).
        for label, run in (('without loader pin', dropped('FREETOKEN_DISABLE_JIT')),
                           ('without arch pin', dropped('TVM_FFI_CUDA_ARCH_LIST'))):
            with self.subTest(label):
                self.build(self.bundle_decision(run=run))

    def test_refill_policy_value_field(self):
        if PLACEHOLDER not in constants.SWITCH_ENV:
            self.skipTest('placeholder layer_major_refill dropped from profile-runtime.py')
        self.bundle_setup(('layer_major_prefill',))
        profiles = copy.deepcopy(self.profiles)
        profiles['profiles']['main128'].update(layer_major_refill=True, layer_major_refill_policy='restore')
        self.write_profiles(profiles)

        def run_with(**env):
            run = self.bundle_run(('layer_major_prefill',))
            run['configuration']['env'].update(env)
            return run
        receipt = self.build(self.decision_with_run(run_with(QWEN36_LAYER_MAJOR_REFILL='1',
                                                             QWEN36_LAYER_MAJOR_REFILL_POLICY='restore')))
        self.assertEqual(receipt['profiles']['main128']['configuration_sha256'],
                         constants.canonical_profile_sha256(profiles['profiles']['main128']))
        for label, env, text in (('default policy run', {'QWEN36_LAYER_MAJOR_REFILL': '1'}, 'differs from the profile'),
                                 ('other policy', {'QWEN36_LAYER_MAJOR_REFILL': '1',
                                                   'QWEN36_LAYER_MAJOR_REFILL_POLICY': 'last_chunk'}, 'differs from the profile'),
                                 ('unknown policy', {'QWEN36_LAYER_MAJOR_REFILL': '1',
                                                     'QWEN36_LAYER_MAJOR_REFILL_POLICY': 'hot'}, 'not an allowed value')):
            with self.subTest(label), self.assertRaises(ValueError) as raised:
                self.build(self.decision_with_run(run_with(**env)))
            self.assertIn(text, str(raised.exception))
        # Rewrite the evidence run with the qualified bytes (each decision rewrites it), then release and start.
        self.assertEqual(self.build(self.decision_with_run(run_with(QWEN36_LAYER_MAJOR_REFILL='1',
                                                                    QWEN36_LAYER_MAJOR_REFILL_POLICY='restore'))), receipt)
        self.qualify(receipt)
        self.install(False)
        installed = self.installed_runtime()
        action, name, profile = installed.prepare([])
        environ = installed.configure_environment(name, profile, {})
        self.assertEqual((environ['QWEN36_LAYER_MAJOR_REFILL'], environ['QWEN36_LAYER_MAJOR_REFILL_POLICY']), ('1', 'restore'))

    def test_bundle_bit_exact_evidence_required(self):
        self.bundle_setup()
        for missing in BUNDLE_EVIDENCE:
            with self.subTest(missing), self.assertRaises(ValueError) as raised:
                self.build(self.bundle_decision(evidence=tuple(key for key in BUNDLE_EVIDENCE if key != missing)))
            self.assertIn(missing, str(raised.exception))
        with self.assertRaises(ValueError):
            self.build(self.bundle_decision(report=dict(BITEXACT_REPORT, logits=dict(BITEXACT_REPORT['logits'], max_abs=1e-3))))
        receipt = self.build(self.bundle_decision())
        for switch in ('cuda_gdn', 'cuda_fp8_gemv'):
            forged = copy.deepcopy(receipt)
            del forged['profiles']['main128'][switch]
            self.qualify(forged)
            with self.subTest(switch):
                self.assertIn('bit-exact', self.refused())
        # Evidence for a CUDA switch the profile does not enable.
        self.bundle_setup(('cuda_gdn',))
        decision = self.bundle_decision(run=self.bundle_run(('cuda_gdn',)),
                                        evidence=('cuda_gdn_evidence', 'cuda_fp8_gemv_evidence'))
        with self.assertRaises(ValueError) as raised:
            self.build(decision)
        self.assertIn('cuda_fp8_gemv_evidence given for a profile without cuda_fp8_gemv', str(raised.exception))

    def test_bundle_library_wrong_abi_refused(self):
        self.bundle_setup()
        receipt = self.qualify(self.build(self.bundle_decision()))
        for source, bad, library, switch in ((self.gdn_source, 'gdn_abi1', GDN, 'cuda_gdn'),
                                             (self.fp8_source, 'fp8_abi1', FP8, 'cuda_fp8_gemv')):
            good = source.read_bytes()
            source.write_bytes(stub(bad).read_bytes())
            with self.subTest(library):
                self.assertIn('Source differs from qualification', self.refused())
                with self.assertRaises(ValueError) as raised:  # re-qualifying the wrong build
                    self.build(self.bundle_decision())
                self.assertIn('ABI differs', str(raised.exception))
                digest = sha(source.read_bytes())
                forged = copy.deepcopy(receipt)
                forged['source_files'][library] = digest
                forged['native_libraries'][library]['sha256'] = digest
                forged['profiles']['main128'][switch]['library_sha256'] = digest
                self.qualify(forged)
                message = self.refused()
                self.assertIn('Native library refused', message)
                self.assertIn('ABI differs', message)
            source.write_bytes(good)
            self.bundle_decision()  # rewrites the evidence files with the original (deterministic) bytes
            self.qualify(receipt)
        self.assertEqual(self.install(True)['check'], 'PASS')

    def test_decode_gather_needs_manifest_bound_prebuilt(self):
        profiles = profiles_doc()
        profiles['profiles']['main128']['decode_gather'] = True
        self.write_profiles(profiles)
        self.add_gather(manifest=False, so=False)  # tree has the source, the cache has no prebuilt
        with self.assertRaises(ValueError) as raised:
            self.build()
        self.assertIn('switched-on kernel module is not a manifest-bound prebuilt: ' + GATHER, str(raised.exception))
        # A receipt made while the switch was off, relabelled for the switch-on profile: release refuses.
        self.write_profiles(profiles_doc())
        stale = self.build()
        self.write_profiles(profiles)
        stale['profiles']['main128']['configuration_sha256'] = constants.canonical_profile_sha256(profiles['profiles']['main128'])
        self.qualify(stale)
        self.assertIn('required modules differ', self.refused())
        # With the prebuilt in the venv cache and bound by the manifest: receipt and release pass.
        so = self.add_gather()
        receipt = self.qualify(self.build())
        self.assertEqual(receipt['kernel_provenance']['required_modules'], [GATHER])
        self.assertEqual(self.install(True)['provenance']['required_modules'], [GATHER])
        self.stopped_mock.reset_mock()
        so.write_bytes(b'\x7fELF decode gather rebuilt under the same name')
        self.assertIn('Runtime environment differs', self.refused())

    def test_cache_shipping_gather_needs_manifest_entry_even_when_off(self):
        # aot.default_kernel_specs() lists the module: a rebuilt release cache ships its .so regardless.
        self.add_gather(manifest=False)
        with self.assertRaises(ValueError) as raised:
            self.build()
        self.assertIn(GATHER + ': unknown', str(raised.exception))
        self.add_gather()
        receipt = self.qualify(self.build())
        self.assertEqual(receipt['kernel_provenance']['required_modules'], [])
        self.assertEqual(receipt['kernel_provenance']['inventory']['modules'][GATHER]['status'], 'prebuilt')
        self.assertEqual(self.install(True)['check'], 'PASS')

    def test_placeholder_refill_switch_can_be_dropped(self):
        real = release.load_validator

        def without_refill():
            validator = real()
            validator.SWITCH_ENV = {k: v for k, v in validator.SWITCH_ENV.items() if k != 'layer_major_refill'}
            validator.FEATURE_SWITCHES = tuple(validator.SWITCH_ENV)
            validator.SWITCH_REQUIRES = {k: v for k, v in validator.SWITCH_REQUIRES.items() if k in validator.SWITCH_ENV}
            validator.VALUE_FIELDS = {k: v for k, v in validator.VALUE_FIELDS.items() if v[2] in validator.SWITCH_ENV}
            validator.OPTIONAL_FIELDS = frozenset({'nvfp4_prefill', 'prefix_margin', *validator.FEATURE_SWITCHES,
                                                   *validator.VALUE_FIELDS})
            return validator
        with mock.patch.object(release, 'load_validator', without_refill):
            receipt = self.qualify(self.build(self.decision_with_run(self.run_receipt())))
            self.assertEqual(self.install(True)['check'], 'PASS')
            run = self.run_receipt()
            run['configuration']['env']['QWEN36_LAYER_MAJOR_REFILL'] = '1'
            with self.assertRaises(ValueError) as raised:
                self.build(self.decision_with_run(run))
            self.assertIn('does not support: layer_major_refill', str(raised.exception))
            profiles = copy.deepcopy(self.profiles)
            profiles['profiles']['main128']['layer_major_refill'] = False
            self.write_profiles(profiles)
            self.stopped_mock.reset_mock()
            self.assertIn('Staged profiles refused', self.refused())
        self.assertNotIn('layer_major_refill', receipt['profiles']['main128'])


class BundleWiringTests(unittest.TestCase):
    """make-qualification's run/decision tables must follow the staged validator's switch tables."""

    def test_make_qualification_tables_follow_the_validator(self):
        optional = {PLACEHOLDER, 'sync_lean'}
        self.assertEqual(set(constants.FEATURE_SWITCHES) - set(maker.RUN_SWITCH_ENV), set())
        self.assertTrue(set(maker.RUN_SWITCH_ENV) - set(constants.FEATURE_SWITCHES) <= optional)
        for switch in constants.FEATURE_SWITCHES:
            key, value = maker.RUN_SWITCH_ENV[switch]
            self.assertEqual(constants.SWITCH_ENV[switch].get(key), value, switch)
        for field, (env, _, switch) in constants.VALUE_FIELDS.items():
            self.assertEqual(maker.RUN_VALUE_ENV[field], env)
            self.assertEqual(maker.RUN_COMPANION_ENV[env], maker.RUN_SWITCH_ENV[switch][0])
        exported = ({key for exports in constants.SWITCH_ENV.values() for key in exports}
                    | {env for env, _ in constants.NATIVE_LIBRARIES.values()} | {'QWEN36_PREFIX_MARGIN', 'QWEN36_KV_DTYPE'}
                    | {spec[0] for spec in constants.VALUE_FIELDS.values()})
        self.assertEqual(exported - maker.RUN_FEATURE_ENV, set())
        pairs = {item for exports in constants.SWITCH_ENV.values() for item in exports.items()}
        dropped = {maker.RUN_SWITCH_ENV[switch][0] for switch in optional if switch not in constants.SWITCH_ENV}
        for item in maker.RUN_EXACT_ENV.items():
            if item[0] not in dropped:
                self.assertIn(item, pairs)
        library_env = {env for env, _ in constants.NATIVE_LIBRARIES.values()}
        for key, library_key in maker.RUN_LIBRARY_ENV.items():
            self.assertIn(key, maker.RUN_EXACT_ENV)
            self.assertIn(library_key, library_env)
        self.assertEqual(set(maker.CUDA_DECISION_EVIDENCE), set(constants.CUDA_EVIDENCE_SWITCHES))
        self.assertTrue(set(maker.CUDA_DECISION_EVIDENCE.values()) <= maker.DECISION_OPTIONAL_KEYS)
        self.assertEqual(constants.KERNEL_LOADER_ENV, {'FREETOKEN_DISABLE_JIT': '1'})
        self.assertEqual(set(constants.KERNEL_LOADER_ENV) & maker.RUN_FEATURE_ENV, set())


if __name__ == '__main__':
    unittest.main(verbosity=2)
