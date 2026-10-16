"""CPU-only: the main128 next candidate through the whole release chain, on synthetic data in temp directories.

A: make-qualification -> release.py install -> installed profile-runtime.py for the exact candidate profile
   (serving/profiles.json main128 + harness/main-release-next-d5.json run env): six CUDA libraries with
   ABI stubs, FLA, T3 evidence, C0/C1 and hit-D2D kernel modules. Reuses the synthetic staging fixture of
   test-release-qualification-cpu.py (real validator, mocked systemctl).
B: release-main128.py (decision from named reports, f1 carry/rerun/drop, export-tree).
C: release-main128.sh in a sandbox (fake systemctl, own python): --dry-run writes nothing, --check, --apply,
   label reuse, units running, --rollback. POSIX + C compiler only.
Never touches /home/USER/qwen36-api or a service.
"""
import argparse
import contextlib
import copy
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
from repo_paths import SERVING, FREETOKEN


def load(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load('release_qualification_fixture_for_main128', HERE / 'test-release-qualification-cpu.py')
helper = load('release_main128_helper_under_test', HERE / 'release-main128.py')
release, maker, constants, sha = base.release, base.maker, base.constants, base.sha
CANDIDATE = json.loads((HERE / 'main-release-next-d5.json').read_text())
PROFILES = json.loads((SERVING / 'profiles.json').read_text())
MAIN128 = dict(PROFILES['profiles']['main128'], qualified=True)
GATHER, MULTI, BATCH = base.GATHER, 'freetoken__fast_index_copy_multi_1024_64', 'freetoken__batch_memcpy'
NEXT_CSRC = {'kernel/csrc/jit/decode_gather.cuh': b'// decode gather (TMA ring)\n',
             'kernel/csrc/jit/fast_index_copy.cuh': b'// multi-bank index copy\n'}
NEXT_MODULES = dict(base.CACHE_MODULES, **{GATHER: ('jit/decode_gather.cuh',), MULTI: ('jit/fast_index_copy.cuh',)})
OPROJ, MOE = constants.OPROJ_GEMV_LIBRARY, constants.MOE_DECODE_LIBRARY
base.STUB_SOURCES.update({
    'oproj': (b"""
#ifndef ABI
#define ABI 1
#endif
int qwen_oproj_tensorwise(void) { return 0; }
int qwen_oproj_init(void) { return 0; }
const char *qwen_oproj_error(int code) { return "synthetic"; }
int qwen_oproj_abi_version(void) { return ABI; }
int qwen_oproj_max_m(void) { return 8; }
int qwen_oproj_splits(void) { return 3; }
const char *qwen_oproj_source_sha256(void) { return "%s"; }
""" % constants.OPROJ_GEMV_ABI['qwen_oproj_source_sha256'].encode(), []),
    'moe': (b"""
int qwen_moe_decode_gemv(void) { return 0; }
const char *qwen_moe_decode_error(int code) { return "synthetic"; }
int qwen_moe_decode_abi_version(void) { return 1; }
const char *qwen_moe_decode_source_sha256(void) { return "%s"; }
""" % constants.MOE_DECODE_ABI['qwen_moe_decode_source_sha256'].encode(), [])})
base.STUB_SOURCES['oproj_abi2'] = (base.STUB_SOURCES['oproj'][0], ['-DABI=2'])
RUN_ID = 'c' * 32


def teacher_report(run_ids):
    cases = maker.quality_evidence_module().TEACHER_CASES
    runs = list(run_ids)
    return {'schema': 2, 'evidence_valid': True, 'bit_exact_pass': True, 'baseline': 'teacher-bf128',
            'candidate': 'teacher-main-release-next-d5', 'forced_targets_complete': True,
            'logit_comparisons': {'expected': 125, 'observed': 125, 'finite_and_complete': True, 'all_byte_equal': True,
                                  'max_abs': 0.0, 'max_relative_l2': 0.0},
            'new_functional_failures': [], 'ppl_gate_pass': True, 'quality_gate_pass': True,
            'relative_perplexity_growth': 0.0,
            'rows': [{'case': case, 'tokens': 48 if i < 19 else 88} for i, case in enumerate(cases)], 'tokens': 1000,
            'fixture_sha256': 'f' * 64, 'reports_sha256': {k: 'a' * 64 for k in
                                                          ('baseline', 'candidate', 'functional_baseline', 'functional_candidate')},
            'case_runs': {'baseline': {c: 'baseline-synthetic' for c in cases},
                          'candidate': {c: runs[i % len(runs)] for i, c in enumerate(cases)}}}


def overlap_component(cases=None, **changes):
    rows = [{'case': c, 'pass': True, 'outcome': 'passed'} for c in sorted(cases or maker.OVERLAP_COMPONENT_CASES)]
    report = {'schema': 1, 'kind': 'gpu-component-report', 'suite': maker.OVERLAP_COMPONENT_SUITE, 'suite_sha256': 'c' * 64,
              'cases': len(rows), 'skipped': 0, 'all_pass': True, 'results': rows}
    report.update(changes)
    return report


def fla_component(gdn_sha256):
    return {'all_pass': True, 'library_sha256': gdn_sha256, 'fla_fused_candidate': True, 'fla_variant': 'wu',
            'fla_fused_calls': 40, 'cases': 1, 'results': [{'case': 'wu-chunk', 'pass': True}]}


def decision_args(evidence_root, **changes):
    values = {'run': ['runs/next.json'], 'teacher': ['reports/next-teacher.json'], 'fla_component': ['reports/next-fla.json'],
              'overlap_component': ['reports/next-overlap.json'], 'ab': ['reports/next-ab.json'], 'evidence': [],
              'limitation': [], 'limitations_file': None, 'long256': 'drop', 'baseline64': 'drop', 'long256_run': [],
              'long256_teacher': [], 'long256_evidence': [], 'long256_bitexact': [], 'long256_limitation': [],
              'long256_limitations_file': None, 'evidence_root': str(evidence_root)}
    values.update(changes)
    return argparse.Namespace(**values)


class NextFixture(base.Staging):
    """Staging with the six pinned libraries (ABI stubs), the hit-D2D/C1 modules and the candidate evidence."""

    run_receipt = base.MakeQualificationTests.run_receipt

    def setUp(self):
        if base.stub() is None:
            self.skipTest('no C compiler for the synthetic ctypes libraries')
        super().setUp()
        self.oproj_source = self.root / 'optimization/private-decode-kernels/libqwen_oproj_gemv-fa9302e31cb88a02.so'
        self.moe_source = self.root / 'optimization/private-decode-kernels/libqwen_moe_decode-83e90c43cb63ce11.so'
        for patcher in (mock.patch.object(release, 'OPROJ_GEMV_LIBRARY_SOURCE', self.oproj_source),
                        mock.patch.object(release, 'MOE_DECODE_LIBRARY_SOURCE', self.moe_source)):
            patcher.start()
            self.patchers.append(patcher)
        for path, name in ((self.nvfp4_source, 'good'), (self.gdn_source, 'gdn'), (self.fp8_source, 'fp8'),
                           (self.oproj_source, 'oproj'), (self.moe_source, 'moe')):
            self.put(path, base.stub(name).read_bytes())
        self.add_modules()
        self.write_profiles({'schema': 1, 'default': 'main128',
                             'profiles': dict(base.profiles_doc(qualified=())['profiles'], main128=dict(MAIN128, qualified=False))})

    def add_modules(self, without=()):
        """Sources in both trees; prebuilt .so and manifest entry for every NEXT module except ``without``."""
        for relative, data in NEXT_CSRC.items():
            for top in (self.runtime_source, self.qualified_runtime):
                self.put(top / relative, data)
        self.kernel_files.update(NEXT_CSRC)
        cache = self.site_packages / constants.KERNEL_CACHE_PACKAGE / 'jit_cache'
        modules = {name: rels for name, rels in NEXT_MODULES.items() if name not in without}
        for name in without:
            shutil.rmtree(cache / name, ignore_errors=True)
        for name in modules:
            if name != BATCH:
                self.put(cache / name / (name + '.so'), b'\x7fELF prebuilt ' + name.encode())
        data = base.kernel_manifest(self.qualified_runtime / 'kernel', cache, modules)
        for top in (self.runtime_source, self.qualified_runtime):
            self.put(top / 'kernel/kernel_cache_manifest.json', data)

    def library_source(self, key):
        return {'QWEN36_FUSED_ATTN_GATE_LIBRARY': self.gate_source, 'QWEN36_CUDA_NVFP4_LIBRARY': self.nvfp4_source,
                'QWEN36_CUDA_GDN_LIBRARY': self.gdn_source, 'QWEN36_FP8_GEMV_LIBRARY': self.fp8_source,
                'QWEN36_OPROJ_GEMV_LIBRARY': self.oproj_source, 'QWEN36_MOE_DECODE_LIBRARY': self.moe_source}[key]

    def next_run(self, run_id=RUN_ID, **env):
        run = self.run_receipt(run_id=run_id, profile='main-release-next-d5')
        configuration = {key: copy.deepcopy(CANDIDATE[key]) for key in ('context', 'chunk', 'memory_ratio', 'qualified', 'flags')}
        configuration['env'] = dict(CANDIDATE['env'], QWEN36_TEACHER_FIXTURE='/x', QWEN36_LOGIT_REFERENCE='/y', **env)
        run['configuration'] = configuration
        run['native_libraries'] = {key: {'path': value, 'sha256': sha(self.library_source(key).read_bytes())}
                                   for key, value in configuration['env'].items() if key.endswith('_LIBRARY')}
        return run

    def evidence(self, run=None, teacher=None, overlap=None, ab=None):
        files = {'runs/next.json': self.next_run() if run is None else run,
                 'reports/next-teacher.json': teacher_report([RUN_ID]) if teacher is None else teacher,
                 'reports/next-fla.json': fla_component(sha(self.gdn_source.read_bytes())),
                 'reports/next-overlap.json': overlap_component() if overlap is None else overlap,
                 'reports/next-ab.json': {'verdict': 'PASS', 'performance_gate_pass': True} if ab is None else ab}
        for relative, value in files.items():
            self.put(self.evidence_root / relative, json.dumps(value).encode())

    def compose(self, **changes):
        self.evidence(**{k: changes.pop(k) for k in ('run', 'teacher', 'overlap', 'ab') if k in changes})
        args = decision_args(self.evidence_root, evidence=['reports/main128-quality.json'], **changes)
        staged, decision = helper.compose(release.load_validator(), maker, self.profiles, {}, args)
        self.write_profiles(staged)
        return decision

class Main128NextChain(NextFixture):
    """A: the exact candidate qualifies only with its evidence, installs, and exports the candidate environment."""

    def test_candidate_qualifies_installs_and_runs_the_candidate_environment(self):
        decision = self.compose()
        entry = decision['profiles']['main128']
        self.assertEqual(sorted(k for k in entry if k.endswith('_evidence')), sorted(
            ['cuda_nvfp4_evidence', 'cuda_gdn_evidence', 'cuda_fp8_gemv_evidence', 'cuda_gdn_fla_evidence',
             'cuda_oproj_gemv_evidence', 'cuda_moe_decode_evidence', 'decode_shared_overlap_evidence']))
        receipt = self.qualify(self.build(decision))
        record = receipt['profiles']['main128']
        self.assertEqual(record['decode_shared_overlap'], {
            'component_reports': ['reports/next-overlap.json'], 'run_receipts': ['runs/next.json'],
            'bitexact_reports': ['reports/next-teacher.json'], 'ab_reports': ['reports/next-ab.json']})
        self.assertEqual(record['cuda_gdn_fla']['variant'], 'wu')
        for library, source, abi in ((OPROJ, self.oproj_source, constants.OPROJ_GEMV_ABI),
                                     (MOE, self.moe_source, constants.MOE_DECODE_ABI)):
            self.assertEqual(receipt['native_libraries'][library]['abi'], abi)
            self.assertEqual(receipt['native_libraries'][library]['sha256'], sha(source.read_bytes()))
        self.assertEqual(receipt['kernel_provenance']['required_modules'], sorted([BATCH, GATHER, MULTI]))
        check = self.install(True)
        self.assertEqual(check['qualified_profiles'], ['main128'])
        self.assertEqual(len(check['native_libraries']), 6)
        self.install(False)
        installed = self.installed_runtime()
        action, name, profile = installed.prepare([])
        self.assertEqual((action, name), ('start', 'main128'))
        environ = installed.configure_environment(name, profile, {'QWEN36_PREFILL_SELECTED_SYNC': '1'})
        for key, value in CANDIDATE['env'].items():
            with self.subTest(key):
                if key.endswith('_LIBRARY'):
                    self.assertTrue(environ[key].startswith(str(self.root / 'release/lib/')))
                else:
                    self.assertEqual(environ[key], value)
        self.assertEqual(environ['QWEN36_MEMORY_RATIO'], '0.85')
        self.assertNotIn('QWEN36_PREFILL_SELECTED_SYNC', environ)  # inherited knobs are cleared
        argv = installed.serve_argv(profile)
        self.assertEqual(argv[argv.index('--memory-ratio') + 1], '0.85')
        self.assertIn('--moe-prefill-hit-d2d', argv)

    def test_t3_qualifies_only_with_its_evidence(self):
        with self.assertRaisesRegex(ValueError, 'decode_shared_overlap needs component_reports'):
            self.compose(overlap_component=[])
        decision = self.compose()
        del decision['profiles']['main128']['decode_shared_overlap_evidence']
        with self.assertRaisesRegex(ValueError, 'decode_shared_overlap requires decode_shared_overlap_evidence'):
            self.build(decision)
        bad = (('component partial', {'overlap': overlap_component(sorted(maker.OVERLAP_COMPONENT_CASES)[:3])}, 'T3 component'),
               ('component skipped', {'overlap': overlap_component(skipped=1)}, 'T3 component'),
               ('a/b inconclusive', {'ab': {'verdict': 'INCONCLUSIVE', 'performance_gate_pass': False}}, 'compare-ab PASS'),
               ('teacher of another run', {'teacher': teacher_report(['d' * 32])}, 'not bound to the listed candidate runs'),
               ('teacher not exact', {'teacher': dict(teacher_report([RUN_ID]), logit_comparisons=dict(
                   teacher_report([RUN_ID])['logit_comparisons'], max_abs=1e-3))}, 'identical logits'))
        for label, changes, text in bad:
            with self.subTest(label), self.assertRaisesRegex(ValueError, text):
                self.build(self.compose(**changes))
        receipt = self.build(self.compose())
        forged = copy.deepcopy(receipt)
        del forged['profiles']['main128']['decode_shared_overlap']
        self.qualify(forged)
        self.assertIn('needs component, teacher and A/B evidence', self.refused())

    def test_run_must_be_the_candidate_configuration(self):
        for label, env, text in (('C1 other threshold', {'QWEN36_PREFILL_SELECTED': '512'}, 'not an allowed value'),
                                 ('C1 shorter threshold', {'QWEN36_PREFILL_SELECTED': '128'}, 'differs from the profile'),
                                 ('C0 without invalidation', {'QWEN36_PREFILL_EPOCH_INVALIDATE': '0'}, 'must be unset or 1'),
                                 ('host embeddings off', {'QWEN36_HOST_EMBEDDINGS': '0'}, 'must be unset or 1')):
            with self.subTest(label), self.assertRaisesRegex(ValueError, text):
                self.build(self.compose(run=self.next_run(**env)))
        run = self.next_run()
        run['configuration']['memory_ratio'] = 0.75
        with self.assertRaisesRegex(ValueError, 'differs from the profile'):
            self.build(self.compose(run=run))
        run = self.next_run()
        run['configuration']['flags'] = []
        with self.assertRaisesRegex(ValueError, 'differs from the profile'):
            self.build(self.compose(run=run))

    def test_transport_modules_must_be_manifest_bound_prebuilts(self):
        self.add_modules(without=(MULTI,))
        with self.assertRaisesRegex(ValueError, 'switched-on kernel module is not a manifest-bound prebuilt: ' + MULTI):
            self.build(self.compose())

    def test_decode_kernel_library_abi_is_probed(self):
        self.put(self.oproj_source, base.stub('oproj_abi2').read_bytes())
        with self.assertRaisesRegex(ValueError, 'ABI differs'):
            self.build(self.compose())


L256_RUN_ID = 'e' * 32
L256_CONFIG = json.loads((HERE / 'teacher-long256-release.json').read_text())


def long256_teacher(deviates=True):
    """A long256 teacher report; with the 262144-context long-120000 deviation f1 saw (not bit-exact)."""
    report = teacher_report([L256_RUN_ID])
    report['candidate'] = 'teacher-long256-release'
    if deviates:
        report.update(bit_exact_pass=False, logit_comparisons=dict(
            report['logit_comparisons'], all_byte_equal=False, max_abs=1.232421875, max_relative_l2=0.0953468456864357))
    return report


class Long256Rerun(NextFixture):
    """long256 rerun: its own run receipts on this tree; the 0-diff main128 teacher can bind its libraries."""

    def setUp(self):
        super().setUp()
        profiles = copy.deepcopy(self.profiles)
        profiles['profiles']['long256'] = dict(PROFILES['profiles']['long256'], qualified=False)
        self.write_profiles(profiles)
        long256 = self.profiles['profiles']['long256']
        self.f1 = {'profiles': {'long256': {
            'configuration_sha256': constants.canonical_profile_sha256(long256), 'acceptance_status': 'accepted_with_known_limits',
            'known_limitations': ['f1 long256 limitation'],
            'evidence': [{'path': 'reports/long256-quality.json', 'sha256': 'a' * 64}],
            **{switch: {'library': 'x', 'library_sha256': 'a' * 64, 'run_receipts': ['reports/long256-quality.json'],
                        'bitexact_reports': ['reports/long256-quality.json']}
               for switch in ('cuda_nvfp4_prefill', 'cuda_gdn', 'cuda_fp8_gemv')}}}}

    def long256_run(self, **changes):
        run = self.run_receipt(run_id=L256_RUN_ID, profile='teacher-long256-release')
        run['configuration'] = {key: copy.deepcopy(L256_CONFIG[key])
                                for key in ('context', 'chunk', 'memory_ratio', 'qualified', 'flags', 'env')}
        run['native_libraries'] = {key: {'path': value, 'sha256': sha(self.library_source(key).read_bytes())}
                                   for key, value in L256_CONFIG['env'].items() if key.endswith('_LIBRARY')}
        run.update(changes)
        return run

    def rerun_decision(self, run=None, teacher=None, **changes):
        self.evidence()
        for relative, value in (('runs/l256.json', self.long256_run() if run is None else run),
                                ('reports/l256-teacher.json', long256_teacher() if teacher is None else teacher)):
            self.put(self.evidence_root / relative, json.dumps(value).encode())
        values = dict(long256='rerun', long256_run=['runs/l256.json'], long256_teacher=['reports/l256-teacher.json'],
                      long256_bitexact=['reports/next-teacher.json'], evidence=['reports/main128-quality.json'])
        values.update(changes)
        staged, decision = helper.compose(release.load_validator(), maker, self.profiles, self.f1,
                                          decision_args(self.evidence_root, **values))
        self.write_profiles(staged)
        return decision

    def test_rerun_qualifies_installs_and_starts_long256(self):
        decision = self.rerun_decision(long256_limitation=['long-120000 differs at 262144 context as in f1'])
        entry = decision['profiles']['long256']
        self.assertEqual(entry['run_receipts'], ['runs/l256.json'])
        self.assertEqual(entry['cuda_gdn_evidence'], {'run_receipts': ['runs/l256.json'],
                                                      'bitexact_reports': ['reports/next-teacher.json']})
        self.assertEqual(entry['known_limitations'], ['f1 long256 limitation', 'long-120000 differs at 262144 context as in f1'])
        self.assertIn('reports/l256-teacher.json', entry['evidence'])
        receipt = self.qualify(self.build(decision))
        self.assertEqual(sorted(receipt['profiles']), ['long256', 'main128'])
        self.assertEqual(receipt['profiles']['long256']['cuda_nvfp4_prefill']['run_receipts'], ['runs/l256.json'])
        self.assertEqual(self.install(True)['qualified_profiles'], ['long256', 'main128'])
        self.install(False)
        installed = self.installed_runtime()
        action, name, profile = installed.prepare(['select', 'long256'])
        self.assertEqual((action, name), ('select', 'long256'))
        environ = installed.configure_environment(name, profile, {})
        for key, value in L256_CONFIG['env'].items():
            if key in ('QWEN36_TEACHER_FIXTURE', 'QWEN36_LOGIT_REFERENCE'):
                self.assertNotIn(key, environ)
            elif key.endswith('_LIBRARY'):
                self.assertTrue(environ[key].startswith(str(self.root / 'release/lib/')), key)
            else:
                self.assertEqual(environ[key], value, key)
        for key in ('QWEN36_DECODE_SHARED_OVERLAP', 'QWEN36_HOST_EMBEDDINGS', 'QWEN36_OPROJ_GEMV', 'QWEN36_PREFILL_SELECTED'):
            self.assertNotIn(key, environ)
        self.assertEqual(environ['QWEN36_MEMORY_RATIO'], '0.73')

    def test_deviating_long256_teacher_cannot_bind_the_libraries(self):
        with self.assertRaisesRegex(ValueError, 'identical logits'):
            self.build(self.rerun_decision(long256_bitexact=[]))
        self.build(self.rerun_decision(long256_bitexact=[], teacher=long256_teacher(deviates=False)))

    def test_rerun_runs_must_be_this_tree_and_long256(self):
        stale = self.long256_run()
        stale['source_files'] = dict(stale['source_files'], **{'engine/engine.py': '0' * 64})
        with self.assertRaisesRegex(ValueError, 'runtime sources differ from staged bytes'):
            self.build(self.rerun_decision(run=stale))
        wrong = self.long256_run()
        wrong['configuration']['context'] = 131072
        with self.assertRaisesRegex(ValueError, 'differs from the profile'):
            self.build(self.rerun_decision(run=wrong))
        with self.assertRaisesRegex(ValueError, 'runtime sources differ from staged bytes'):
            self.build(self.carry_decision(stale))

    def carry_decision(self, f1_run):
        """carry keeps the f1 run receipts: bound to another runtime, make-qualification refuses them."""
        self.evidence()
        self.put(self.evidence_root / 'reports/long256-quality.json', json.dumps(f1_run).encode())
        staged, decision = helper.compose(release.load_validator(), maker, self.profiles, self.f1,
                                          decision_args(self.evidence_root, long256='carry',
                                                        evidence=['reports/main128-quality.json']))
        self.write_profiles(staged)
        return decision


class DecisionHelper(unittest.TestCase):
    """B: release-main128.py composes the decision from report names and the installed f1 receipt."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='qwen-release-main128-helper-')
        self.root = pathlib.Path(self.tmp.name)
        for relative in ('runs/next.json', 'reports/next-teacher.json', 'reports/next-fla.json', 'reports/next-overlap.json',
                         'reports/next-ab.json', 'reports/l256-run.json', 'reports/l256-teacher.json',
                         'reports/f1-l256-run.json', 'reports/f1-quality.json', 'reports/f1-b64.json'):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
        self.f1 = {'profiles': {
            'long256': {'configuration_sha256': constants.canonical_profile_sha256(PROFILES['profiles']['long256']),
                        'acceptance_status': 'accepted_with_known_limits', 'known_limitations': ['f1 limit'],
                        'evidence': [{'path': p, 'sha256': 'a' * 64} for p in ('reports/f1-l256-run.json', 'reports/f1-quality.json')],
                        **{switch: {'library': 'x', 'library_sha256': 'a' * 64, 'run_receipts': ['reports/f1-l256-run.json'],
                                    'bitexact_reports': ['reports/f1-quality.json']}
                           for switch in ('cuda_nvfp4_prefill', 'cuda_gdn', 'cuda_fp8_gemv')}},
            'baseline64': {'configuration_sha256': constants.canonical_profile_sha256(PROFILES['profiles']['baseline64']),
                           'acceptance_status': 'accepted', 'known_limitations': [],
                           'evidence': [{'path': 'reports/f1-b64.json', 'sha256': 'b' * 64}]}}}

    def tearDown(self):
        self.tmp.cleanup()

    def compose(self, profiles=PROFILES, **changes):
        return helper.compose(constants, maker, profiles, self.f1, decision_args(self.root, **changes))

    def test_drop_keeps_only_main128(self):
        staged, decision = self.compose()
        self.assertEqual({n: p['qualified'] for n, p in staged['profiles'].items()},
                         {'main128': True, 'long256': False, 'baseline64': False})
        self.assertEqual(set(decision['profiles']), {'main128'})
        entry = decision['profiles']['main128']
        self.assertEqual(entry['acceptance_status'], 'accepted')
        self.assertEqual(entry['cuda_gdn_fla_evidence']['component_reports'], ['reports/next-fla.json'])
        self.assertEqual(entry['decode_shared_overlap_evidence']['ab_reports'], ['reports/next-ab.json'])
        self.assertEqual({k: v for k, v in staged['profiles']['main128'].items() if k != 'qualified'},
                         {k: v for k, v in PROFILES['profiles']['main128'].items() if k != 'qualified'})

    def test_carry_and_rerun(self):
        staged, decision = self.compose(long256='carry', baseline64='carry', limitation=['owner caveat'])
        self.assertEqual(decision['profiles']['main128']['acceptance_status'], 'accepted_with_known_limits')
        self.assertEqual(decision['profiles']['baseline64'], {'acceptance_status': 'accepted', 'known_limitations': [],
                                                               'evidence': ['reports/f1-b64.json']})
        carried = decision['profiles']['long256']
        self.assertEqual(carried['cuda_gdn_evidence'], {'run_receipts': ['reports/f1-l256-run.json'],
                                                        'bitexact_reports': ['reports/f1-quality.json']})
        staged, decision = self.compose(long256='rerun', long256_run=['reports/l256-run.json'],
                                        long256_teacher=['reports/l256-teacher.json'])
        rerun = decision['profiles']['long256']
        self.assertEqual(rerun['run_receipts'], ['reports/l256-run.json'])
        self.assertEqual(rerun['cuda_fp8_gemv_evidence'], {'run_receipts': ['reports/l256-run.json'],
                                                           'bitexact_reports': ['reports/l256-teacher.json']})
        self.assertIn('reports/f1-quality.json', rerun['evidence'])
        with self.assertRaisesRegex(ValueError, 'rerun needs'):
            self.compose(long256='rerun')
        staged, decision = self.compose(long256='rerun', long256_run=['reports/l256-run.json'],
                                        long256_teacher=['reports/l256-teacher.json'],
                                        long256_bitexact=['reports/next-teacher.json'])
        rerun = decision['profiles']['long256']
        self.assertEqual(rerun['cuda_nvfp4_evidence']['bitexact_reports'], ['reports/next-teacher.json'])
        self.assertTrue({'reports/l256-teacher.json', 'reports/next-teacher.json'} <= set(rerun['evidence']))
        (self.root / 'l256-limits.json').write_text(json.dumps(['new tree caveat']))
        staged, decision = self.compose(long256='rerun', long256_run=['reports/l256-run.json'],
                                        long256_teacher=['reports/l256-teacher.json'],
                                        long256_limitations_file=str(self.root / 'l256-limits.json'))
        self.assertEqual(decision['profiles']['long256']['known_limitations'], ['new tree caveat'])
        with self.assertRaisesRegex(ValueError, 'only with --long256 rerun'):
            self.compose(long256='drop', long256_limitation=['x'])
        with self.assertRaisesRegex(ValueError, 'only with --long256 rerun'):
            self.compose(long256='carry', long256_run=['reports/l256-run.json'])

    def test_refusals(self):
        changed = copy.deepcopy(PROFILES)
        changed['profiles']['long256']['memory_ratio'] = 0.7
        with self.assertRaisesRegex(ValueError, 'configuration changed since the installed receipt'):
            self.compose(profiles=changed, long256='carry')
        for label, changes, text in (('no FLA', {'fla_component': []}, 'cuda_gdn_fla needs'),
                                     ('no A/B', {'ab': []}, 'ab_reports'),
                                     ('no teacher', {'teacher': []}, 'needs run receipts'),
                                     ('missing file', {'evidence': ['reports/absent.json']}, 'missing'),
                                     ('escape', {'evidence': ['../x.json']}, 'invalid evidence path'),
                                     ('blank limitation', {'limitation': [' ']}, 'non-blank')):
            with self.subTest(label), self.assertRaisesRegex(ValueError, text):
                self.compose(**changes)

    def test_cli_dry_run_and_never_overwrite(self):
        argv = ['decision', '--validator', str(SERVING / 'profile-runtime.py'), '--profiles-in', str(SERVING / 'profiles.json'),
                '--f1-receipt', str(self.root / 'absent.json'), '--evidence-root', str(self.root),
                '--run', 'runs/next.json', '--teacher', 'reports/next-teacher.json', '--fla-component', 'reports/next-fla.json',
                '--overlap-component', 'reports/next-overlap.json', '--ab', 'reports/next-ab.json',
                '--long256', 'drop', '--baseline64', 'drop']
        with contextlib.redirect_stdout(io.StringIO()) as out:
            helper.main(argv)
        self.assertEqual(json.loads(out.getvalue().splitlines()[-1])['qualified'], ['main128'])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['reports', 'runs'])
        outputs = ['--profiles-out', str(self.root / 'p.json'), '--decision-out', str(self.root / 'd.json')]
        with contextlib.redirect_stdout(io.StringIO()):
            helper.main(argv + outputs)
        self.assertTrue(json.loads((self.root / 'p.json').read_text())['profiles']['main128']['qualified'])
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            helper.main(argv + outputs)

    def test_export_tree_keeps_line_ending_only_files(self):
        if shutil.which('git') is None:
            self.skipTest('git missing')
        clone, source = self.root / 'clone', self.root / 'source'
        files = {'python/freetoken/a.py': b'A = 1\n', 'python/freetoken/b.py': b'B = 1\n', 'tests/t.py': b'T = 1\n'}
        for relative, data in files.items():
            (clone / relative).parent.mkdir(parents=True, exist_ok=True)
            (clone / relative).write_bytes(data)
        run = lambda *a: subprocess.run(['git', '-C', str(clone), '-c', 'core.autocrlf=false', *a], check=True, capture_output=True)
        run('init', '--quiet')
        run('add', '.')
        run('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '--quiet', '-m', 'base')
        for relative, data in {'python/freetoken/a.py': b'A = 1\r\n', 'python/freetoken/b.py': b'B = 2\r\n',
                               'python/freetoken/new.py': b'N = 1\r\n', 'tests/t.py': b'T = 1\n',
                               'python/freetoken/__pycache__/a.cpython-312.pyc': b'pyc'}.items():
            (source / relative).parent.mkdir(parents=True, exist_ok=True)
            (source / relative).write_bytes(data)
        self.assertEqual(helper.export_tree(source, clone), {'changed': 1, 'new': 1, 'deleted': 0})
        status = run('status', '--porcelain').stdout.decode().split('\n')
        self.assertEqual(sorted(line for line in status if line), [' M python/freetoken/b.py', '?? python/freetoken/new.py'])
        self.assertEqual((clone / 'python/freetoken/b.py').read_bytes(), b'B = 2\n')
        with self.assertRaisesRegex(ValueError, 'not a git checkout'):
            helper.export_tree(source, self.root / 'source')


# ------------------------------------------------------------------------------------------------ C: the shell script
FAKE_SYSTEMCTL = """#!/bin/sh
state=${FAKE_UNIT_STATE:-inactive}
case " $* " in *" --value "*) echo "$state"; exit 0;; esac
echo LoadState=loaded
echo ActiveState=$state
echo SubState=dead
"""
RENDER = (('/path/' 'to/qwen36-workspace/outputs/qwen36-api/optimization', None), ('/home/' 'USER/qwen36-api', 'root'),
          ('/path/' 'to/LLM/qwen36-api/model', 'model'), ('llm.' 'example.com', 'public'), ('EDGE_SSH' '_HOST', 'edge'))
SCRIPT_TOOLS = ('release-main128.sh', 'release-main128.py', 'release.py', 'make-qualification.py', 'export-release-evidence.py',
                'repo_paths.py', 'qwen36.cmd')


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash') and shutil.which('git'), 'POSIX bash/git sandbox only')
class ReleaseScriptSandbox(unittest.TestCase):
    """C: release-main128.sh end to end in a sandbox site (own root, overlay, venv fingerprint, model, units)."""

    def setUp(self):
        if base.stub() is None:
            self.skipTest('no C compiler for the synthetic ctypes libraries')
        self.tmp = tempfile.TemporaryDirectory(prefix='qwen_release_main128_')
        top = pathlib.Path(self.tmp.name)
        self.checkout, self.root, self.model = top / 'checkout', top / 'root', top / 'model'
        self.native = self.root / 'optimization'
        self.values = {'root': str(self.root), 'model': str(self.model), 'public': 'public.test', 'edge': 'edge-ssh.test'}
        self.bin = top / 'bin'
        self.bin.mkdir()
        (self.bin / 'systemctl').write_text(FAKE_SYSTEMCTL)
        (self.bin / 'systemctl').chmod(0o755)
        (top / 'tmp').mkdir()
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ.get('PATH', ''), TMPDIR=str(top / 'tmp'))
        for name in ('QWEN36_CHECKOUT', 'QWEN36_SERVING_SOURCE', 'QWEN36_HARNESS_SOURCE', 'QWEN36_FREETOKEN_SOURCE', 'QWEN36_MODEL_DIR'):
            self.env.pop(name, None)
        # checkout: tools, real serving files, synthetic runtime tree (real provenance/quality modules)
        for name in SCRIPT_TOOLS:
            self.put(self.checkout / 'harness' / name, (HERE / name).read_bytes())
        for name in release.FILES + ['qwen36-inference.service', 'manage.ps1', 'supervisor.ps1', 'watch-supervisor.ps1',
                                     'host-metrics.ps1', 'get-api-key.ps1']:
            self.put(self.checkout / 'serving' / name, (SERVING / name).read_bytes())
        runtime = dict(base.RUNTIME_FILES, **base.KERNEL_SOURCES, **NEXT_CSRC)
        runtime['quality_evidence.py'] = (FREETOKEN / 'python/freetoken/quality_evidence.py').read_bytes()
        source_tree = self.checkout / 'freetoken/python/freetoken'
        overlay = self.native / 'python/freetoken'
        for relative, data in runtime.items():
            self.put(source_tree / relative, data)
            self.put(overlay / relative, data)
        for relative in release.NATIVE_EXTENSIONS:
            self.put(overlay / relative, b'\x7fELF ' + relative.encode())
        site = base.make_site_packages(self.root)
        cache = site / constants.KERNEL_CACHE_PACKAGE / 'jit_cache'
        for name in NEXT_MODULES:
            if name != BATCH:
                self.put(cache / name / (name + '.so'), b'\x7fELF prebuilt ' + name.encode())
        manifest = base.kernel_manifest(overlay / 'kernel', cache, NEXT_MODULES)
        for tree in (source_tree, overlay):
            self.put(tree / 'kernel/kernel_cache_manifest.json', manifest)
        base.make_model(self.model)
        # rendered serving python in the overlay, as sync-v5 put it there
        for name in release.FILES:
            if name.endswith('.py'):
                self.put(self.native / name, self.render(SERVING / name))
        # pinned libraries at the release.py source paths
        sources = {'QWEN36_FUSED_ATTN_GATE_LIBRARY': ('private-libraries/libqwen_attention_gate.so', None),
                   'QWEN36_CUDA_NVFP4_LIBRARY': ('private-libraries/libqwen_nvfp4_bitexact.so', 'good'),
                   'QWEN36_CUDA_GDN_LIBRARY': ('private-cuda-gdn/pinned/libqwen36_gdn-a8a806eda3a90a65.so', 'gdn'),
                   'QWEN36_FP8_GEMV_LIBRARY': ('private-fp8gemv/libqwen_fp8_gemv-685c78cf89e60d02.so', 'fp8'),
                   'QWEN36_OPROJ_GEMV_LIBRARY': ('private-decode-kernels/libqwen_oproj_gemv-fa9302e31cb88a02.so', 'oproj'),
                   'QWEN36_MOE_DECODE_LIBRARY': ('private-decode-kernels/libqwen_moe_decode-83e90c43cb63ce11.so', 'moe')}
        libraries = {}
        for key, (relative, stub) in sources.items():
            data = b'\x7fELF qualified gate' if stub is None else base.stub(stub).read_bytes()
            libraries[key] = {'path': str(self.native / relative), 'sha256': sha(self.put(self.native / relative, data).read_bytes())}
        # S5 evidence: candidate run receipt bound to the runtime and the rendered service files
        staged = {relative: sha(data) for relative, data in runtime.items() if relative.endswith(maker.RUN_SOURCE_SUFFIXES)}
        run = {'profile': 'main-release-next-d5', 'run_id': RUN_ID, 'checkpoint': constants.CHECKPOINT['id'],
               'checkpoint_revision': constants.CHECKPOINT['revision'],
               'configuration': dict({k: CANDIDATE[k] for k in ('context', 'chunk', 'memory_ratio', 'qualified', 'flags')},
                                     env=dict(CANDIDATE['env'], QWEN36_TEACHER_FIXTURE='/x')),
               'source_files': staged, 'native_libraries': libraries,
               'service_source_files': {name: sha(self.render(SERVING / name)) for name in maker.RUN_SERVICE_FILES},
               'packages': {name: base.PACKAGE_VERSIONS[name] for name in maker.RUN_PACKAGES}}
        for relative, value in (('reports/next-run.json', run), ('reports/next-teacher.json', teacher_report([RUN_ID])),
                                ('reports/next-fla.json', fla_component(libraries['QWEN36_CUDA_GDN_LIBRARY']['sha256'])),
                                ('reports/next-overlap.json', overlap_component()),
                                ('reports/next-ab.json', {'verdict': 'PASS', 'performance_gate_pass': True}),
                                ('reports/next-soak.json', {'verdict': 'PASS'})):
            self.put(self.native / relative, json.dumps(value).encode())
        # a pre-existing installed state for the first release (its receipt is only read for carry)
        self.put(self.root / 'release/qualification.json', b'{}\n')
        self.put(self.checkout / 'config/local.env',
                 ('QWEN36_ROOT=%s\nQWEN36_NATIVE=%s\nQWEN36_MODEL_DIR=%s\nQWEN36_PUBLIC_HOST=public.test\nQWEN36_EDGE_SSH_HOST=edge-ssh.test\n'
                  % (self.root, self.native, self.model)).encode())

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def put(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def render(self, path):
        data = path.read_bytes()
        for placeholder, key in RENDER:
            value = str(self.native / 'config') if key is None else self.values[key]
            data = data.replace(placeholder.encode(), value.encode())
        return data

    def script(self, *args, state='inactive'):
        argv = ['bash', str(self.checkout / 'harness/release-main128.sh'), '--python', sys.executable, *args]
        return subprocess.run(argv, env=dict(self.env, FAKE_UNIT_STATE=state), capture_output=True, text=True, timeout=600)

    def release_args(self, label, mode, *extra):
        return (mode, '--label', label, '--run', 'reports/next-run.json', '--teacher', 'reports/next-teacher.json',
                '--fla-component', 'reports/next-fla.json', '--overlap-component', 'reports/next-overlap.json',
                '--ab', 'reports/next-ab.json', '--long256', 'drop', '--baseline64', 'drop', '--no-export', *extra)

    def tree(self):
        return {p.relative_to(self.tmp.name).as_posix(): p.read_bytes() for p in sorted(pathlib.Path(self.tmp.name).rglob('*'))
                if p.is_file() and not p.is_symlink()}

    def ok(self, result):
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
        return result.stdout

    def test_dry_run_and_check_write_nothing(self):
        before = self.tree()
        out = self.ok(self.script(*self.release_args('r1', '--dry-run')))
        self.assertIn('dry-run: nothing written', out)
        self.assertIn('decode_shared_overlap_evidence', out)
        self.assertEqual(self.tree(), before)
        out = self.ok(self.script(*self.release_args('r1', '--check'), state='active'))  # the service may run
        self.assertIn('"preflight": "PASS"', out)
        self.assertEqual(self.tree(), before)

    def test_long256_flags_reach_the_decision(self):
        refused = self.script(*self.release_args('r1', '--dry-run', '--long256-limitation', 'x',
                                                 '--long256-bitexact', 'reports/next-teacher.json'))
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn('--long256-bitexact, --long256-limitation only with --long256 rerun', refused.stderr)

    def test_overlay_drift(self):
        readiness = self.native / 'readiness.py'
        readiness.write_bytes(readiness.read_bytes() + b'# overlay-only edit\n')
        out = self.ok(self.script(*self.release_args('r1', '--dry-run')))
        self.assertIn('not run-bound): readiness.py', out)
        guard = self.native / 'guard.py'
        guard.write_bytes(guard.read_bytes() + b'# the runs served another guard\n')
        refused = self.script(*self.release_args('r1', '--dry-run'))
        self.assertEqual(refused.returncode, 65, refused.stderr)
        self.assertIn('run-bound service files differ from the overlay (rendered): guard.py', refused.stderr)

    def test_apply_rollback_and_refusals(self):
        refused = self.script(*self.release_args('r1', '--apply'), state='active')
        self.assertEqual(refused.returncode, 73, refused.stderr)
        self.assertIn('stop it first', refused.stderr)
        self.assertFalse((self.native / 'release-staging/r1').exists())  # refused before staging
        self.ok(self.script(*self.release_args('r1', '--apply')))
        first = (self.root / 'release/qualification.json').read_bytes()
        self.assertEqual(first, (self.native / 'release-staging/r1/serving/qualification.json').read_bytes())
        reused = self.script(*self.release_args('r1', '--apply'))
        self.assertEqual(reused.returncode, 73)
        self.assertIn('never reused', reused.stderr)
        out = self.ok(self.script(*self.release_args('r2', '--apply', '--evidence', 'reports/next-soak.json')))
        self.assertIn('"installed_gate": "PASS"', out)
        record = json.loads((self.native / 'release-staging/r2/install.json').read_text())
        self.assertTrue(pathlib.Path(record['f1_snapshot']).is_file())
        second = (self.root / 'release/qualification.json').read_bytes()
        self.assertNotEqual(first, second)
        before = self.tree()
        self.ok(self.script('--rollback', '--dry-run', '--label', 'r2'))
        self.assertIn('"rollback_check": "PASS"', self.ok(self.script('--rollback', '--check', '--label', 'r2')))
        self.assertEqual(self.tree(), before)
        self.assertEqual(self.script('--rollback', '--apply', '--label', 'r2', state='active').returncode, 73)
        out = self.ok(self.script('--rollback', '--apply', '--label', 'r2'))
        self.assertIn('"installed_gate": "PASS"', out)
        self.assertEqual((self.root / 'release/qualification.json').read_bytes(), first)
        installed = json.loads((self.root / 'profiles.json').read_text())
        self.assertTrue(installed['profiles']['main128']['qualified'])
        self.assertTrue(constants.ROOT != self.root)  # the real service root was never involved


if __name__ == '__main__':
    unittest.main(verbosity=2)
