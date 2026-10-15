"""S2 memory/profile contracts: memory_ratio stage configs, host embeddings switch, T3 evidence gate."""
import copy
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from repo_paths import FREETOKEN, HARNESS, SERVING


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load('s2_profile_runtime', SERVING / 'profile-runtime.py')
maker = load('s2_make_qualification', HARNESS / 'make-qualification.py')
component = load('s2_component_report', HARNESS / 'component-report.py')

BASE = 'main-release-shortroute-fla-wu-overlap'
STAGES = {'main-release-next-r080': (0.80, False), 'main-release-next-r082': (0.82, False),
          'main-release-next-r0845': (0.845, False), 'main-release-next-he': (0.75, True),
          'main-release-next-he-r085': (0.85, True), 'main-release-next-he-r090': (0.90, True),
          'main-release-next-he-r094': (0.94, True)}


def config(name):
    return json.loads((HARNESS / (name + '.json')).read_text())


class StageConfigs(unittest.TestCase):
    def test_stages_differ_from_the_combined_candidate_only_in_memory(self):
        base = config(BASE)
        for name, (ratio, host) in STAGES.items():
            with self.subTest(name=name):
                cfg = config(name)
                self.assertIs(cfg['qualified'], False)
                self.assertEqual(cfg['memory_ratio'], ratio)
                expected = copy.deepcopy(base)
                expected['memory_ratio'] = ratio
                if host:
                    expected['env']['QWEN36_HOST_EMBEDDINGS'] = '1'
                self.assertEqual(cfg, expected)

    def test_stages_map_to_production_profiles(self):
        for name, (ratio, host) in STAGES.items():
            with self.subTest(name=name):
                mapped = maker.run_equivalent_profile(config(name), runtime.FEATURE_SWITCHES,
                                                      runtime.KERNEL_LOADER_ENV, runtime.VALUE_FIELDS)
                self.assertEqual(mapped['memory_ratio'], ratio)
                self.assertIs(mapped['host_embeddings'], host)
                self.assertIs(mapped['decode_shared_overlap'], True)
                profile = {k: v for k, v in mapped.items() if v is not None}  # unset value fields
                runtime.validate_profile_schema('main128', dict(profile, qualified=False))

    def test_host_embeddings_value_is_exact(self):
        cfg = config('main-release-next-he')
        for value in ('true', '2', ''):
            cfg['env']['QWEN36_HOST_EMBEDDINGS'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                maker.run_equivalent_profile(cfg, runtime.FEATURE_SWITCHES, runtime.KERNEL_LOADER_ENV,
                                             runtime.VALUE_FIELDS)

    def test_teacher_names_are_allowlisted(self):
        source = (FREETOKEN / 'python/freetoken/engine/qualification.py').read_text()
        allowed = set(re.findall(r"'(teacher-[a-z0-9.-]+)'", source))
        for name in STAGES:
            self.assertIn('teacher-' + name, allowed)

    def test_memory_ratio_ceiling_admits_every_stage(self):
        self.assertGreater(runtime.MAX_MEMORY_RATIO, max(ratio for ratio, _ in STAGES.values()))
        profile = json.loads((SERVING / 'profiles.json').read_text())['profiles']['main128']
        for ratio, _ in STAGES.values():
            runtime.validate_profile_schema('main128', dict(profile, memory_ratio=ratio, qualified=False))
        with self.assertRaisesRegex(ValueError, 'memory ratio'):
            runtime.validate_profile_schema('main128', dict(profile, memory_ratio=runtime.MAX_MEMORY_RATIO))


class HostEmbeddingsSwitch(unittest.TestCase):
    def test_switch_exports_exactly_its_variable(self):
        profile = json.loads((SERVING / 'profiles.json').read_text())['profiles']['main128']
        env = {'QWEN36_HOST_EMBEDDINGS': '1'}
        runtime.configure_environment('main128', dict(profile, host_embeddings=False), env)
        self.assertNotIn('QWEN36_HOST_EMBEDDINGS', env)
        runtime.configure_environment('main128', dict(profile, host_embeddings=True), env)
        self.assertEqual(env['QWEN36_HOST_EMBEDDINGS'], '1')
        for value in ('1', 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime.validate_profile_schema('main128', dict(profile, host_embeddings=value))
        self.assertNotEqual(runtime.canonical_profile_sha256(dict(profile, host_embeddings=True)),
                            runtime.canonical_profile_sha256(dict(profile, host_embeddings=False)))


def _overlap_profile(qualified):
    profile = json.loads((SERVING / 'profiles.json').read_text())['profiles']['main128']
    return dict(profile, decode_shared_overlap=True, qualified=qualified)


def _record(paths, overlap=None):
    record = {'configuration_sha256': 'a' * 64, 'acceptance_status': 'accepted', 'known_limitations': [],
              'evidence': [{'path': p, 'sha256': 'b' * 64} for p in paths]}
    if overlap is not None:
        record['decode_shared_overlap'] = overlap
    return record


class OverlapQualificationGate(unittest.TestCase):
    EVIDENCE = {'component_reports': ['component.json'], 'run_receipts': ['run.json'],
                'bitexact_reports': ['teacher.json'], 'ab_reports': ['ab.json']}
    PATHS = ('component.json', 'run.json', 'teacher.json', 'ab.json')

    def test_gate_is_closed_until_the_switch_is_listed(self):
        self.assertEqual(runtime.QUALIFIABLE_EXPERIMENTAL, frozenset())
        with self.assertRaisesRegex(ValueError, 'experimental'):
            runtime.validate_profile_schema('main128', _overlap_profile(True))
        with self.assertRaisesRegex(ValueError, 'experimental'):
            runtime._experimental_evidence_valid('main128', _overlap_profile(True),
                                                 _record(self.PATHS, dict(self.EVIDENCE)))
        runtime.validate_profile_schema('main128', _overlap_profile(False))

    def test_open_gate_requires_bound_evidence(self):
        with patch.object(runtime, 'QUALIFIABLE_EXPERIMENTAL', frozenset({'decode_shared_overlap'})):
            profile = _overlap_profile(True)
            runtime.validate_profile_schema('main128', profile)
            runtime._experimental_evidence_valid('main128', profile, _record(self.PATHS, dict(self.EVIDENCE)))
            with self.assertRaisesRegex(ValueError, 'needs component, teacher and A/B'):
                runtime._experimental_evidence_valid('main128', profile, _record(self.PATHS))
            for key in self.EVIDENCE:
                for bad in ([], ['unlisted.json'], ['run.json', 'run.json'], 'run.json'):
                    evidence = dict(self.EVIDENCE, **{key: bad})
                    with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                        runtime._experimental_evidence_valid('main128', profile, _record(self.PATHS, evidence))
                missing = {k: v for k, v in self.EVIDENCE.items() if k != key}
                with self.assertRaises(ValueError):
                    runtime._experimental_evidence_valid('main128', profile, _record(self.PATHS, missing))
            # refill stays unsupported even with an open gate
            with self.assertRaisesRegex(ValueError, 'refill'):
                runtime.validate_profile_schema('main128', dict(profile, layer_major_refill=True))

    def test_evidence_for_a_disabled_switch_is_refused(self):
        profile = dict(_overlap_profile(True), decode_shared_overlap=False)
        with self.assertRaisesRegex(ValueError, 'does not enable'):
            runtime._experimental_evidence_valid('main128', profile, _record(self.PATHS, dict(self.EVIDENCE)))
        runtime._experimental_evidence_valid('main128', profile, _record(self.PATHS))

    def test_receipt_records_admit_the_experimental_key(self):
        self.assertEqual(set(maker.EXPERIMENTAL_DECISION_EVIDENCE), set(runtime.EXPERIMENTAL_EVIDENCE))


def _component(names=None, **over):
    rows = [{'case': c, 'pass': True, 'outcome': 'passed'} for c in sorted(names or maker.OVERLAP_COMPONENT_CASES)]
    report = {'schema': 1, 'kind': 'gpu-component-report', 'suite': maker.OVERLAP_COMPONENT_SUITE,
              'suite_sha256': 'c' * 64, 'cases': len(rows), 'skipped': 0, 'all_pass': True, 'results': rows}
    report.update(over)
    return report


class OverlapEvidenceChecks(unittest.TestCase):
    def test_component_report(self):
        maker.check_overlap_component_report(_component())
        partial = sorted(maker.OVERLAP_COMPONENT_CASES)[:3]
        for bad in (_component(partial), _component(skipped=1), _component(all_pass=False),
                    _component(suite='freetoken/tests/moe/test_short_route_gpu.py'), _component(cases=5), None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                maker.check_overlap_component_report(bad)

    def test_ab_report(self):
        maker.check_ab_report({'verdict': 'PASS', 'performance_gate_pass': True})
        for bad in ({'verdict': 'INCONCLUSIVE', 'performance_gate_pass': False},
                    {'verdict': 'PASS', 'performance_gate_pass': False}, {}, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                maker.check_ab_report(bad)

    def test_junit_conversion(self):
        cases = sorted(maker.OVERLAP_COMPONENT_CASES)
        with tempfile.TemporaryDirectory() as temp:
            junit = Path(temp) / 'run.xml'
            body = ''.join(f'<testcase classname="t" name="{c}"/>' for c in cases)
            junit.write_text(f'<testsuites><testsuite name="pytest">{body}</testsuite></testsuites>')
            out = Path(temp) / 'report.json'
            self.assertEqual(component.main(['--junit', str(junit), '--suite', maker.OVERLAP_COMPONENT_SUITE,
                                             '--root', str(FREETOKEN.parent), '--out', str(out)]), 0)
            maker.check_overlap_component_report(json.loads(out.read_text()))
            with self.assertRaises(FileExistsError):
                component.main(['--junit', str(junit), '--suite', maker.OVERLAP_COMPONENT_SUITE,
                                '--root', str(FREETOKEN.parent), '--out', str(out)])
            skipped = body.replace('/>', '><skipped message="no cuda"/></testcase>', 1)
            junit.write_text(f'<testsuites><testsuite name="pytest">{skipped}</testsuite></testsuites>')
            out2 = Path(temp) / 'skipped.json'
            self.assertEqual(component.main(['--junit', str(junit), '--suite', maker.OVERLAP_COMPONENT_SUITE,
                                             '--root', str(FREETOKEN.parent), '--out', str(out2)]), 1)
            with self.assertRaises(ValueError):
                maker.check_overlap_component_report(json.loads(out2.read_text()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
