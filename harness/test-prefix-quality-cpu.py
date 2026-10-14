"""Pure stdlib gate regressions: no tokenizer, authorization, network or GPU."""
import copy
import importlib.util
import json
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prefix_quality', HERE/'prefix-quality.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return gate.canonical([messages, kwargs])

    def encode(self, text, **_kwargs):
        return list(text.encode())


class PrefixGateTests(unittest.TestCase):
    def test_strict_json_accepts_exact_object_only(self):
        self.assertTrue(gate.verdict(' \n'+gate.canonical(gate.GOLD)+'\n', gate.GOLD)['pass'])
        for text in ('```json\n'+gate.canonical(gate.GOLD)+'\n```',
                     'answer: '+gate.canonical(gate.GOLD), gate.canonical(gate.GOLD)+' trailing',
                     '[1,2]', 'null', 'NaN', '{"count":23,"count":23}'):
            with self.subTest(text_sha=gate.digest(text)):
                self.assertFalse(gate.verdict(text, gate.GOLD)['pass'])

    def test_types_keys_and_values_are_not_coerced(self):
        for changed in (dict(gate.GOLD, count=23.0), dict(gate.GOLD, price='14'),
                        dict(gate.GOLD, extra=0), dict(gate.GOLD, total=321)):
            self.assertFalse(gate.verdict(gate.canonical(changed), gate.GOLD)['pass'])
        self.assertFalse(gate.verdict('{"a":true}', {'a': 1})['pass'])
        self.assertFalse(gate.verdict('{"a":NaN}', {'a': 1})['pass'])
        self.assertFalse(gate.verdict('{"a":1e999}', {'a': 1})['pass'])

    def test_deterministic_targets_and_edited_primers(self):
        def document(_tokenizer, size, variant):
            return ('Record distractor '+str(variant)+'\n')*size
        first = gate.build_cases(FakeTokenizer(), document, 'fixture', [32])
        second = gate.build_cases(FakeTokenizer(), document, 'fixture', [32])
        self.assertEqual([c['manifest'] for c in first], [c['manifest'] for c in second])
        self.assertEqual(len(first), 4)
        self.assertEqual(len({c['id'] for c in first}), 4)
        for case in first:
            self.assertGreater(case['manifest']['shared_prompt_tokens'], 64)
            self.assertNotIn(case['manifest']['target_sha256'], case['manifest']['primer_sha256'])
            self.assertEqual(case['manifest']['gold_sha256'], gate.digest(case['expected']))
            # No model/token text or arguments enter the manifest.
            encoded = gate.canonical(case['manifest'])
            for forbidden in ('lookup_fixture', 'АЛЬФА', 'Ирина', 'Record distractor', 'arguments'):
                self.assertNotIn(forbidden, encoded)

    def test_metrics_drop_arbitrary_text(self):
        result = {'status': 200, 'error': {'text': 'SENTINEL_SECRET'},
                  'engine': {'note': 'SENTINEL_PROMPT', 'duration': 1.0, 'bad': float('nan')},
                  'content': 'SENTINEL_OUTPUT', 'usage': {'completion_tokens': 8},
                  'engine_metrics_status': 'joined', 'finish_reason': 'stop'}
        encoded = gate.canonical(gate.result_metrics(result))
        self.assertNotIn('SENTINEL', encoded)
        self.assertNotIn('NaN', encoded)
        self.assertIn('completion_tokens', encoded)

    def test_cached_tokens_accepts_zero_and_rejects_bool(self):
        self.assertEqual(gate.cached_tokens({'cached_prompt_tokens': 0}), 0)
        self.assertEqual(gate.cached_tokens({'usage': {'prompt_tokens_details': {'cached_tokens': 4032}}}), 4032)
        self.assertIsNone(gate.cached_tokens({'cached_prompt_tokens': True}))
        self.assertIsNone(gate.cached_tokens({'cached_prompt_tokens': -1}))

    def test_anchor_excludes_instruction_example(self):
        # Token 99 appears in the shared tools example AND after generation.
        primer = [1, 99, 2, 3, 4]
        target = [1, 99, 2, 3, 4, 5, 99, 8]
        self.assertEqual(gate.tool_anchor_boundary(target, primer, [99]), 7)
        self.assertIsNone(gate.tool_anchor_boundary([1, 99, 2, 3, 4, 5], primer, [99]))
        self.assertIsNone(gate.tool_anchor_boundary(target, primer, [99, 100]))

    def test_runtime_matches_source_and_config_but_not_run_identity(self):
        receipt = {'run_id': 'a'*32, 'profile': 'example', 'source_sha256': 'b'*64,
                   'configuration': {'context': 131072, 'chunk': 4096,
                      'env': {'QWEN36_PREFIX_MARGIN': '8'}, 'flags': ['--enable-special-token-ckpt']}}
        left = gate.run_identity(receipt)
        right = gate.run_identity(dict(receipt, run_id='c'*32))
        self.assertTrue(gate.same_runtime(left, right))
        self.assertNotEqual(left['run_id_sha256'], right['run_id_sha256'])
        self.assertEqual(left['prefix_margin'], 8)
        self.assertTrue(left['anchor_enabled'])
        right['source_sha256'] = 'd'*64
        self.assertFalse(gate.same_runtime(left, right))
        with self.assertRaises(ValueError):
            gate.run_identity(dict(receipt, source_sha256=None))
        teacher = copy.deepcopy(receipt)
        teacher['configuration']['env']['QWEN36_TEACHER_FIXTURE'] = 'fixture'
        with self.assertRaises(ValueError):
            gate.run_identity(teacher)

    def test_functional_fail_and_missing_evidence_never_pass(self):
        baseline = {'gate_pass': True}
        row = {'verdict': {'pass': True}, 'status': 200, 'fresh_zero_cache': True,
               'prefix_hit_observed': True, 'primers': [{'pass': True}], 'kind': 'last_message_edit',
               'anchor_observed': False, 'token_count_matches': True, 'request_run_matches': True,
               'no_unexpected_tools': True, 'metrics': {'engine_metrics_joined': True}}
        self.assertTrue(gate.phase_gate('fresh', row))
        self.assertTrue(gate.phase_gate('cached', row, baseline))
        for key in ('prefix_hit_observed', 'token_count_matches', 'request_run_matches', 'no_unexpected_tools'):
            bad = dict(row, **{key: False})
            self.assertFalse(gate.phase_gate('cached', bad, baseline))
        bad = copy.deepcopy(row); bad['verdict']['pass'] = False
        self.assertFalse(gate.phase_gate('cached', bad, baseline))
        bad = copy.deepcopy(row); bad['primers'][0]['pass'] = False
        self.assertFalse(gate.phase_gate('cached', bad, baseline))
        self.assertFalse(gate.phase_gate('cached', row, {'gate_pass': False}))
        bad = dict(row, kind='special_anchor')
        self.assertTrue(gate.phase_gate('cached', bad, baseline, False))
        self.assertFalse(gate.phase_gate('cached', bad, baseline, True))
        self.assertTrue(gate.phase_gate('cached', dict(bad, anchor_observed=True), baseline, True))

    def test_two_phase_restart_guard_and_sanitized_report(self):
        receipt = {'run_id': 'a'*32, 'profile': 'example', 'source_sha256': 'b'*64,
                   'configuration': {'context': 131072, 'chunk': 128,
                                     'flags': ['--enable-special-token-ckpt']}}
        case_id = gate.digest('case')
        target = [{'role': 'user', 'content': 'SENTINEL_TARGET'}]
        primer = [{'role': 'user', 'content': 'SENTINEL_PRIMER'}]
        fixture = {'id': case_id, 'kind': 'special_anchor', 'length': 512, 'target': target,
                   'primers': [primer], 'tools': None, 'expected': gate.GOLD,
                   'manifest': {'case_id': case_id, 'prompt_tokens': 1000,
                                'shared_prompt_tokens': 900, 'anchor_boundary_tokens': 950,
                                'generation_prefix_aligned': True}}
        current_phase = ['fresh']
        requests = []
        def send(messages, tools, force_tool=False):
            requests.append(force_tool)
            result = {'status': 200, 'usage': {'prompt_tokens': 1000}, 'finish_reason': 'stop',
                      'engine_metrics_status': 'joined', 'engine': {'run_id': receipt['run_id']},
                      'cached_prompt_tokens': 0 if current_phase[0] == 'fresh' else 1000,
                      'tool_call_valid': True, 'tool_call_count': 1 if force_tool else 0}
            return result, 'SENTINEL_PRIMER_RESPONSE' if force_tool else gate.canonical(gate.GOLD)
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)/'suite'
            bench = SimpleNamespace(ROOT=Path(directory), run_receipt=lambda: copy.deepcopy(receipt), document=None)
            transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: None))
            with patch.dict(sys.modules, {'bench': bench, 'transformers': transformers}), \
                    patch.object(gate, 'build_cases', return_value=[fixture]), patch.object(gate, 'send', side_effect=send), \
                    redirect_stdout(io.StringIO()):
                args = ['prefix-quality.py', '--name', 'unit', '--lengths', '512', '--report-prefix', str(prefix)]
                with patch.object(sys, 'argv', args+['--phase', 'fresh']):
                    gate.main()
                with patch.object(sys, 'argv', args+['--phase', 'cached']):
                    with self.assertRaisesRegex(ValueError, 'fresh_receipt_mismatch_or_no_restart'):
                        gate.main()
                self.assertEqual(len(requests), 1)
                receipt['run_id'] = 'c'*32
                current_phase[0] = 'cached'
                with patch.object(sys, 'argv', args+['--phase', 'cached', '--require-anchor']):
                    gate.main()
            text = Path(str(prefix)+'-cached.json').read_text()
            report = json.loads(text)
            self.assertTrue(report['gate_pass'])
            self.assertTrue(report['anchor_gate_pass'])
            self.assertTrue(report['rows'][0]['fresh_cached_output_sha_equal'])
            self.assertNotIn('SENTINEL', text)
            self.assertNotIn('АЛЬФА', text)
            self.assertNotIn('Ирина', text)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PrefixGateTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'pass': result.wasSuccessful(), 'model_requests': 0, 'gpu_launched': False,
              'script_sha256': gate.hashlib.sha256((HERE/'prefix-quality.py').read_bytes()).hexdigest()}
    gate.write_report(HERE/'prefix-quality-cpu.json', report)
    print(json.dumps(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
