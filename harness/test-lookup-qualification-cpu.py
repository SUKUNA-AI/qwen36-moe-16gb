"""CPU-only checks of lookup-qualification.py: no model, GPU, CUDA, torch, network or systemd.

HTTP is replaced by an in-process fake `requests` module (including a model of the
guard.py APIGuard rate bucket on a virtual clock); systemd/port/lock calls are patched
or use temporary files. Run inside WSL with CUDA_VISIBLE_DEVICES= (empty).
"""
import ast
import contextlib
import copy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lookupq', HERE/'lookup-qualification.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)

SRC, ART = 'a'*64, 'b'*64
LOOKUP_FILES = {rel: 'c'*64 for rel in q.LOOKUP_SOURCE_FILES}
CFG = {arm: q.digest(arm) for arm in q.ARMS}
CASES = q.fixtures()
SPEC_OK = {'attempts': 3, 'accepted': 6, 'proposed': 9, 'emitted': 9, 'wall_timing_includes_publication': True,
           'verify_replay_in_engine_gpu_timers': False, 'publication_failed': False}
OK_CANCEL = [{'arm': 'overlap', 'pass': True}, {'arm': 'lookup', 'pass': True}]


def make_row(case, repeat, arm, elapsed=1.0):
    content = case['expected'] if case['expected'] is not None else 'deterministic free output'
    value = q.evaluate(case, {'status': 200, 'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120},
                              'finish_reason': 'stop', '_content': content, '_reasoning': '',
                              'elapsed_s': elapsed, 'ttft_content_s': .05})
    lookup_active = arm == 'lookup' and case['sampling'] == 'greedy' and case['case'] != 'eos-short'
    value.update(repeat=repeat, arm=arm, source_sha256=SRC, artifact_sha256=ART, cached_prompt_tokens=0,
                 engine_metrics_joined=True, speculation=dict(SPEC_OK) if lookup_active else {})
    return value


def make_run(repeat, arm):
    return {'repeat': repeat, 'arm': arm, 'source_sha256': SRC, 'artifact_sha256': ART,
            'config_sha256': CFG[arm], 'lookup_source_files': dict(LOOKUP_FILES)}


def lookup_fast(arm, case, repeat):
    if arm == 'lookup' and case['case'] in q.SPEED_WIN_CASES:
        return .80+.002*repeat
    return 1.0+.001*((repeat+len(arm)) % 3)


def dataset(pairs, speed=lookup_fast):
    rows = [make_row(case, repeat, arm, speed(arm, case, repeat))
            for repeat in range(pairs) for arm in q.ORDERS[repeat] for case in CASES]
    runs = [make_run(repeat, arm) for repeat in range(pairs) for arm in q.ORDERS[repeat]]
    return rows, runs


def verdict(rows, runs, pairs, state='complete', cancellation=OK_CANCEL):
    summary = q.paired_summary(rows, runs, pairs, CFG)
    return q.decide(summary, state, pairs, cancellation), summary


def find(rows, repeat, arm, case):
    return next(row for row in rows if (row['repeat'], row['arm'], row['case']) == (repeat, arm, case))


class Profiles(unittest.TestCase):
    def test_arms_differ_only_in_overlap_and_spec_switches(self):
        base = {'context': 131072, 'chunk': 4096, 'memory_ratio': .73, 'env': {'QWEN36_FUSED_ATTN_GATE': '1'}, 'flags': []}
        old = copy.deepcopy(base)
        arms = q.arm_configs(base)
        self.assertEqual(base, old)
        self.assertEqual(set(arms), set(q.ARMS))
        self.assertFalse(q.env_true(arms['overlap']['env']['FREETOKEN_DISABLE_OVERLAP_SCHEDULING']))
        self.assertTrue(q.env_true(arms['nooverlap']['env']['FREETOKEN_DISABLE_OVERLAP_SCHEDULING']))
        self.assertTrue(q.env_true(arms['lookup']['env']['FREETOKEN_DISABLE_OVERLAP_SCHEDULING']))
        self.assertEqual([arms[a]['env']['QWEN36_SPEC'] for a in q.ARMS], ['', '', 'lookup'])
        self.assertTrue(all(arms[a]['env']['QWEN36_SPEC_SAMPLING'] == '0' for a in q.ARMS))
        stripped = []
        for arm in q.ARMS:
            config = copy.deepcopy(arms[arm])
            self.assertIs(config['qualified'], False)
            del config['env']['FREETOKEN_DISABLE_OVERLAP_SCHEDULING'], config['env']['QWEN36_SPEC']
            stripped.append(config)
        self.assertTrue(stripped[0] == stripped[1] == stripped[2])

    def test_env_parser_mirrors_freetoken(self):
        path = __import__('repo_paths').FREETOKEN/'python/freetoken/env.py'
        if not path.is_file():
            self.skipTest('FreeToken fork env.py not present')
        module_spec = importlib.util.spec_from_file_location('freetoken_env_probe', path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        for value in ('0', '1', 'true', 'True', 'yes', '', 'no', 'on'):
            self.assertEqual(q.env_true(value), module._TO_BOOL(value), value)

    def test_base_must_be_plain_overlap_profile(self):
        good = {'context': 1, 'chunk': 1, 'memory_ratio': .73, 'flags': [], 'env': {}}
        for key in ('QWEN36_TEACHER_FIXTURE', 'QWEN36_NSYS_FORWARDS', 'QWEN36_SPEC', 'QWEN36_SPEC_SAMPLING',
                    'QWEN36_LOGIT_REFERENCE', 'QWEN36_KV_CALIBRATE'):
            with self.assertRaises(ValueError):
                q.arm_configs(dict(good, env={key: 'on'}))
        for value in ('1', 'true', 'YES'):
            with self.assertRaises(ValueError):
                q.arm_configs(dict(good, env={'FREETOKEN_DISABLE_OVERLAP_SCHEDULING': value}))
        for flags in (['--page-size', '16'], ['--page-size=8']):
            with self.assertRaises(ValueError):
                q.arm_configs(dict(good, flags=flags))
        q.arm_configs(dict(good, flags=['--page-size=1'], env={'FREETOKEN_DISABLE_OVERLAP_SCHEDULING': '0'}))
        with self.assertRaises(ValueError):
            q.arm_configs({'context': 1, 'env': {}, 'flags': []})

    def test_schedule_is_balanced_permutations(self):
        order = q.schedule(5)
        self.assertEqual(len(order), 15)
        self.assertEqual(q.schedule(1), [(0, arm) for arm in q.ORDERS[0]])
        lookup_first = []
        for pair in range(5):
            arms = [arm for repeat, arm in order if repeat == pair]
            self.assertEqual(sorted(arms), sorted(q.ARMS))
            lookup_first.append(arms.index('lookup') < arms.index('overlap'))
        self.assertEqual(lookup_first, [False, True, False, True, False])
        for bad in (0, 6):
            with self.assertRaises(ValueError):
                q.schedule(bad)


class Fixtures(unittest.TestCase):
    def test_payloads_share_seed_thinking_and_greedy_is_engine_greedy(self):
        names = [case['case'] for case in CASES]
        self.assertEqual(len(names), len(set(names)))
        for case in CASES:
            doc = q.payload(case)
            self.assertEqual(doc, q.payload(copy.deepcopy(case)))
            self.assertEqual(doc['seed'], q.SEED)
            self.assertIs(doc['chat_template_kwargs']['enable_thinking'], False)
            self.assertIs(doc['ignore_eos'], False)
            if case['sampling'] == 'greedy':
                # FreeToken SamplingParams.is_greedy: (temperature<=0 or top_k==1) and top_p==1.0
                self.assertTrue(doc['temperature'] <= 0 and doc['top_p'] == 1.0)
            else:
                self.assertFalse({'temperature', 'top_p', 'top_k'} & set(doc))
        self.assertEqual({c['case'] for c in CASES if c['sampling'] != 'greedy'}, {'sampled-default'})
        self.assertGreater(len(q.fixture('copy-code-long')['messages'][1]['content']), 30000)
        self.assertTrue(set(q.EXERCISE_CASES) <= set(names) and set(q.SPEED_WIN_CASES) <= set(names))

    def test_expected_output_and_raw_privacy(self):
        case = q.fixture('copy-code')
        value = make_row(case, 0, 'overlap')
        self.assertTrue(value['functional_pass'])
        self.assertNotIn(case['expected'], json.dumps(value, ensure_ascii=False))
        self.assertNotIn('_content', value)
        usage = {'completion_tokens': 2, 'prompt_tokens': 1, 'total_tokens': 3}
        self.assertFalse(q.evaluate(case, {'status': 200, '_content': 'wrong', 'usage': usage, 'finish_reason': 'stop'})['functional_pass'])
        self.assertFalse(q.evaluate(case, {'status': 200, '_content': case['expected'], '_reasoning': 'x', 'usage': usage,
                                           'finish_reason': 'stop'})['functional_pass'])
        free = q.fixture('free-code')
        self.assertFalse(q.evaluate(free, {'status': 200, '_content': '  ', 'usage': usage, 'finish_reason': 'stop'})['functional_pass'])
        self.assertFalse(q.evaluate(free, {'status': 500, '_content': 'x', 'usage': usage, 'finish_reason': 'stop'})['functional_pass'])

    def test_budget_and_length_rules(self):
        case = q.fixture('copy-code')
        exhausted = {'status': 200, '_content': case['expected'], 'finish_reason': 'length',
                     'usage': {'completion_tokens': case['max_tokens']}}
        self.assertFalse(q.evaluate(case, exhausted)['functional_pass'])
        self.assertFalse(q.evaluate(case, dict(exhausted, finish_reason='stop', _content=case['expected']))['functional_pass'])
        free = q.fixture('free-code')
        self.assertTrue(q.evaluate(free, {'status': 200, '_content': 'x', 'finish_reason': 'length',
                                          'usage': {'completion_tokens': free['max_tokens']}})['functional_pass'])
        self.assertFalse(q.evaluate(free, {'status': 200, '_content': 'x', 'finish_reason': 'length',
                                           'usage': {'completion_tokens': 5}})['functional_pass'])
        self.assertFalse(q.evaluate(free, {'status': 200, '_content': 'x', 'finish_reason': 'stop',
                                           'usage': {'completion_tokens': free['max_tokens']+1}})['functional_pass'])
        self.assertFalse(q.evaluate(free, {'status': 200, '_content': 'x', 'finish_reason': None,
                                           'usage': {'completion_tokens': 5}})['functional_pass'])


class Summary(unittest.TestCase):
    def test_screen_and_acceptance_paths(self):
        (decision, reasons), summary = verdict(*dataset(1), 1)
        self.assertEqual((decision, reasons), ('SCREEN_PASS', []))
        self.assertFalse(summary['speed_gate_pass'])
        (decision, reasons), summary = verdict(*dataset(5), 5)
        self.assertEqual((decision, reasons), ('ACCEPTANCE_GATES_PASS', []))
        self.assertIs(summary['qualified'], False)
        self.assertEqual(summary['reference_arm'], 'overlap')
        self.assertEqual(summary['lookup_vs_overlap_by_case']['copy-code']['valid_pairs'], 5)

    def test_changed_output_counts_cache_source_or_metrics_fail(self):
        rows, runs = dataset(5)
        for key, value in [('content_sha256', 'f'*64), ('reasoning_sha256', 'e'*64), ('finish_reason', 'length'),
                           ('artifact_sha256', 'd'*64), ('source_sha256', 'd'*64), ('engine_metrics_joined', False),
                           ('cached_prompt_tokens', 10), ('cached_prompt_tokens', None),
                           ('usage', {'prompt_tokens': 100, 'completion_tokens': 21, 'total_tokens': 121}),
                           ('usage', {'completion_tokens': 20})]:
            for case in ('copy-code', 'free-code', 'sampled-default'):
                changed = copy.deepcopy(rows)
                find(changed, 3, 'lookup', case)[key] = value
                (decision, _), summary = verdict(changed, runs, 5)
                self.assertEqual(decision, 'OFF', (key, case))
                self.assertFalse(summary['gates_pass'], (key, case))
        changed = copy.deepcopy(rows)
        find(changed, 0, 'lookup', 'free-code')['content_sha256'] = 'f'*64
        (_, reasons), _ = verdict(changed, runs, 5)
        self.assertIn('lookup_output_differs_from_overlap:free-code', reasons)

    def test_empty_missing_duplicate_and_receipt_errors_never_pass(self):
        rows, runs = dataset(5)
        self.assertEqual(verdict([], [], 5)[0][0], 'OFF')
        self.assertEqual(verdict([], [], 1)[0][0], 'OFF')
        mutations = []
        mutations.append(([r for r in rows if (r['repeat'], r['arm'], r['case']) != (4, 'overlap', 'quote-ru')], runs))
        mutations.append(([r for r in rows if not (r['repeat'] == 2 and r['arm'] == 'nooverlap')], runs))
        mutations.append((rows, [r for r in runs if (r['repeat'], r['arm']) != (1, 'lookup')]))
        mutations.append((rows+[copy.deepcopy(rows[0])], runs))
        mutations.append((rows, runs+[copy.deepcopy(runs[0])]))
        bad_cfg = copy.deepcopy(runs); bad_cfg[4]['config_sha256'] = CFG['overlap'] if bad_cfg[4]['arm'] != 'overlap' else CFG['lookup']
        mutations.append((rows, bad_cfg))
        bad_files = copy.deepcopy(runs); bad_files[7]['lookup_source_files']['scheduler/speculative.py'] = 'd'*64
        mutations.append((rows, bad_files))
        missing_files = copy.deepcopy(runs); del missing_files[2]['lookup_source_files']
        mutations.append((rows, missing_files))
        bad_src = copy.deepcopy(runs); bad_src[9]['source_sha256'] = 'd'*64
        mutations.append((rows, bad_src))
        unexpected = copy.deepcopy(rows); unexpected.append(dict(copy.deepcopy(rows[0]), arm='mtp'))
        mutations.append((unexpected, runs))
        for index, (changed_rows, changed_runs) in enumerate(mutations):
            (decision, _), summary = verdict(changed_rows, changed_runs, 5)
            self.assertEqual(decision, 'OFF', index)
            self.assertFalse(summary['gates_pass'], index)

    def test_speculation_scope_and_exercise(self):
        rows, runs = dataset(5)
        cases = []
        reference = copy.deepcopy(rows); find(reference, 1, 'overlap', 'copy-code')['speculation'] = dict(SPEC_OK)
        cases.append((reference, 'speculation_in_overlap_arm:copy-code'))
        diagnostic = copy.deepcopy(rows); find(diagnostic, 1, 'nooverlap', 'quote-ru')['speculation'] = dict(SPEC_OK)
        cases.append((diagnostic, 'speculation_in_nooverlap_arm:quote-ru'))
        sampled = copy.deepcopy(rows); find(sampled, 0, 'lookup', 'sampled-default')['speculation'] = dict(SPEC_OK)
        cases.append((sampled, 'speculation_on_sampled_request:sampled-default'))
        idle = copy.deepcopy(rows); find(idle, 4, 'lookup', 'copy-code-long')['speculation'] = {}
        cases.append((idle, 'lookup_not_exercised_in_every_repetition_row'))
        flags = copy.deepcopy(rows); find(flags, 2, 'lookup', 'quote-ru')['speculation']['verify_replay_in_engine_gpu_timers'] = True
        cases.append((flags, 'speculation_timing_scope_unconfirmed'))
        malformed = copy.deepcopy(rows); find(malformed, 2, 'overlap', 'quote-ru')['speculation'] = {'attempts': 'x'}
        cases.append((malformed, 'speculation_in_overlap_arm:quote-ru'))
        for changed, reason in cases:
            (decision, reasons), _ = verdict(changed, runs, 5)
            self.assertEqual(decision, 'OFF', reason)
            self.assertIn(reason, reasons)

    def test_acceptance_reference_is_overlap_not_nooverlap(self):
        def slower_than_overlap(arm, case, repeat):
            if arm == 'nooverlap':
                return 1.30+.001*repeat
            if arm == 'lookup':
                return (1.10 if case['case'] in q.SPEED_WIN_CASES else 1.25)+.002*repeat
            return 1.0+.001*repeat
        (decision, reasons), summary = verdict(*dataset(5, slower_than_overlap), 5)
        self.assertEqual((decision, reasons), ('OFF', ['speed_gate_vs_overlap_failed']))
        self.assertGreater(summary['diagnostic_by_case']['copy-code']['lookup_vs_nooverlap']['geometric_mean_speedup'], 1.1)
        self.assertLess(summary['lookup_vs_overlap_by_case']['copy-code']['geometric_mean_speedup'], 1)
        (decision, reasons), _ = verdict(*dataset(1, slower_than_overlap), 1)
        self.assertEqual((decision, reasons), ('OFF', ['screen_no_speed_signal_vs_overlap']))

    def test_short_or_sampled_case_regression_vs_overlap_blocks(self):
        for victim in ('eos-short', 'sampled-default', 'copy-code-long'):
            def regress(arm, case, repeat, victim=victim):
                if arm == 'lookup' and case['case'] == victim:
                    return 1.10+.001*repeat
                return lookup_fast(arm, case, repeat)
            (decision, reasons), summary = verdict(*dataset(5, regress), 5)
            self.assertEqual((decision, reasons), ('OFF', ['speed_gate_vs_overlap_failed']), victim)
            self.assertTrue(summary['speed_win_vs_overlap'])
            self.assertFalse(summary['no_regression_vs_overlap'])

    def test_fewer_than_five_pairs_never_accept(self):
        for pairs in (1, 2, 3, 4):
            (decision, _), summary = verdict(*dataset(pairs), pairs)
            self.assertEqual(decision, 'SCREEN_PASS', pairs)
            self.assertFalse(summary['speed_gate_pass'], pairs)
        rows, runs = dataset(5)
        (decision, _), _ = verdict(rows, runs, 5)
        self.assertEqual(decision, 'ACCEPTANCE_GATES_PASS')
        # Five requested pairs, only four present: never acceptance.
        four = [r for r in rows if r['repeat'] < 4]
        self.assertEqual(verdict(four, [r for r in runs if r['repeat'] < 4], 5)[0][0], 'OFF')

    def test_state_and_cancellation_are_required(self):
        rows, runs = dataset(5)
        for state in ('error', 'screen_failed', 'running', 'interrupted', 'prepared_only'):
            (decision, reasons), _ = verdict(rows, runs, 5, state=state)
            self.assertEqual(decision, 'OFF')
            self.assertIn('run_state_'+state, reasons)
        for cancellation in ([], None, [{'arm': 'overlap', 'pass': True}], [{'arm': 'lookup', 'pass': False}],
                             [{'arm': 'lookup', 'pass': 'yes'}]):
            (decision, reasons), _ = verdict(rows, runs, 5, cancellation=cancellation)
            self.assertEqual(decision, 'OFF')
            self.assertIn('lookup_cancellation_probe_failed_or_missing', reasons)

    def test_reference_failure_is_not_called_candidate_regression(self):
        rows, runs = dataset(1)
        find(rows, 0, 'overlap', 'eos-short')['functional_pass'] = False
        (decision, reasons), _ = verdict(rows, runs, 1)
        self.assertEqual(decision, 'OFF')
        self.assertIn('reference_functional_fail:eos-short', reasons)
        self.assertFalse(any(reason.startswith('lookup_functional_fail') for reason in reasons))

    def test_summary_is_json_safe(self):
        _, summary = verdict(*dataset(5), 5)
        json.dumps(summary, allow_nan=False)


class Owner(unittest.TestCase):
    def test_busy_or_unknown_service_refused_without_mutation(self):
        owner = q.ServiceOwner()
        for value, code in (('active', 'a_model_service_is_busy'), ('activating', 'a_model_service_is_busy'),
                            (None, 'service_state_unknown')):
            with patch.object(owner, 'state', return_value=value), patch.object(owner, 'command') as command:
                with self.assertRaises(RuntimeError) as raised:
                    owner.start('lookupq-new-overlap')
                self.assertEqual(str(raised.exception), code)
                command.assert_not_called()

    def test_stop_requires_matching_profile_and_invocation(self):
        owner = q.ServiceOwner(); owner.profile = 'lookupq-own'; owner.invocation = 'expected'
        values = {'ActiveState': 'active', 'ExecStart': 'run-experiment.sh unrelated', 'InvocationID': 'different'}
        with patch.object(owner, 'state', side_effect=lambda unit, key: values[key]), patch.object(owner, 'command') as command:
            with self.assertRaises(RuntimeError):
                owner.stop()
            command.assert_not_called()
        values.update(ExecStart='run-experiment.sh lookupq-own', InvocationID='expected')
        with patch.object(owner, 'state', side_effect=lambda unit, key: values[key]), patch.object(owner, 'command') as command:
            owner.stop()
            command.assert_called_once_with(['systemctl', '--user', 'stop', q.UNIT])
        owner.profile = 'lookupq-own'
        with patch.object(owner, 'state', return_value=None), patch.object(owner, 'command') as command:
            with self.assertRaises(RuntimeError):
                owner.stop()
            command.assert_not_called()


class Paths(unittest.TestCase):
    def test_output_path_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); workspace = tmp/'ws'; root = tmp/'root'
            (workspace/'lookup-runs').mkdir(parents=True); (root/'config').mkdir(parents=True)
            self.assertEqual(q.output_path_problem(workspace/'x.json', workspace, root),
                             'output_in_directory_copied_to_native_config_use_subdirectory')
            self.assertEqual(q.output_path_problem(HERE/'x.json', workspace, root),
                             'output_in_directory_copied_to_native_config_use_subdirectory')
            self.assertEqual(q.output_path_problem(root/'config/x.json', workspace, root), 'output_inside_private_config')
            self.assertEqual(q.output_path_problem(workspace/'lookup-runs/x.txt', workspace, root), 'output_must_be_json')
            self.assertIsNone(q.output_path_problem(workspace/'lookup-runs/x.json', workspace, root))
            (workspace/'lookup-runs/old.json').write_text('{}')
            self.assertEqual(q.output_path_problem(workspace/'lookup-runs/old.json', workspace, root), 'output_exists_use_new_name')

    def test_harness_does_not_import_bench_or_tokenizer(self):
        tree = ast.parse((HERE/'lookup-qualification.py').read_text(encoding='utf-8'))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split('.')[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split('.')[0])
        self.assertFalse(imported & {'bench', 'transformers', 'torch', 'freetoken'}, imported)


class Metrics(unittest.TestCase):
    def write_logs(self, root, records, engines, engine_name='engine.jsonl'):
        (root/'logs').mkdir(parents=True, exist_ok=True); (root/'reports').mkdir(parents=True, exist_ok=True)
        (root/'logs/requests.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n', encoding='utf-8')
        (root/'reports'/engine_name).write_text('not json\n'+'\n'.join(json.dumps(r) for r in engines)+'\n', encoding='utf-8')

    def test_join_requires_current_run(self):
        run = 'f'*32
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_logs(root, [{'request_id': 'r1', 'engine_uid': 7, 'run_id': run, 'cached_prompt_tokens': 12,
                                    'peaks': {'gpu_used_bytes': 5}, 'decode_ack_tps': 80.5}],
                            [{'uid': 7, 'run_id': 'e'*32, 'pid': 1}, {'uid': 7, 'run_id': run, 'pid': 2, 'decode_gpu_ms': 3.0,
                                                                     'speculation': dict(SPEC_OK)}])
            result = {'request_id': 'r1'}
            engine = q.join_engine_metrics(result, run, root, attempts=1, delay=0)
            self.assertEqual(result['engine_metrics_status'], 'joined')
            self.assertEqual(result['cached_prompt_tokens'], 12)
            self.assertNotIn('request_id', result)
            self.assertFalse({'uid', 'pid', 'run_id'} & set(engine))
            self.assertEqual(engine['speculation']['attempts'], 3)
            other = {'request_id': 'r1'}
            self.assertEqual(q.join_engine_metrics(other, 'd'*32, root, attempts=1, delay=0), {})
            self.assertEqual(other['engine_metrics_status'], 'run_id_mismatch')
            missing = {'request_id': 'nope'}
            q.join_engine_metrics(missing, run, root, attempts=1, delay=0)
            self.assertEqual(missing['engine_metrics_status'], 'request_row_timeout')
            none = {}
            q.join_engine_metrics(none, run, root, attempts=1, delay=0)
            self.assertEqual(none['engine_metrics_status'], 'no_request_id')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_logs(root, [{'request_id': 'r2', 'engine_uid': 9, 'run_id': run}],
                            [{'uid': 9, 'run_id': run}], engine_name='engine.jsonl.1')
            rotated = {'request_id': 'r2'}
            q.join_engine_metrics(rotated, run, root, attempts=1, delay=0)
            self.assertEqual(rotated['engine_metrics_status'], 'joined')


class FakeResponse:
    def __init__(self, lines, rid, status=200, headers=None):
        self.lines, self.status_code, self.closed = lines, status, False
        self.headers = {'x-qwen36-request-id': rid, **(headers or {})}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def iter_lines(self, chunk_size=1):
        yield from self.lines

    def json(self):
        return {}


def sse(text_chunks, usage, finish='stop'):
    lines = [b'data: '+json.dumps({'choices': [{'delta': {'content': chunk}}]}).encode() for chunk in text_chunks]
    lines.append(b'data: '+json.dumps({'choices': [{'delta': {}, 'finish_reason': finish}]}).encode())
    lines.append(b'data: '+json.dumps({'choices': [], 'usage': usage}).encode())
    lines.append(b'data: [DONE]')
    return lines


class FakeClock:
    """Virtual monotonic/perf_counter/sleep, patched in as the harness `time` module."""
    def __init__(self, start=5000.0):
        self.now, self.sleeps = start, []

    def monotonic(self):
        return self.now

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        assert seconds >= 0, seconds
        self.sleeps.append(seconds)
        self.now += seconds


class JsonResponse:
    def __init__(self, status, body, headers=None):
        self.status_code, self.body, self.headers = status, body, dict(headers or {})

    def json(self):
        return self.body


class GuardedFakeServer:
    """guard.APIGuard admission in front of an early-bound FreeToken server.

    Mirrors guard.py: one bucket of 60 tokens refilled at 1 token/s; every authenticated
    route (GET /health included) needs bucket >= 1 and costs one token; otherwise
    429 'Request rate exceeded' with Retry-After: 2. uvicorn answers /health from t0
    ('loading') and reports 'ok' after load_s (~86-203 s in native load_ready_s history).
    """
    REQUEST_S = .004

    def __init__(self, clock, load_s, rid='req-warm', bucket=60.0, refill=1.0, fail_status=None):
        self.clock, self.ready_at, self.rid = clock, clock.now+load_s, rid
        self.bucket, self.refilled, self.refill, self.fail_status = bucket, clock.now, refill, fail_status
        self.log = []  # (time, route, admitted status)

    def admit(self, route):
        now = self.clock.now
        self.bucket = min(60.0, self.bucket+(now-self.refilled)*self.refill)
        self.refilled = now
        admitted = self.bucket >= 1
        if admitted:
            self.bucket -= 1
        self.log.append((now, route, 200 if admitted else 429))
        self.clock.now += self.REQUEST_S
        return admitted

    def get(self, url, headers=None, timeout=None):
        assert url.endswith('/health') and headers['Authorization'].startswith('Bearer ')
        if not self.admit('health'):
            return JsonResponse(429, {'error': {'message': 'Request rate exceeded'}}, {'Retry-After': '2'})
        return JsonResponse(200, {'status': 'ok' if self.clock.now >= self.ready_at else 'loading'})

    def post(self, url, headers=None, json=None, stream=False, timeout=None):
        assert headers['Authorization'].startswith('Bearer ')
        if not self.admit('chat'):
            return FakeResponse([], self.rid, status=429, headers={'Retry-After': '2'})
        if self.fail_status:
            return FakeResponse([], self.rid, status=self.fail_status)
        text = ['8'] if '3 + 5' in json['messages'][-1]['content'] else ['4', '2']
        return FakeResponse(sse(text, {'prompt_tokens': 20, 'completion_tokens': len(text),
                                       'total_tokens': 20+len(text)}), self.rid)

    def module(self):
        return types.SimpleNamespace(get=self.get, post=self.post, RequestException=Exception)

    def rejected(self, route=None):
        return [entry for entry in self.log if entry[2] == 429 and route in (None, entry[1])]


class RateGuardAdmission(unittest.TestCase):
    """The live run must survive APIGuard's rate bucket after a ~100-200 s model load."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        (self.root/'config').mkdir()
        (self.root/'config/experiment.env').write_text('QWEN36_API_KEY=cpu-test-placeholder\n')
        self.run, self.profile, self.config = 'ab'*16, 'lookupq-cpu-0-lookup', {'context': 1, 'env': {}}
        (self.root/'active-run.json').write_text(json.dumps({'profile': self.profile, 'configuration': self.config,
                                                             'run_id': self.run}))
        Metrics.write_logs(self, self.root, [{'request_id': 'req-warm', 'engine_uid': 1, 'run_id': self.run,
                                              'cached_prompt_tokens': 0}], [{'uid': 1, 'run_id': self.run}])
        self.owner = types.SimpleNamespace(root=self.root, state=lambda unit, prop: 'active')

    def tearDown(self):
        self.tmp.cleanup()

    def patched(self, clock, server):
        stack = contextlib.ExitStack()
        for item in (patch.object(q, 'ROOT', self.root), patch.object(q, 'time', clock),
                     patch.dict(sys.modules, {'requests': server.module()})):
            stack.enter_context(item)
        return stack

    def test_retry_after_parsing_is_bounded(self):
        for headers, expected in (({'Retry-After': '2'}, 2.0), ({'retry-after': '3'}, 3.0), ({'retry-after': '100'}, 5.0),
                                  ({'retry-after': '0'}, 1.0), ({'retry-after': 'nan'}, 2.0), ({'retry-after': 'inf'}, 2.0),
                                  ({'retry-after': 'Wed, 21 Oct 2026 07:28:00 GMT'}, 2.0), ({}, 2.0), (None, 2.0)):
            self.assertEqual(q.retry_after_s(headers), expected, headers)

    def test_one_hz_readiness_keeps_bucket_and_first_request_is_admitted(self):
        for load_s in (100.0, 137.0, 203.0):
            clock = FakeClock(); server = GuardedFakeServer(clock, load_s)
            with self.patched(clock, server):
                receipt = q.wait_ready(self.owner, self.config, self.profile, timeout=300)
                bucket_at_ready = server.bucket
                row = q.call_model_admitted(copy.deepcopy(q.WARMUP), self.run)
            self.assertEqual(receipt['run_id'], self.run)
            self.assertTrue(row['functional_pass'], row)
            self.assertEqual(row['admission_retries'], 0)
            self.assertEqual(server.rejected(), [], load_s)
            self.assertGreater(bucket_at_ready, 50, load_s)
            health = [when for when, route, _ in server.log if route == 'health']
            self.assertGreaterEqual(len(health), int(load_s))
            self.assertGreaterEqual(min(b-a for a, b in zip(health, health[1:])), q.HEALTH_POLL_INTERVAL_S-1e-6)

    def test_health_429_waits_retry_after(self):
        clock = FakeClock(); server = GuardedFakeServer(clock, 10.0, bucket=0.5)
        with self.patched(clock, server):
            q.wait_ready(self.owner, self.config, self.profile, timeout=60)
        health = [(when, status) for when, route, status in server.log if route == 'health']
        self.assertEqual(health[0][1], 429)
        self.assertGreaterEqual(health[1][0]-health[0][0], 2.0-1e-6)

    def test_pre_fix_half_second_polling_reproduces_429_and_admission_retry_recovers(self):
        clock = FakeClock(); server = GuardedFakeServer(clock, 100.0)
        auth = {'Authorization': 'Bearer cpu-test-placeholder'}
        # Pre-fix wait_ready: /health every 0.5 s, 429 not handled, warmup right after ready.
        while True:
            response = server.get(q.URL+'/health', headers=auth)
            if response.status_code == 200 and response.json()['status'] == 'ok':
                break
            clock.sleep(.5)
        self.assertTrue(server.rejected('health'))
        with self.patched(clock, server):
            direct = q.call_model(copy.deepcopy(q.WARMUP), self.run)
            row = q.call_model_admitted(copy.deepcopy(q.WARMUP), self.run)
        self.assertEqual(direct['status'], 429)  # the reviewed failure: warmup rejected, run would be OFF
        self.assertFalse(direct['functional_pass'])
        self.assertEqual(direct['retry_after_s'], 2.0)
        self.assertTrue(row['functional_pass'], row)
        self.assertGreaterEqual(row['admission_retries'], 1)

    def test_admission_retry_is_bounded_and_fails_closed(self):
        clock = FakeClock(); server = GuardedFakeServer(clock, 0.0, bucket=0.0, refill=0.0)
        with self.patched(clock, server):
            row = q.call_model_admitted(q.fixture('eos-short'), self.run)
        self.assertEqual(row['status'], 429)
        self.assertFalse(row['functional_pass'])
        self.assertEqual(row['admission_retries'], q.ADMISSION_RETRY_ATTEMPTS-1)
        self.assertEqual(len(server.rejected('chat')), q.ADMISSION_RETRY_ATTEMPTS)
        self.assertLessEqual(sum(clock.sleeps), q.ADMISSION_RETRY_ATTEMPTS*q.ADMISSION_RETRY_MAX_S)
        clock = FakeClock(); server = GuardedFakeServer(clock, 0.0, fail_status=503)
        with self.patched(clock, server):
            row = q.call_model_admitted(q.fixture('eos-short'), self.run)
        self.assertEqual((row['status'], row['admission_retries'], row['functional_pass']), (503, 0, False))
        self.assertEqual(len([entry for entry in server.log if entry[1] == 'chat']), 1)  # non-429 never retried


@unittest.skipUnless(os.name == 'posix', 'main --run is WSL/posix only')
class MainRunWithFakes(unittest.TestCase):
    """End-to-end --run loop with every external effect faked (no systemd, HTTP, GPU)."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); tmp = Path(self.tmp.name)
        self.root, self.workspace = tmp/'root', tmp/'ws'
        (self.root/'config').mkdir(parents=True); (self.workspace/'lookup-runs').mkdir(parents=True)
        base = {'context': 131072, 'chunk': 2048, 'memory_ratio': .75, 'flags': [], 'env': {'QWEN36_FUSED_ATTN_GATE': '1'}}
        (self.root/'config/main-final.json').write_text(json.dumps(base))
        self.clock, self.lock_path = FakeClock(), tmp/'gpu.lock'
        self.calls, self.quote_seen, self.reject_left, self.warmup_rejections = [], set(), 0, 2
        self.arm = self.repeat = None
        harness = self

        class Owner:
            def __init__(self, root=None):
                self.root = harness.root

            def require_idle(self):
                pass

            def start(self, profile):
                _, _, repeat, arm = profile.split('-')
                harness.repeat, harness.arm = int(repeat), arm
                harness.reject_left = harness.warmup_rejections  # drained bucket right after a load

            def stop(self):
                pass

        def wait_ready(owner, config, profile, timeout=300):
            return {'profile': profile, 'configuration': config, 'run_id': 'ab'*16, 'source_sha256': SRC,
                    'source_files': dict(LOOKUP_FILES), 'packages': {'torch': 'x'}, 'checkpoint_revision': 'rev',
                    'service_source_files': {}, 'native_libraries': {}}

        def call_model(case, run_id):
            harness.clock.now += .01  # request latency on the fake clock (not a harness sleep)
            first_quote = case['case'] == 'quote-ru' and (harness.repeat, harness.arm) not in harness.quote_seen
            if case['case'] == 'quote-ru':
                harness.quote_seen.add((harness.repeat, harness.arm))
            if harness.reject_left > 0 or first_quote:
                harness.reject_left = max(0, harness.reject_left-1)
                harness.calls.append((harness.arm, case['case'], 429))
                return q.evaluate(case, {'status': 429, 'retry_after_s': 2.0, 'elapsed_s': .001,
                                         'engine_metrics_joined': False})
            harness.calls.append((harness.arm, case['case'], 200))
            if case['case'] == 'warmup':
                return q.evaluate(case, {'status': 200, 'usage': {'prompt_tokens': 20, 'completion_tokens': 1,
                                                                  'total_tokens': 21},
                                         'finish_reason': 'stop', '_content': '8', '_reasoning': '', 'elapsed_s': .2,
                                         'engine_metrics_joined': True})
            row = make_row(case, harness.repeat, harness.arm, lookup_fast(harness.arm, case, harness.repeat))
            for key in ('repeat', 'arm', 'source_sha256', 'artifact_sha256'):
                row.pop(key)
            return row

        def lock(path=None):
            return os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600), 'acquired'
        self.patches = [patch.object(q, 'ROOT', self.root), patch.object(q, 'WORKSPACE_OPT', self.workspace),
                        patch.object(q, 'preflight', return_value={'ok': True, 'checks': {}, 'info': {}}),
                        patch.object(q, 'acquire_gpu_lock', side_effect=lock), patch.object(q, 'ServiceOwner', Owner),
                        patch.object(q, 'wait_ready', side_effect=wait_ready), patch.object(q, 'call_model', side_effect=call_model),
                        patch.object(q, 'cancellation_probe', side_effect=lambda arm, run_id: {'arm': arm, 'pass': True}),
                        patch.object(q, 'time', self.clock)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def run_screen(self, name):
        output = self.workspace/'lookup-runs'/name
        with contextlib.redirect_stdout(io.StringIO()):
            code = q.main(['--base-profile', 'main-final', '--pairs', '1', '--output', str(output), '--run'])
        return code, json.loads(output.read_text(encoding='utf-8'))

    def test_rate_limited_first_requests_are_retried_not_turned_into_off(self):
        code, report = self.run_screen('screen.json')
        self.assertEqual((code, report['state'], report['decision']), (0, 'complete', 'SCREEN_PASS'), report.get('error_code'))
        self.assertEqual([run['warmup_admission_retries'] for run in report['runs']], [2, 2, 2])
        retries = {(row['arm'], row['case']): row['admission_retries'] for row in report['rows']}
        self.assertEqual(len(retries), len(q.ARMS)*len(CASES))
        self.assertTrue(all(value == (1 if case == 'quote-ru' else 0) for (_, case), value in retries.items()), retries)
        self.assertTrue(all(row['status'] == 200 for row in report['rows']))
        self.assertEqual(len([call for call in self.calls if call[2] == 429]), len(q.ARMS)*3)
        self.assertEqual(self.clock.sleeps.count(2.0), len(q.ARMS)*3)  # Retry-After honoured per rejection

    def test_persistent_rate_limit_fails_closed_with_explicit_code(self):
        self.warmup_rejections = 10**6
        code, report = self.run_screen('screen-limited.json')
        self.assertEqual((code, report['state'], report['decision']), (1, 'error', 'OFF'))
        self.assertEqual(report['error_code'], 'warmup_rate_limited')
        self.assertEqual(report['rows'], [])
        self.assertEqual(len(self.calls), q.ADMISSION_RETRY_ATTEMPTS)


class LiveContractWithFakeHttp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        (self.root/'config').mkdir()
        (self.root/'config/experiment.env').write_text('QWEN36_API_KEY=cpu-test-placeholder\n')
        self.run = 'a1'*16
        records = [{'request_id': rid, 'engine_uid': uid, 'run_id': self.run, 'cached_prompt_tokens': 0}
                   for rid, uid in (('req-eos', 1), ('req-copy', 2))]
        engines = [{'uid': 1, 'run_id': self.run}, {'uid': 2, 'run_id': self.run, 'cancelled': True,
                                                     'speculation': dict(SPEC_OK)}]
        Metrics.write_logs(self, self.root, records, engines)
        self.calls = []

        def post(url, headers=None, json=None, stream=False, timeout=None):
            self.calls.append(json)
            if 'Сколько будет 17' in json['messages'][-1]['content']:
                return FakeResponse(sse(['4', '2'], {'prompt_tokens': 30, 'completion_tokens': 2, 'total_tokens': 32}), 'req-eos')
            return FakeResponse(sse(['x']*60, {'prompt_tokens': 300, 'completion_tokens': 60, 'total_tokens': 360},
                                    finish='length'), 'req-copy')
        self.fake = types.SimpleNamespace(post=post, RequestException=Exception, get=None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_call_model_parses_stream_and_keeps_report_private(self):
        with patch.object(q, 'ROOT', self.root), patch.dict(sys.modules, {'requests': self.fake}):
            row = q.call_model(q.fixture('eos-short'), self.run)
        self.assertTrue(row['functional_pass'])
        self.assertTrue(row['engine_metrics_joined'])
        self.assertEqual(row['usage'], {'prompt_tokens': 30, 'completion_tokens': 2, 'total_tokens': 32})
        self.assertEqual(row['content_sha256'], q.digest('42'.encode()))
        text = json.dumps(row)
        self.assertNotIn(self.run, text)
        self.assertNotIn('req-eos', text)
        self.assertNotIn('cpu-test-placeholder', text)
        self.assertFalse({'request_id', 'serving_run_id', '_content', '_reasoning'} & set(row))
        self.assertEqual(self.calls[0]['temperature'], 0)
        self.assertEqual(self.calls[0]['top_p'], 1)

    def test_cancellation_probe_requires_followup_and_reports_speculation(self):
        with patch.object(q, 'ROOT', self.root), patch.dict(sys.modules, {'requests': self.fake}):
            probe = q.cancellation_probe('lookup', self.run)
            self.assertTrue(probe['pass'])
            self.assertIs(probe['speculation_active_at_cancel'], True)
            self.assertIs(probe['engine_marked_cancelled'], True)
            self.assertGreaterEqual(probe['content_chunks_before_cancel'], q.CANCEL_MIN_CHUNKS)
            short = q.cancellation_probe('lookup', self.run, min_chunks=100)
            self.assertFalse(short['pass'])
        json.dumps(probe, allow_nan=False)

    def test_cancellation_stream_admission_429_is_retried(self):
        clock, rejected = FakeClock(), []
        original = self.fake.post

        def post(url, headers=None, json=None, stream=False, timeout=None):
            clock.now += .01  # request latency on the fake clock (not a harness sleep)
            if not rejected:
                rejected.append(True)
                return FakeResponse([], 'req-rejected', status=429, headers={'Retry-After': '2'})
            return original(url, headers=headers, json=json, stream=stream, timeout=timeout)
        fake = types.SimpleNamespace(post=post, RequestException=Exception, get=None)
        with patch.object(q, 'ROOT', self.root), patch.object(q, 'time', clock), patch.dict(sys.modules, {'requests': fake}):
            probe = q.cancellation_probe('lookup', self.run)
        self.assertTrue(probe['pass'], probe)
        self.assertEqual(probe['stream_admission_retries'], 1)
        self.assertEqual(probe['stream_status'], 200)
        self.assertEqual(clock.sleeps[0], 2.0)
        self.assertIs(probe['speculation_active_at_cancel'], True)


class MainEntry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); tmp = Path(self.tmp.name)
        self.root, self.workspace = tmp/'root', tmp/'ws'
        (self.root/'config').mkdir(parents=True); (self.workspace/'lookup-runs').mkdir(parents=True)
        self.base = {'context': 131072, 'chunk': 2048, 'memory_ratio': .75, 'flags': [], 'env': {'QWEN36_FUSED_ATTN_GATE': '1'}}
        (self.root/'config/main-final.json').write_text(json.dumps(self.base))
        self.patches = [patch.object(q, 'ROOT', self.root), patch.object(q, 'WORKSPACE_OPT', self.workspace),
                        patch.object(q, 'port_free', return_value=True),
                        patch.object(q.ServiceOwner, 'state', return_value='inactive'),
                        patch.object(q.ServiceOwner, 'command', side_effect=AssertionError('no subprocess')),
                        patch.object(q.ServiceOwner, 'start', side_effect=AssertionError('no model start')),
                        patch.object(q, 'acquire_gpu_lock', side_effect=AssertionError('no GPU lock'))]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def sync_start_script(self, workspace_line):
        script = f"#!/usr/bin/env bash\nworkspace='{workspace_line}'\n"
        (self.root/'start-experiment.sh').write_text(script); (self.workspace/'start-experiment.sh').write_text(script)
        (self.root/'run-experiment.sh').write_text('#'); (self.root/'experiment.py').write_text('#')

    def test_plan_mode_never_starts_and_records_hashes_only(self):
        output = self.workspace/'lookup-runs/plan.json'
        self.assertEqual(q.main(['--base-profile', 'main-final', '--pairs', '1', '--output', str(output)]), 0)
        report = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual((report['state'], report['decision'], report['qualified']), ('prepared_only', 'OFF', False))
        self.assertEqual(len(report['order']), 3)
        self.assertEqual(len(report['fixtures']), len(CASES))
        text = json.dumps(report, ensure_ascii=False)
        for case in CASES:
            self.assertNotIn(case['messages'][1]['content'][:40], text)
        self.assertFalse(report['preflight']['ok'])  # start-experiment.sh not synced in this fixture
        with self.assertRaises(SystemExit):
            q.main(['--base-profile', 'main-final', '--pairs', '1', '--output', str(output)])  # never overwrite

    def test_preflight_detects_stale_native_start_script_and_workspace_base_drift(self):
        self.sync_start_script('/mnt/c/old/workspace/that/does/not/exist')
        result = q.preflight('main-final', self.base, root=self.root, workspace_opt=self.workspace, check_port=False)
        self.assertFalse(result['checks']['start_experiment_workspace_is_current'])
        self.sync_start_script(str(self.workspace))
        result = q.preflight('main-final', self.base, root=self.root, workspace_opt=self.workspace, check_port=False)
        self.assertTrue(result['ok'], result['checks'])
        (self.workspace/'main-final.json').write_text(json.dumps(dict(self.base, chunk=4096)))
        result = q.preflight('main-final', self.base, root=self.root, workspace_opt=self.workspace, check_port=False)
        self.assertFalse(result['checks']['workspace_base_matches_native'])
        (self.root/'start-experiment.sh').write_text('different')
        result = q.preflight('main-final', self.base, root=self.root, workspace_opt=self.workspace, check_port=False)
        self.assertFalse(result['checks']['start_experiment_synced'])

    @unittest.skipUnless(os.name == 'posix', 'WSL/posix only')
    def test_run_refuses_on_failed_preflight_without_lock_or_start(self):
        output = self.workspace/'lookup-runs/refused.json'
        self.assertEqual(q.main(['--base-profile', 'main-final', '--pairs', '1', '--output', str(output), '--run']), 1)
        report = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual((report['state'], report['decision']), ('preflight_failed', 'OFF'))

    def test_multi_pair_run_requires_matching_passing_screen(self):
        plan = self.workspace/'lookup-runs/screen-plan.json'
        q.main(['--base-profile', 'main-final', '--pairs', '1', '--output', str(plan)])
        screen = json.loads(plan.read_text(encoding='utf-8'))
        screen.update(state='complete', decision='SCREEN_PASS', runs=[make_run(0, arm) for arm in q.ORDERS[0]])
        dummy = {'base_config_sha256': screen['base_config_sha256'], 'arm_config_sha256': screen['arm_config_sha256'],
                 'fixture_sha256': screen['fixture_sha256'], 'harness': screen['harness']}
        self.assertIsNone(q.screen_report_problem(screen, dummy))
        for key, value in (('decision', 'OFF'), ('state', 'error'), ('requested_pairs', 5), ('fixture_sha256', 'f'*64),
                           ('base_config_sha256', 'f'*64), ('harness', {'lookup_qualification_sha256': 'f'*64}),
                           ('runs', []), ('runs', [make_run(0, 'overlap'), dict(make_run(0, 'lookup'), artifact_sha256='d'*64)])):
            self.assertIsNotNone(q.screen_report_problem(dict(screen, **{key: value}), dummy), key)
        self.assertIsNotNone(q.screen_report_problem(None, dummy))
        output = str(self.workspace/'lookup-runs/five.json')
        with self.assertRaises(SystemExit):
            q.main(['--base-profile', 'main-final', '--pairs', '5', '--output', output, '--run'])
        bad = self.workspace/'lookup-runs/bad-screen.json'
        bad.write_text(json.dumps(dict(screen, decision='OFF')))
        with self.assertRaises(SystemExit):
            q.main(['--base-profile', 'main-final', '--pairs', '5', '--output', output, '--run', '--screen-report', str(bad)])
        self.assertFalse(Path(output).exists())
        good = self.workspace/'lookup-runs/good-screen.json'
        good.write_text(json.dumps(screen))
        if os.name == 'posix':
            # Passes the screen check, then stops at the (unsynced) preflight: nothing starts.
            self.assertEqual(q.main(['--base-profile', 'main-final', '--pairs', '5', '--output', output, '--run',
                                     '--screen-report', str(good)]), 1)
            report = json.loads(Path(output).read_text(encoding='utf-8'))
            self.assertEqual(report['state'], 'preflight_failed')
            self.assertEqual(report['screen_report_sha256'], q.sha_file(good))

    def test_bad_inputs_are_rejected_before_any_action(self):
        output = str(self.workspace/'lookup-runs/x.json')
        for argv in (['--base-profile', 'Main_Final', '--pairs', '1', '--output', output],
                     ['--base-profile', 'missing', '--pairs', '1', '--output', output],
                     ['--base-profile', 'main-final', '--pairs', '6', '--output', output],
                     ['--base-profile', 'main-final', '--output', output],
                     ['--base-profile', 'main-final', '--pairs', '1', '--output', str(self.workspace/'x.json')]):
            with self.assertRaises(SystemExit):
                q.main(argv)
        (self.root/'config/spec.json').write_text(json.dumps(dict(self.base, env={'QWEN36_SPEC': 'lookup'})))
        with self.assertRaises(SystemExit):
            q.main(['--base-profile', 'spec', '--pairs', '1', '--output', output])
        self.assertFalse(Path(output).exists())


@unittest.skipUnless(os.name == 'posix', 'flock is posix only')
class GpuLock(unittest.TestCase):
    def test_busy_lock_refuses_and_own_or_inherited_lock_is_shared(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'gpu.lock'
            holder = subprocess.Popen([sys.executable, '-c', (
                'import fcntl,os,sys,time\n'
                f'fd=os.open({str(path)!r},os.O_RDWR|os.O_CREAT,0o600)\n'
                'fcntl.flock(fd,fcntl.LOCK_EX)\nprint("locked",flush=True)\ntime.sleep(60)\n')],
                stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(holder.stdout.readline().strip(), 'locked')
                with self.assertRaises(q.GpuLockBusy):
                    q.acquire_gpu_lock(path)
            finally:
                holder.kill(); holder.wait(); holder.stdout.close()
            fd, mode = q.acquire_gpu_lock(path)
            self.assertEqual(mode, 'acquired')
            again, mode = q.acquire_gpu_lock(path)
            self.assertEqual((again, mode), (fd, 'inherited'))
            os.close(fd)
            # flock(1)-style wrapper: the child receives the locked descriptor and shares it.
            wrapper = subprocess.run([sys.executable, '-c', (
                'import fcntl,os,subprocess,sys\n'
                f'fd=os.open({str(path)!r},os.O_RDWR|os.O_CREAT,0o600)\n'
                'fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\nos.set_inheritable(fd,True)\n'
                'code="import importlib.util,sys;s=importlib.util.spec_from_file_location(\'m\',sys.argv[1]);'
                'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);print(m.acquire_gpu_lock(sys.argv[2])[1])"\n'
                'sys.exit(subprocess.run([sys.executable,"-c",code,sys.argv[1],sys.argv[2]],close_fds=False).returncode)\n'),
                str(HERE/'lookup-qualification.py'), str(path)], capture_output=True, text=True, timeout=60)
            self.assertEqual(wrapper.returncode, 0, wrapper.stderr)
            self.assertEqual(wrapper.stdout.strip(), 'inherited')


if __name__ == '__main__':
    unittest.main(verbosity=2)
