"""Synthetic SQLite/interval tests. No GPU, torch, profiler, model or API key."""
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("analysis", Path(__file__).with_name("analyze-model-nsys.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture(path, *, phases=True, copies=True):
    pid = 23 << 24
    tid = pid + 9
    with sqlite3.connect(path) as conn:
        conn.executescript('''
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,endGlobalTid INTEGER,text TEXT,textId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,contextId INTEGER,streamId INTEGER,globalPid INTEGER,correlationId INTEGER,demangledName INTEGER,graphId INTEGER,graphNodeId INTEGER);
        ''')
        conn.executemany('INSERT INTO StringIds VALUES(?,?)', [
            (1, '_prefill_nvfp4_moe_kernel'), (2, 'cudaLaunchKernel'),
            (3, 'attention_gate_fp8'), (4, 'cudaGraphLaunch'),
            (5, 'VERY_PRIVATE_KERNEL_SYMBOL_5'), (6, 'Qwen36/decode')])
        if phases:
            conn.executemany('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?,?)', [
                (0, 100, tid, None, 'Qwen36/prefill', None),
                (100, 200, tid, None, None, 6),
                (10, 90, tid, None, 'VERY_PRIVATE_PROMPT_7', None)])
        conn.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?)', [
            (10, 20, tid, 11, 2), (110, 120, tid, 12, 4),
            (210, 220, tid, 13, 2)])
        # Actual GPU work runs after the CPU NVTX ranges, and prefill/decode
        # overlaps. Proper attribution must follow launch correlation.
        conn.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?,?,?)', [
            (1000, 1200, 0, 1, 1, pid, 11, 1, None, None),
            (1100, 1300, 0, 1, 2, pid, 12, 3, 5, 6),
            (1400, 1450, 0, 1, 1, pid, 13, 5, 7, 8),
            # Same correlation ID in another process must not leak phase.
            (1500, 1550, 0, 1, 1, 24 << 24, 11, 5, None, None)])
        if copies:
            conn.executescript('''
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,deviceId INTEGER,contextId INTEGER,streamId INTEGER,globalPid INTEGER,correlationId INTEGER,bytes INTEGER,copyKind INTEGER);
            CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER,name TEXT,label TEXT);
            ''')
            conn.executemany('INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES(?,?,?)', [
                (1, 'CUDA_MEMCPY_KIND_HTOD', 'Host-to-Device'),
                (2, 'CUDA_MEMCPY_KIND_DTOH', 'Device-to-Host')])
            conn.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,?,?,?,?,?,?,?)', [
                (1150, 1250, 0, 1, 3, pid, 11, 4096, 1),
                (1500, 1600, 0, 1, 3, pid, 12, 8, 2)])


class Tests(unittest.TestCase):
    def test_interval_union_overlap_and_touching(self):
        self.assertEqual(analysis.duration([(1, 5), (2, 3), (5, 7), (9, 10)]), 7)
        self.assertEqual(analysis.intersection([(1, 5), (4, 8)], [(3, 6)]), 3)
        self.assertEqual(analysis.simultaneous_streams([
            [{'start': 1, 'end': 5}], [{'start': 5, 'end': 8}]]), (0, 1))

    def test_gaps_are_union_based(self):
        result = analysis.timeline([{'start': 0, 'end': 10}, {'start': 3, 'end': 4}, {'start': 20, 'end': 30}])
        self.assertEqual(result['envelope_ms'], 30e-6)
        self.assertEqual(result['activity_union_ms'], 20e-6)
        self.assertEqual(result['unobserved_inside_envelope_ms'], 10e-6)
        self.assertEqual(result['gap_count'], 1)

    def test_asynchronous_phase_graph_and_process_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.sqlite'
            fixture(path)
            result = analysis.analyze(path)
        self.assertEqual(result['phase_kernels']['prefill']['kernel_count'], 1)
        self.assertEqual(result['phase_kernels']['decode']['kernel_count'], 1)
        self.assertEqual(result['phase_kernels']['unknown']['kernel_count'], 2)
        self.assertEqual(result['graph_membership_kernels']['graph']['kernel_count'], 2)
        self.assertEqual(result['devices'][0]['kernel_memcpy_overlap_ms'], 150e-6)
        self.assertEqual(result['devices'][0]['peak_simultaneously_active_traced_streams'], 3)
        self.assertEqual(result['devices'][0]['two_or_more_streams_overlap_ms'], 200e-6)
        self.assertEqual(sum(row['bytes'] for row in result['memory_copies']), 4104)
        self.assertEqual(result['memory_copies'][0]['direction'], 'd2h')
        self.assertEqual(result['memory_copies'][1]['direction'], 'h2d')
        self.assertNotIn('VERY_PRIVATE', json.dumps(result))
        self.assertNotIn(str(path), json.dumps(result))

    def test_graph_without_phase_is_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.sqlite'
            fixture(path, phases=False, copies=False)
            result = analysis.analyze(path)
        self.assertEqual(result['phase_kernels']['unknown']['kernel_count'], 4)
        self.assertEqual(result['phase_kernels']['decode']['kernel_count'], 0)
        self.assertIsNone(result['memory_copies'])
        self.assertFalse(result['available_tables']['memcpy'])

    def test_multiple_devices_separate_envelopes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.sqlite'
            fixture(path, copies=False)
            with sqlite3.connect(path) as conn:
                conn.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(0,100000,1,1,1,0,999,1,NULL,NULL)')
            result = analysis.analyze(path)
        self.assertEqual(len(result['devices']), 2)
        self.assertEqual(result['devices'][0]['envelope_ms'], 550e-6)
        self.assertEqual(result['devices'][1]['envelope_ms'], .1)

    def test_phase_nesting_and_cross_thread(self):
        rows = [dict(start=0, end=20, globalTid=1, name='prefill'),
                dict(start=2, end=18, globalTid=1, name='prefill'),
                dict(start=30, end=40, globalTid=1, endGlobalTid=2, name='decode')]
        phase = analysis.PhaseIndex(rows, analysis.PHASE_LABELS)
        self.assertEqual(phase.find(dict(start=3, end=4, globalTid=1)), 'prefill')
        self.assertEqual(phase.find(dict(start=35, end=36, globalTid=1)), 'unknown')
        self.assertEqual(phase.summary()['prefill']['cpu_union_ms'], 20e-6)
        self.assertEqual(phase.summary()['prefill']['inclusive_cpu_sum_ms'], 36e-6)
        self.assertEqual(phase.cross_thread_ranges_ignored, 1)

    def test_conflicting_correlation_is_unknown(self):
        pid = 4 << 24
        phase = analysis.PhaseIndex([dict(start=0, end=10, globalTid=pid+2, name='prefill'),
                                    dict(start=11, end=20, globalTid=pid+2, name='decode')], analysis.PHASE_LABELS)
        index = analysis.CorrelationIndex([
            dict(start=1, end=2, globalTid=pid+2, correlationId=5),
            dict(start=12, end=13, globalTid=pid+2, correlationId=5)], phase)
        self.assertEqual(index.find(dict(start=25, end=30, globalPid=pid, correlationId=5)),
                         ('unknown', 'ambiguous_phase'))

    def test_incomplete_rows_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.sqlite'
            fixture(path)
            with sqlite3.connect(path) as conn:
                conn.execute('INSERT INTO NVTX_EVENTS VALUES(0,NULL,0,NULL,\'prefill\',NULL)')
                conn.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(50,10,0,0,0,0,0,1,NULL,NULL)')
            result = analysis.analyze(path)
        self.assertEqual(result['invalid_or_incomplete_rows_excluded']['nvtx'], 1)
        self.assertEqual(result['invalid_or_incomplete_rows_excluded']['kernel'], 1)

    def test_absent_input_not_created(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'absent.sqlite'
            with self.assertRaises(FileNotFoundError):
                analysis.analyze(path)
            self.assertFalse(path.exists())

    def test_input_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.sqlite'
            fixture(path)
            original = path.read_bytes()
            analysis.analyze(path)
            self.assertEqual(path.read_bytes(), original)


# ---- schema 2: forward instances, critical chain, Sampler, layers, ncu targets ----

PID = 31 << 24
TID = PID + 5
STRINGS = [
    (1, 'cudaLaunchKernel'), (2, 'cudaGraphLaunch'),
    (10, 'cutlass_MainloopSm120_e4m3_gemm_PRIVATE_SYMBOL'), (11, '_router_triton_kernel'),
    (12, 'BatchDecodeWithPagedKVCacheKernel'), (13, 'sampling::TopKTopPSamplingFromProbKernel'),
    (20, 'Qwen36/prefill'), (21, 'Qwen36/decode'), (22, 'Sampler'), (23, 'SamplerKernelDispatch'),
    (24, 'Layer_3'), (25, 'MHA'), (26, 'Layer_0'), (27, 'Layer_5'), (28, 'VERY_PRIVATE_PROMPT_TEXT'),
]


def build(path, nvtx, runtime, kernels, graph_nodes=None, memcpy=None):
    """nvtx: (start,end,textId); runtime: (start,end,corr,nameId);
    kernels: (start,end,stream,corr,nameId,graphId,graphNodeId,gridX,blockX)."""
    with sqlite3.connect(path) as conn:
        conn.executescript('''
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,eventType INTEGER,globalTid INTEGER,endGlobalTid INTEGER,text TEXT,textId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,contextId INTEGER,streamId INTEGER,globalPid INTEGER,correlationId INTEGER,demangledName INTEGER,shortName INTEGER,graphId INTEGER,graphNodeId INTEGER,gridX INTEGER,gridY INTEGER,gridZ INTEGER,blockX INTEGER,blockY INTEGER,blockZ INTEGER,staticSharedMemory INTEGER,dynamicSharedMemory INTEGER);
        ''')
        conn.executemany('INSERT INTO StringIds VALUES(?,?)', STRINGS)
        conn.executemany('INSERT INTO NVTX_EVENTS VALUES(?,?,59,?,NULL,NULL,?)',
                         [(s, e, TID, t) for s, e, t in nvtx])
        conn.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?)',
                         [(s, e, TID, c, n) for s, e, c, n in runtime])
        conn.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,0,1,?,?,?,?,?,?,?,?,1,1,?,1,1,0,0)',
                         [(s, e, st, PID, c, n, n, g, gn, gx, bx) for s, e, st, c, n, g, gn, gx, bx in kernels])
        if graph_nodes is not None:
            conn.execute('CREATE TABLE CUDA_GRAPH_NODE_EVENTS(start INTEGER,end INTEGER,eventClass INTEGER,globalTid INTEGER,nameId INTEGER,graphNodeId INTEGER,originalGraphNodeId INTEGER)')
            conn.executemany('INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES(?,?,0,?,0,?,?)',
                             [(s, s, TID, node, orig) for s, node, orig in graph_nodes])
        if memcpy is not None:
            conn.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,deviceId INTEGER,contextId INTEGER,streamId INTEGER,globalPid INTEGER,correlationId INTEGER,bytes INTEGER,copyKind INTEGER,graphNodeId INTEGER)')
            conn.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,0,1,?,?,?,?,1,NULL)',
                             [(s, e, st, PID, c, b) for s, e, st, c, b in memcpy])


def two_requests(path):
    # Request 0 (warmup): one prefill chunk + two decode steps.
    # Request 1 (measured): two prefill chunks + three decode steps; step 0 has
    # a PDL-style overlap of 10 ns between its two graph kernels.
    nvtx = [(0, 100, 20), (200, 300, 21), (400, 500, 21),
            (600, 700, 20), (800, 900, 20), (1000, 1100, 21), (1200, 1300, 21), (1400, 1500, 21)]
    runtime = [(10, 20, 1, 1), (210, 220, 2, 2), (410, 420, 3, 2), (610, 620, 4, 1), (810, 820, 5, 1),
               (1010, 1020, 6, 2), (1210, 1220, 7, 2), (1410, 1420, 8, 2)]
    kernels = [(1000, 1100, 1, 1, 10, None, None, 64, 128),
               (1200, 1300, 1, 2, 11, 9, 900, 1, 32), (1400, 1500, 1, 3, 11, 9, 900, 1, 32),
               (1600, 1800, 1, 4, 10, None, None, 64, 128), (1900, 2000, 1, 5, 10, None, None, 16, 128),
               (2100, 2150, 1, 6, 11, 9, 900, 1, 32), (2140, 2200, 1, 6, 12, 9, 901, 2, 64),
               (2300, 2400, 1, 7, 11, 9, 900, 1, 32), (2450, 2500, 1, 8, 11, 9, 900, 1, 32)]
    build(path, nvtx, runtime, kernels)


class SchemaTwoTests(unittest.TestCase):
    def analyze(self, builder, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.sqlite'
            builder(path)
            return analysis.analyze(path, **kwargs)

    def test_critical_chain_overlap_gap_and_hidden_side_stream(self):
        ops = [dict(start=0, end=10, n='A'), dict(start=8, end=20, n='B'),
               dict(start=25, end=30, n='C'), dict(start=2, end=6, n='hidden')]
        contributions, gaps, envelope = analysis.critical_chain(ops)
        got = {op['n']: ns for op, ns in contributions}
        self.assertEqual(got, {'C': 5, 'B': 10, 'A': 10})
        self.assertEqual(gaps, 5)
        self.assertEqual(envelope, 30)
        self.assertEqual(sum(got.values()) + gaps, envelope)

    def test_concurrency_shares_sum_to_union_not_kernel_sum(self):
        ops = [dict(start=0, end=10), dict(start=5, end=15), dict(start=5, end=15), dict(start=20, end=30)]
        shares = analysis.concurrency_shares(ops)
        self.assertAlmostEqual(sum(shares), analysis.duration([(o['start'], o['end']) for o in ops]))
        self.assertAlmostEqual(shares[0], 5 + 5 / 3)
        self.assertAlmostEqual(shares[3], 10)

    def test_nested_lookup_walks_to_enclosing_range(self):
        lookup = analysis.NestedLookup([(0, 100, 'outer'), (10, 20, 'inner'), (30, 40, 'second')])
        self.assertEqual(lookup.find(15), 'inner')
        self.assertEqual(lookup.find(50), 'outer')
        self.assertEqual(lookup.find(35, 41), 'outer')
        self.assertIsNone(lookup.find(150))

    def test_requests_periods_and_measured_critical_totals(self):
        result = self.analyze(two_requests, request_labels=['warmup_short', 'warm_decode'])
        self.assertEqual(result['forward_instances']['count'], 8)
        self.assertEqual(result['forward_instances']['gpu_ops_without_instance'], 0)
        requests = result['requests']
        self.assertEqual([r['label'] for r in requests], ['warmup_short', 'warm_decode'])
        self.assertEqual([r['measured'] for r in requests], [False, True])
        self.assertEqual((requests[1]['prefill_instances'], requests[1]['decode_instances']), (2, 3))
        self.assertAlmostEqual(requests[1]['prefill_gpu_window_ms'], 400e-6)
        self.assertAlmostEqual(requests[1]['decode_step_period_ms']['mean_ms'], 175e-6)
        prefill = result['critical_path']['phases']['prefill']
        decode = result['critical_path']['phases']['decode']
        # Prefill: 200 + 100 chunk chains + 100 idle between chunks = window.
        self.assertAlmostEqual(prefill['critical_total_ms'], 400e-6)
        self.assertAlmostEqual(prefill['gpu_idle_between_forwards_ms'], 100e-6)
        # Decode: last step excluded; 100 + 100 chains + 100 + 50 gaps = sum of periods.
        self.assertAlmostEqual(decode['critical_total_ms'], 350e-6)
        self.assertAlmostEqual(decode['ops_critical_ms'], 200e-6)
        self.assertEqual(decode['instances_in_chain_total'], 2)
        categories = {row['category']: row['critical_ms'] for row in decode['by_category']}
        # The overlapped 10 ns are attributed once (to the later kernel's tail only).
        self.assertAlmostEqual(categories['full_attention'], 50e-6)
        self.assertAlmostEqual(categories['moe_router_alignment_reduction'], 150e-6)
        top = result['ncu_candidates'][0]
        self.assertEqual((top['phase'], top['shape_class'], top['category']),
                         ('prefill', 'prefill_full_chunk', 'dense_sm120_fp8_gemm'))
        self.assertAlmostEqual(top['share_of_phase_critical_percent'], 50.0)
        text = json.dumps(result)
        self.assertNotIn('PRIVATE_SYMBOL', text)
        self.assertNotIn('Qwen36/', text)

    def test_run_receipt_labels_and_capture_completeness(self):
        receipt = {'requests': [{'label': 'warmup_short', 'completion_tokens': 3},
                                {'label': 'warm_decode', 'completion_tokens': 5}]}
        result = self.analyze(two_requests, run_receipt=receipt)
        first, second = result['requests']
        self.assertEqual(first['decode_instances_minus_expected'], 0)
        self.assertTrue(first['capture_complete'])
        self.assertEqual(second['decode_instances_minus_expected'], -1)
        self.assertFalse(second['capture_complete'])

    def test_unlabeled_extra_segment_is_not_measured(self):
        result = self.analyze(two_requests, request_labels=['warmup_short'])
        self.assertEqual([r['measured'] for r in result['requests']], [False, False])
        self.assertEqual(result['critical_path']['phases']['decode']['critical_total_ms'], 0)

    def test_request_label_validation(self):
        with self.assertRaises(ValueError):
            self.analyze(two_requests, request_labels=['Bad Label'])

    def sampler_fixture(self, inner_text, path):
        nvtx = [(0, 100, 21), (50, 90, 22), (55, 85, inner_text), (60, 70, 28)]
        runtime = [(10, 12, 1, 2), (61, 62, 2, 1)]
        kernels = [(1000, 1100, 1, 1, 11, 9, 900, 1, 32), (1100, 1110, 1, 2, 13, None, None, 1, 1024)]
        build(path, nvtx, runtime, kernels)

    def test_double_sampler_same_label_is_not_double_counted(self):
        result = self.analyze(lambda path: self.sampler_fixture(22, path))
        sampler = result['sampler_nvtx']
        self.assertEqual(sampler['mode'], 'nested_same_label')
        self.assertEqual(sampler['outer_call_count'], 1)
        self.assertEqual(sampler['nested_same_label_ranges_ignored'], 1)
        self.assertAlmostEqual(sampler['cpu_union_ms'], 40e-6)
        self.assertAlmostEqual(sampler['inclusive_cpu_sum_all_ranges_ms'], 70e-6)
        self.assertEqual(sampler['ranges_per_forward_instance_distribution'], {'2': 1})
        regions = {row['region'] for row in result['critical_path']['phases']['decode']['by_nvtx_region']}
        # Single decode step = last step of the request -> excluded from periods.
        self.assertEqual(regions, set())
        methods = {(row['kind'], row['method']) for row in result['layer_attribution']['method_counts']}
        self.assertIn(('kernel', 'launch_nvtx'), methods)
        self.assertNotIn('VERY_PRIVATE_PROMPT_TEXT', json.dumps(result))

    def test_sampler_dispatch_inner_range(self):
        result = self.analyze(lambda path: self.sampler_fixture(23, path))
        sampler = result['sampler_nvtx']
        self.assertEqual(sampler['mode'], 'outer_plus_dispatch')
        self.assertEqual(sampler['nested_dispatch_ranges_ignored'], 1)
        self.assertNotIn('SamplerKernelDispatch', json.dumps(result))

    def test_eager_layer_attribution_and_mha_kind(self):
        def builder(path):
            nvtx = [(0, 1000, 20), (10, 400, 26), (500, 990, 24), (600, 700, 25),
                    (2000, 3000, 20), (2010, 2400, 26), (2500, 2990, 24), (2600, 2700, 25),
                    (4000, 4100, 21)]
            runtime = [(20, 30, 1, 1), (650, 660, 2, 1), (995, 998, 3, 1),
                       (2020, 2030, 4, 1), (2650, 2660, 5, 1), (4010, 4020, 6, 2)]
            kernels = [(10000, 10100, 1, 1, 10, None, None, 8, 128), (10100, 10300, 1, 2, 12, None, None, 8, 128),
                       (10300, 10310, 1, 3, 11, None, None, 1, 32),
                       (20000, 20100, 1, 4, 10, None, None, 8, 128), (20100, 20300, 1, 5, 12, None, None, 8, 128),
                       (30000, 30010, 1, 6, 11, 9, 900, 1, 32)]
            build(path, nvtx, runtime, kernels)
        result = self.analyze(builder)
        self.assertEqual(result['layer_attribution']['mha_layers'], [3])
        prefill = result['critical_path']['phases']['prefill']
        layers = {row['layer']: row for row in prefill['by_layer']}
        self.assertEqual(layers[3]['layer_kind'], 'mha')
        self.assertEqual(layers[0]['layer_kind'], 'no_mha')
        self.assertAlmostEqual(layers[3]['critical_ms'], 400e-6)
        self.assertAlmostEqual(layers[0]['critical_ms'], 200e-6)
        kinds = {row['layer_kind']: row['critical_ms'] for row in prefill['by_layer_kind']}
        self.assertAlmostEqual(kinds['unattributed'], 10e-6)
        self.assertNotIn('Layer_', json.dumps(result))
        self.assertNotIn('"MHA"', json.dumps(result))

    def graph_fixture(self, path, capture_nvtx):
        # Original nodes 100/101 are created during capture (t=100/300); clones
        # 200/201 at instantiation (t=500). Only node 100 lies inside Layer_5.
        nvtx = [(1000, 2000, 21)] + ([(50, 200, 27)] if capture_nvtx else [])
        runtime = [(1010, 1020, 7, 2)]
        kernels = [(5000, 5100, 1, 7, 11, 9, 200, 1, 32), (5100, 5300, 1, 7, 12, 9, 201, 2, 64)]
        nodes = [(100, 100, None), (300, 101, None), (500, 200, 100), (500, 201, 101)]
        build(path, nvtx, runtime, kernels, graph_nodes=nodes)

    def test_graph_node_layer_from_capture_time_nvtx(self):
        result = self.analyze(lambda path: self.graph_fixture(path, True))
        methods = {row['method']: row['count'] for row in result['layer_attribution']['method_counts']
                   if row['kind'] == 'kernel'}
        self.assertEqual(methods.get('graph_node_capture_time_nvtx'), 1)
        self.assertEqual(methods.get('graph_node_creation_outside_recorded_nvtx'), 1)
        self.assertEqual(result['layer_attribution']['graph_node_events'], 4)

    def test_graph_node_without_capture_nvtx_stays_unattributed(self):
        result = self.analyze(lambda path: self.graph_fixture(path, False))
        methods = {row['method'] for row in result['layer_attribution']['method_counts']}
        self.assertEqual(methods, {'graph_node_creation_outside_recorded_nvtx'})

    def test_hidden_long_kernel_is_not_a_candidate(self):
        def builder(path):
            nvtx = [(0, 100, 20)]
            runtime = [(10, 20, 1, 1), (30, 40, 2, 1)]
            # Long kernel on a side stream is fully hidden by the main kernel.
            kernels = [(1000, 1100, 1, 1, 12, None, None, 4, 64), (1010, 1090, 2, 2, 10, None, None, 64, 128)]
            build(path, nvtx, runtime, kernels)
        result = self.analyze(builder)
        candidates = result['ncu_candidates']
        self.assertEqual([row['category'] for row in candidates], ['full_attention'])
        self.assertEqual(candidates[0]['shape_class'], 'prefill_last_chunk')
        self.assertAlmostEqual(candidates[0]['share_of_phase_critical_percent'], 100.0)

    def test_graph_replay_fallback_without_phase_nvtx(self):
        result = self.analyze(lambda path: build(path, [], [(10, 20, 7, 2)],
                              [(100, 200, 1, 7, 11, 9, 900, 1, 32), (190, 260, 1, 7, 12, 9, 901, 1, 32)]))
        replays = result['critical_path']['graph_replays']
        self.assertEqual(replays['replay_count'], 1)
        self.assertAlmostEqual(replays['chain_total_ms'], 160e-6)
        self.assertEqual(result['forward_instances']['count'], 0)

    def test_regex_literal_escapes_metacharacters_only(self):
        self.assertEqual(analysis.regex_literal('void k<(int)1>::run(float*, a.b)'),
                         'void k<\\(int\\)1>::run\\(float\\*, a\\.b\\)')

    @unittest.skipUnless(os.name == 'posix', 'owner-only permission semantics')
    def test_private_targets_need_owner_only_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.sqlite'
            two_requests(path)
            private = Path(folder) / 'private'
            private.mkdir(mode=0o755)
            os.chmod(private, 0o755)
            labels = ['warmup_short', 'warm_decode']
            with self.assertRaises(PermissionError):
                analysis.analyze(path, request_labels=labels, private_targets=private / 'targets.json')
            os.chmod(private, 0o700)
            result = analysis.analyze(path, request_labels=labels, private_targets=private / 'targets.json')
            self.assertTrue(result['private_targets_written'])
            self.assertNotIn('PRIVATE_SYMBOL', json.dumps(result))
            target = private / 'targets.json'
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
            payload = json.loads(target.read_text())
            self.assertEqual(payload['schema_version'], 3)
            names = {row['demangled_name'] for row in payload['candidates']}
            self.assertIn('cutlass_MainloopSm120_e4m3_gemm_PRIVATE_SYMBOL', names)
            rows = [row for row in payload['candidates'] if row['scope'] == 'all_measured']
            self.assertEqual([row['rank'] for row in rows], list(range(1, len(rows) + 1)))
            first = next(row for row in rows if row['shape_class'] == 'prefill_full_chunk')
            self.assertTrue(first['ncu_kernel_regex'].startswith('regex:^'))
            # The warmup request launched the same kernel/config once before.
            self.assertEqual(first['launch_skip_same_name_in_window'], 1)
            self.assertEqual(first['launch_skip_same_launch_config'], 1)
            self.assertEqual((first['instance_request_index'], first['instance_request_label']), (1, 'warm_decode'))
            self.assertEqual(first['instance_shape_class'], 'prefill_full_chunk')
            # The warmup's single (last) chunk shares grid/block: LaunchStats cannot tell them apart.
            self.assertFalse(first['launch_config_identifies_instance_group'])
            self.assertEqual(first['other_shape_classes_same_launch_config'], ['prefill_last_chunk'])
            self.assertEqual([(g['request_label'], g['shape_class'], g['selected']) for g in first['launch_config_groups']],
                             [('warmup_short', 'prefill_last_chunk', False), ('warm_decode', 'prefill_full_chunk', True)])
            last = next(row for row in rows if row['shape_class'] == 'prefill_last_chunk')
            self.assertEqual((last['launch_skip_same_name_in_window'], last['launch_skip_same_launch_config']), (2, 0))
            self.assertTrue(last['launch_config_identifies_instance_group'])
            self.assertIn('request:warm_decode', {row['scope'] for row in payload['candidates']})
            self.assertNotIn('request:warmup_short', {row['scope'] for row in payload['candidates']})


# ---- schema 3: per-request critical path and shape-exact Compute instances ----

def shared_config_requests(path):
    """Persistent-style kernel 10 has grid 70 for every M (review counterexample).

    req0 warmup_short: one prefill chunk + 1 decode step; req1 warm_decode: one short
    (last) chunk M~278 + 2 decode steps; req2 fresh_16k: full chunk M=chunk + last chunk
    + 2 decode steps.
    """
    nvtx = [(0, 100, 20), (200, 300, 21),
            (400, 500, 20), (600, 700, 21), (800, 900, 21),
            (1000, 1100, 20), (1200, 1300, 20), (1400, 1500, 21), (1600, 1700, 21)]
    runtime = [(10, 20, 1, 1), (210, 220, 2, 1), (410, 420, 3, 1), (610, 620, 4, 1), (810, 820, 5, 1),
               (1010, 1020, 6, 1), (1210, 1220, 7, 1), (1410, 1420, 8, 1), (1610, 1620, 9, 1)]
    kernels = [(5000, 5050, 1, 1, 10, None, None, 70, 128), (5100, 5110, 1, 2, 11, None, None, 1, 32),
               (5200, 5250, 1, 3, 10, None, None, 70, 128),
               (5300, 5310, 1, 4, 11, None, None, 1, 32), (5400, 5410, 1, 5, 11, None, None, 1, 32),
               (5500, 5900, 1, 6, 10, None, None, 70, 128),
               (6000, 6100, 1, 7, 10, None, None, 70, 128),
               (6200, 6210, 1, 8, 11, None, None, 1, 32), (6300, 6310, 1, 9, 11, None, None, 1, 32)]
    build(path, nvtx, runtime, kernels)


def decode_context_requests(path):
    """Same decode graph (same grid/block) at short and long context.

    Attention (kernel 12) takes 10 ns per step in warm_decode and 300 ns in fresh_120k.
    """
    nvtx = [(0, 100, 20), (200, 300, 21), (400, 500, 21),
            (600, 700, 20), (800, 900, 21), (1000, 1100, 21), (1200, 1300, 21),
            (1400, 1500, 20), (1600, 1700, 21), (1800, 1900, 21), (2000, 2100, 21)]
    runtime = [(10, 20, 1, 1), (210, 220, 2, 2), (410, 420, 3, 2),
               (610, 620, 4, 1), (810, 820, 5, 2), (1010, 1020, 6, 2), (1210, 1220, 7, 2),
               (1410, 1420, 8, 1), (1610, 1620, 9, 2), (1810, 1820, 10, 2), (2010, 2020, 11, 2)]
    kernels = [(1000, 1100, 1, 1, 10, None, None, 64, 128)]
    steps = {2: 1200, 3: 1400, 5: 1800, 6: 2000, 7: 2200, 9: 2600, 10: 3000, 11: 3400}
    for corr, start in steps.items():
        attention = 300 if corr >= 9 else 10
        kernels.append((start, start + 20, 1, corr, 11, 9, 900, 1, 32))
        kernels.append((start + 20, start + 20 + attention, 1, corr, 12, 9, 901, 2, 64))
    kernels += [(1600, 1700, 1, 4, 10, None, None, 64, 128), (2400, 2500, 1, 8, 10, None, None, 64, 128)]
    build(path, nvtx, runtime, kernels)


class SchemaThreeTests(unittest.TestCase):
    def private(self, builder, labels):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.sqlite'
            builder(path)
            private = Path(folder) / 'private'
            private.mkdir(mode=0o700)
            os.chmod(private, 0o700)
            result = analysis.analyze(path, request_labels=labels, private_targets=private / 'targets.json')
            return result, json.loads((private / 'targets.json').read_text())

    @unittest.skipUnless(os.name == 'posix', 'owner-only permission semantics')
    def test_full_chunk_target_is_not_the_short_chunk_with_same_launch_config(self):
        result, payload = self.private(shared_config_requests, ['warmup_short', 'warm_decode', 'fresh_16k'])
        rows = [row for row in payload['candidates'] if row['scope'] == 'all_measured' and row['phase'] == 'prefill']
        full = next(row for row in rows if row['shape_class'] == 'prefill_full_chunk')
        # Kernel 10 launches: warmup chunk, warm_decode short chunk, fresh_16k full chunk (3rd launch,
        # skip 2), fresh_16k last chunk. Schema 2 pointed at warm_decode's M~278 chunk (skip 1).
        self.assertEqual((full['instance_request_index'], full['instance_request_label']), (2, 'fresh_16k'))
        self.assertEqual(full['launch_skip_same_name_in_window'], 2)
        self.assertEqual(full['launch_skip_same_launch_config'], 2)
        self.assertEqual(full['instance_shape_class'], 'prefill_full_chunk')
        self.assertAlmostEqual(full['instance_duration_us'], .4)
        self.assertFalse(full['launch_config_identifies_instance_group'])
        self.assertEqual(full['other_shape_classes_same_launch_config'], ['prefill_last_chunk'])
        groups = {(g['request_label'], g['shape_class']): g for g in full['launch_config_groups']}
        self.assertEqual(set(groups), {('warmup_short', 'prefill_last_chunk'), ('warm_decode', 'prefill_last_chunk'),
                                       ('fresh_16k', 'prefill_full_chunk'), ('fresh_16k', 'prefill_last_chunk')})
        self.assertTrue(groups[('fresh_16k', 'prefill_full_chunk')]['selected'])
        self.assertEqual(groups[('warm_decode', 'prefill_last_chunk')]['launch_skip_same_name_in_window'], 1)
        self.assertAlmostEqual(groups[('warm_decode', 'prefill_last_chunk')]['p50_duration_us'], .05)
        # Aggregate last-chunk target: request with most critical time (fresh_16k 100 ns > warm_decode 50 ns).
        last = next(row for row in rows if row['shape_class'] == 'prefill_last_chunk')
        self.assertEqual((last['instance_request_label'], last['launch_skip_same_name_in_window']), ('fresh_16k', 3))
        public = next(row for row in result['ncu_candidates']
                      if row['phase'] == 'prefill' and row['shape_class'] == 'prefill_last_chunk')
        self.assertEqual(public['critical_ms_by_request'], {'warm_decode': 50e-6, 'fresh_16k': 100e-6})
        self.assertEqual(public['top_request_label'], 'fresh_16k')
        # Per-request scope: warm_decode's own last chunk, not fresh_16k's.
        own = next(row for row in payload['candidates'] if row['scope'] == 'request:warm_decode'
                   and row['shape_class'] == 'prefill_last_chunk')
        self.assertEqual((own['instance_request_label'], own['launch_skip_same_name_in_window']), ('warm_decode', 1))

    def test_by_request_separates_scenarios_and_sums_to_aggregate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.sqlite'
            decode_context_requests(path)
            result = analysis.analyze(path, request_labels=['warmup_short', 'warm_decode', 'fresh_120k'])
        critical = result['critical_path']
        self.assertEqual(critical['phases_scope'], 'all_measured_requests_mixed_by_forward_count')
        by_request = {entry['label']: entry for entry in critical['by_request']}
        self.assertEqual(list(by_request), ['warm_decode', 'fresh_120k'])
        short = {row['category']: row['critical_ms'] for row in by_request['warm_decode']['phases']['decode']['by_category']}
        long = {row['category']: row['critical_ms'] for row in by_request['fresh_120k']['phases']['decode']['by_category']}
        self.assertAlmostEqual(short['full_attention'], 20e-6)
        self.assertAlmostEqual(long['full_attention'], 600e-6)
        self.assertAlmostEqual(by_request['warm_decode']['phases']['decode']['critical_total_ms'], 400e-6)
        self.assertAlmostEqual(by_request['fresh_120k']['phases']['decode']['critical_total_ms'], 800e-6)
        for phase in ('prefill', 'decode'):
            self.assertAlmostEqual(critical['phases'][phase]['critical_total_ms'],
                                   sum(entry['phases'][phase]['critical_total_ms'] for entry in by_request.values()))
        self.assertAlmostEqual(by_request['fresh_120k']['request_critical_total_ms'], 900e-6)
        top_long = by_request['fresh_120k']['phases']['decode']['ncu_candidates'][0]
        self.assertEqual(top_long['category'], 'full_attention')
        self.assertAlmostEqual(top_long['share_of_request_critical_percent'], 100 * 600 / 900)
        self.assertAlmostEqual(top_long['share_of_phase_critical_percent'], 75.0)
        top = result['ncu_candidates'][0]
        self.assertEqual((top['phase'], top['category'], top['top_request_label']), ('decode', 'full_attention', 'fresh_120k'))
        self.assertEqual(top['critical_ms_by_request'], {'warm_decode': 20e-6, 'fresh_120k': 600e-6})
        text = json.dumps(result)
        self.assertNotIn('_key', text)
        self.assertNotIn('PRIVATE_SYMBOL', text)

    @unittest.skipUnless(os.name == 'posix', 'owner-only permission semantics')
    def test_decode_target_points_at_long_context_step(self):
        _, payload = self.private(decode_context_requests, ['warmup_short', 'warm_decode', 'fresh_120k'])
        row = next(row for row in payload['candidates'] if row['scope'] == 'all_measured'
                   and row['category'] == 'full_attention')
        # warmup (2) and warm_decode (3) launched the same graph node before fresh_120k's first step.
        self.assertEqual((row['instance_request_label'], row['instance_chunk_or_step_position']), ('fresh_120k', 0))
        self.assertEqual(row['launch_skip_same_name_in_window'], 5)
        self.assertEqual(row['launch_skip_same_name_same_phase'], 5)
        self.assertAlmostEqual(row['instance_duration_us'], .3)
        groups = [(g['request_label'], g['launches'], round(g['p50_duration_us'], 3), g['selected'])
                  for g in row['launch_config_groups']]
        self.assertEqual(groups, [('warmup_short', 2, .01, False), ('warm_decode', 3, .01, False),
                                  ('fresh_120k', 3, .3, True)])
        self.assertEqual(row['other_shape_classes_same_launch_config'], [])
        scoped = next(row for row in payload['candidates'] if row['scope'] == 'request:warm_decode'
                      and row['category'] == 'full_attention')
        self.assertEqual((scoped['instance_request_label'], scoped['launch_skip_same_name_in_window']), ('warm_decode', 2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
