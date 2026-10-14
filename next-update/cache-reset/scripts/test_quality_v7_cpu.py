"""CPU-only tests of the harness-v7 quality.py (patch/quality.py) with stubbed bench/fixtures/matrix/transformers.

No model, no HTTP, no systemctl: the restart, the engine's teacher.jsonl rows and active-run.json are simulated in
a temp ROOT. Also checks that the functions other harness code depends on (request, code_check, functional)
are byte-identical to the frozen quality.py.
"""
import ast, importlib.util, json, pathlib, sys, tempfile, types, unittest

HERE = pathlib.Path(__file__).resolve().parents[1]
PATCHED = HERE / 'patch/quality.py'
ORIG = HERE / 'orig/quality.py'
CASES = [f'c{i}' for i in range(19)] + ['long-120000']


class Sim:
    """Simulated experiment unit: active-run.json receipts and engine teacher rows."""
    def __init__(self, root, profile='teacher-x', lose=None, extra_on_clean=False, change_identity=False):
        self.root, self.profile = root, profile
        self.n = 0; self.calls = []; self.lose = lose or set(); self.extra_on_clean = extra_on_clean
        self.change_identity = change_identity
        (root / 'reports').mkdir(parents=True, exist_ok=True)
        self.start()

    def receipt(self):
        return json.loads((self.root / 'active-run.json').read_text())

    def start(self):
        self.n += 1
        ident = {'context': 262144} if (self.change_identity and self.n > 1) else {'context': 131072}
        (self.root / 'active-run.json').write_text(json.dumps({'run_id': f'run{self.n}', 'started_utc': f't{self.n}',
                                                               'profile': self.profile, 'configuration': ident,
                                                               'source_sha256': 'abc'}))
        self.served = 0

    def run(self, args, timeout=1200):
        self.calls.append(args)
        if args[:3] == ['systemctl', '--user', 'stop']: return ''
        if args[0] == 'bash':
            assert args[2] == self.profile; self.start(); return ''
        return '{"ready": true}'

    def generate(self, messages, maximum, seed, temperature, ignore_eos):
        case = CASES[seed]
        rid = self.receipt()['run_id']
        if self.extra_on_clean and case == 'long-120000':
            self.log({'profile': self.profile, 'run_id': rid, 'case': 'zz', 'nll': [1.0]})
        if case not in self.lose:
            self.log({'profile': self.profile, 'run_id': rid, 'case': case, 'nll': [0.5] * maximum})
        self.served += 1
        return {'status': 200, 'usage': {'completion_tokens': maximum}}, 'T' + case

    def log(self, row):
        with open(self.root / 'reports/teacher.jsonl', 'a') as out: out.write(json.dumps(row) + '\n')


def load(sim):
    bench = types.SimpleNamespace(ROOT=sim.root, generate=sim.generate, headers=lambda: {}, document=None,
                                  run_receipt=sim.receipt)
    stubs = {'bench': bench, 'fixtures': types.SimpleNamespace(prepare=None),
             'matrix': types.SimpleNamespace(run=sim.run, PY='/py'),
             'transformers': types.SimpleNamespace(AutoTokenizer=None), 'requests': types.SimpleNamespace()}
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location('quality_v7_under_test', PATCHED)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    finally:
        for k, v in saved.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v
    module.time = types.SimpleNamespace(monotonic=iter(range(0, 10 ** 6, 10)).__next__, sleep=lambda s: None)
    sys.modules['matrix'] = stubs['matrix']  # restart_clean imports matrix lazily
    return module


class Tok:
    def decode(self, ids): return 'T' + CASES[ids[0]]


def cases():
    return [{'case': c, 'prompt': 'p' + c, 'ids': [i] * 4, 'seed': i} for i, c in enumerate(CASES)]


class QualityV7(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup(); sys.modules.pop('matrix', None)

    def test_restart_once_before_long120000_and_bound(self):
        sim = Sim(self.root); q = load(sim)
        rows, first, clean = q.teacher(Tok(), cases())
        self.assertEqual([r['case'] for r in rows], CASES)
        self.assertTrue(all(r['forced_text_matches'] and r['status'] == 200 for r in rows))
        self.assertEqual(first['run_id'], 'run1'); self.assertEqual([c['run_id'] for c in clean], ['run2'])
        self.assertEqual([r['run_id'] for r in rows], ['run1'] * 19 + ['run2'])
        self.assertEqual([r['clean_cache'] for r in rows], [False] * 19 + [True])
        self.assertEqual(sim.calls, [['systemctl', '--user', 'stop', 'qwen36-experiment'],
                                     ['bash', str(self.root / 'start-experiment.sh'), 'teacher-x'],
                                     ['/py', str(self.root / 'check-experiment.py'), '300']])
        # compare-quality.py's grouping (last row per profile/case) selects exactly these rows
        latest = {}
        for line in (self.root / 'reports/teacher.jsonl').read_text().splitlines():
            r = json.loads(line); latest[r['case']] = r['run_id']
        self.assertEqual(latest, {c: r['run_id'] for c, r in zip(CASES, rows)})

    def test_stale_rows_of_an_older_run_are_superseded(self):
        sim = Sim(self.root)
        for c in CASES: sim.log({'profile': 'teacher-x', 'run_id': 'old', 'case': c, 'nll': [9.0]})
        q = load(sim); rows, _, _ = q.teacher(Tok(), cases())
        self.assertEqual(rows[-1]['run_id'], 'run2')

    def test_missing_engine_row_fails(self):
        sim = Sim(self.root, lose={'long-120000'})
        for c in CASES: sim.log({'profile': 'teacher-x', 'run_id': 'old', 'case': c, 'nll': [9.0]})
        q = load(sim)
        with self.assertRaisesRegex(RuntimeError, 'not bound'): q.teacher(Tok(), cases())

    def test_clean_run_serving_other_cases_fails(self):
        sim = Sim(self.root, extra_on_clean=True); q = load(sim)
        with self.assertRaisesRegex(RuntimeError, 'served other teacher cases'): q.teacher(Tok(), cases())

    def test_restart_with_other_identity_fails(self):
        sim = Sim(self.root, change_identity=True); q = load(sim)
        with self.assertRaisesRegex(RuntimeError, 'identical receipt'): q.teacher(Tok(), cases())

    def test_no_receipt_fails_before_any_request(self):
        sim = Sim(self.root); (self.root / 'active-run.json').write_text('{}'); q = load(sim)
        with self.assertRaisesRegex(RuntimeError, 'active-run.json'): q.teacher(Tok(), cases())
        self.assertFalse((self.root / 'reports/teacher.jsonl').exists())

    def test_shared_functions_byte_identical(self):
        def funcs(path):
            text = path.read_text(encoding='utf-8'); tree = ast.parse(text)
            return {n.name: ast.get_source_segment(text, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
        a, b = funcs(ORIG), funcs(PATCHED)
        for name in ('code_check', 'request', 'functional'):
            self.assertEqual(a[name], b[name], name)
        self.assertEqual(set(b) - set(a), {'teacher_rows', 'restart_clean', 'teacher'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
