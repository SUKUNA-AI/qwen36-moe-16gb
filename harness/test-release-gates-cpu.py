"""release-gates.sh (M5): dry-run wiring, unique names, site paths, helpers and the 0.8 GiB VRAM floor (CPU only).

The driver runs against a synthetic site: config/local.env points at a temp overlay whose FreeToken tree is this
checkout (symlink), systemctl/flock are stubs on PATH, and the GPU python is this interpreter. Nothing starts.
"""
import copy
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
DRIVER = HERE / 'gates/release-gates.sh'
CANDIDATE, BASELINE = 'main-release-shortroute-fla-wu-overlap', 'main-release'
GIB = 1 << 30


def load(name, path):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quality = load('quality_fixtures', HERE / 'test-quality-evidence-cpu.py')
helpers = load('release_helpers', HERE / 'gates/release-helpers.py')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


class Site:
    """Temp owner site: local.env, overlay N with this checkout's harness files, REF/FB evidence, stub tools."""

    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.S, self.N = self.root / 'runtime', self.root / 'runtime/optimization'
        self.N.mkdir(parents=True)
        self.env = self.root / 'local.env'
        write(self.env, f"# synthetic\nQWEN36_ROOT={self.S}   # runtime\nQWEN36_NATIVE='{self.N}'\nQWEN36_WSL_DISTRO=archlinux\n")
        for name in ('qualify.py', 'quality.py', 'compare-quality.py', 'bench.py', 'matrix.py', 'compare-ab.py', 'tool-pilot.py',
                     'soak.py', 'soak-analysis.py', 'make-qualification.py', 'run-context-gates.sh'):
            shutil.copyfile(HERE / name, self.N / name)
        for name in ('start-experiment.sh', 'check-experiment.py'):
            write(self.N / name, '# site launcher\n')
        (self.N / 'python').symlink_to(REPO / 'freetoken/python', target_is_directory=True)
        for name in (CANDIDATE, BASELINE, 'bf128'):
            write(self.N / f'config/{name}.json', {'context': 131072, 'chunk': 2048, 'qualified': False, 'flags': [], 'env': {}})
        write(self.N / 'config/teacher-fixture.json', '{}')
        f = quality.fixtures()
        reports = self.N / 'reports'
        write(reports / 'teacher.jsonl', ''.join(json.dumps(v) + '\n' for k, v in f[4].items() if k[0] == 'teacher-bf128'))
        write(reports / 'ref.json', f[0])
        write(reports / 'fb.json', f[2])
        other = copy.deepcopy(f[2]); other['run']['packages']['torch'] = 'other'
        write(reports / 'fb-other.json', other)
        soak_tests = load('soak_tests_ref', HERE / 'test-soak-cpu.py')
        soak_tests.write_reference(reports, soak_tests.pass_receipt(), name='soak-f1')
        self.bin = self.root / 'bin'
        for tool, body in (('systemctl', 'exit 3'), ('flock', 'echo "flock $*" >> "$STUB_LOG"; exit 0')):
            write(self.bin / tool, f'#!/bin/sh\n{body}\n')
            (self.bin / tool).chmod(0o755)

    def run(self, *args, env=None):
        environment = dict(os.environ, PATH=f'{self.bin}{os.pathsep}{os.environ["PATH"]}', QWEN36_GPU_PYTHON=sys.executable,
                           STUB_LOG=str(self.root / 'stub.log'), **(env or {}))
        environment.pop('QWEN36_GPU_LOCK', None)
        return subprocess.run(['bash', str(DRIVER), '--env', str(self.env), *args], capture_output=True, text=True,
                              env=environment, timeout=120)

    def listing(self):
        return sorted(str(p.relative_to(self.N)) for p in self.N.rglob('*') if 'python' not in p.parts)


BASE = ('--candidate', CANDIDATE, '--reference-report', 'ref', '--functional-baseline', 'fb', '--reference-soak',
        'soak-f1-analysis.json', '--run-id', 'r1', '--dry-run')


class DryRun(unittest.TestCase):
    def test_screen_dry_run_wires_arguments_names_and_site_paths(self):
        with tempfile.TemporaryDirectory() as d:
            site = Site(d); before = site.listing()
            r = site.run(*BASE)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            out = r.stdout
            self.assertIn('precondition failures: 0', out)
            self.assertIn(f'{site.N}/qualify.py --candidate {CANDIDATE} --name rg-r1-q-{CANDIDATE} '
                          '--reference-report ref --functional-baseline fb', out)
            self.assertIn(f'{site.N}/tool-pilot.py --profile bf128 --repeats 3 --name rg-r1-tool', out)
            self.assertIn(f'--profiles {BASELINE},{CANDIDATE} --pairs 3 --name rg-r1-m --output 128 --tool-pilot rg-r1-tool', out)
            self.assertIn('screen --matrix rg-r1-m', out)
            self.assertNotIn('compare-ab.py --name', out)
            self.assertIn(f'run-context-gates.sh {CANDIDATE} 131072 120000 rg-r1-c128', out)
            self.assertIn('soak-rg-r1-pilot 30', out)
            self.assertIn('soak-rg-r1 1800', out)
            self.assertIn(f'--expected-identity {site.N}/reports/rg-r1-m-manifest.json', out)
            self.assertIn(f'--reference-soak {site.N}/reports/soak-f1-analysis.json; soak-check', out)
            self.assertIn('ok: reference soak soak-f1-analysis.json is a PASS analysis bound to its receipt', out)
            self.assertIn(f'flock -w 3600 -E 75 {site.S}/run/gpu-exclusive.lock', out)
            self.assertIn('ok: overlay harness + FreeToken sources == this checkout', out)
            self.assertIn('ok: REF ref / FB fb are run-bound schema-2 evidence', out)
            # Read-only: no steps, log, summary or report appears in the overlay.
            self.assertEqual(site.listing(), before)

    def test_release_mode_uses_five_pairs_compare_ab_and_two_hours(self):
        with tempfile.TemporaryDirectory() as d:
            site = Site(d)
            r = site.run(*BASE, '--release')
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('--pairs 5', r.stdout)
            self.assertIn(f'compare-ab.py --name rg-r1-m --baseline {BASELINE} --candidate {CANDIDATE} --output 128', r.stdout)
            self.assertIn('soak-rg-r1 7200', r.stdout)
            self.assertIn('soak-check --release', r.stdout)
            self.assertIn('release mode', r.stdout)

    def test_dry_run_reports_each_failed_precondition(self):
        cases = {
            'existing report name': lambda s: write(s.N / f'reports/quality-rg-r1-q-{CANDIDATE}.json', '{}'),
            'overlay file missing': lambda s: (s.N / 'tool-pilot.py').unlink(),
            'overlay differs from checkout': lambda s: write(s.N / 'quality.py', '# stale\n'),
            'candidate qualified': lambda s: write(s.N / f'config/{CANDIDATE}.json', {'qualified': True}),
            'reference soak receipt changed': lambda s: write(s.N / 'reports/soak-f1.json', '{}'),
        }
        for label, mutate in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as d:
                site = Site(d); mutate(site)
                r = site.run(*BASE)
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                self.assertIn('DRY-RUN FAIL', r.stdout)
                self.assertNotIn('precondition failures: 0', r.stdout)
        with tempfile.TemporaryDirectory() as d:
            site = Site(d)
            r = site.run('--candidate', CANDIDATE, '--reference-report', 'ref', '--functional-baseline', 'fb-other',
                         '--reference-soak', 'soak-f1-analysis.json', '--run-id', 'r1', '--dry-run')
            self.assertEqual(r.returncode, 1)
            self.assertIn('DRY-RUN FAIL: REF ref / FB fb-other', r.stdout)
        with tempfile.TemporaryDirectory() as d:
            site = Site(d)
            r = site.run('--candidate', 'main-release-unlisted', '--reference-report', 'ref', '--functional-baseline', 'fb',
                         '--reference-soak', 'soak-f1-analysis.json', '--run-id', 'r1', '--dry-run')
            self.assertEqual(r.returncode, 1)
            self.assertIn('DRY-RUN FAIL: teacher allowlist', r.stdout)

    def test_usage_errors_exit_2_before_any_check(self):
        with tempfile.TemporaryDirectory() as d:
            site = Site(d)
            no_soak = tuple(a for a in BASE if a not in ('--reference-soak', 'soak-f1-analysis.json'))
            for args in (BASE[2:], ('--candidate', CANDIDATE, '--reference-report', 'ref', '--dry-run'), no_soak,
                         (*BASE[:-3], '--run-id', 'Bad_Run', '--dry-run'), (*BASE, '--baseline', CANDIDATE), (*BASE, '--bogus')):
                with self.subTest(args=args):
                    self.assertEqual(site.run(*args).returncode, 2)
            write(site.env, 'QWEN36_ROOT=/home/USER/qwen36-api\nQWEN36_NATIVE=/home/USER/qwen36-api/optimization\n')
            r = site.run(*BASE)
            self.assertEqual(r.returncode, 2)
            self.assertIn('placeholders', r.stderr)

    def test_f1_driver_is_left_as_the_archive(self):
        text = (HERE / 'gates/final-gates.sh').read_text(encoding='utf-8')
        self.assertIn('qualify.py" --candidate "$c" >>', text)
        self.assertNotIn('release-helpers', text)


class Helpers(unittest.TestCase):
    def run_helper(self, *args):
        try:
            return helpers.main(list(args))
        except SystemExit as exit_:
            return exit_.code

    def test_extract_run_ignores_only_the_declared_teacher_knobs(self):
        knobs = load('mq', HERE / 'make-qualification.py').RUN_DIAGNOSTIC_ENV
        cfg = {'context': 131072, 'chunk': 2048, 'qualified': False, 'flags': [], 'env': {'QWEN36_PREFIX_MARGIN': '8'}}
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d); write(root / 'cfg.json', cfg)
            def case(env, teacher, drop_flags=False, profile='teacher-x'):
                configuration = dict(copy.deepcopy(cfg), env=env)
                if drop_flags:
                    configuration.pop('flags')
                run = dict(quality.make_receipt(profile, 'r' * 32), configuration=configuration)
                write(root / 'report.json', {'run': run})
                out = root / f'out-{len(list(root.iterdir()))}.json'
                return self.run_helper('extract-run', '--report', str(root / 'report.json'), '--profile', profile,
                                       '--config', str(root / 'cfg.json'), *(['--teacher'] if teacher else []), '--out', str(out))
            teacher_env = dict(cfg['env'], **{k: '/synthetic' for k in knobs})
            self.assertEqual(case(teacher_env, True), 0)
            self.assertEqual(case(teacher_env, True, drop_flags=True), 0)
            self.assertEqual(case(cfg['env'], False, profile='x'), 0)
            missing = dict(teacher_env); missing.pop('QWEN36_TEACHER_REFERENCE_REPORT')
            for env, teacher in ((missing, True), (dict(teacher_env, QWEN36_TEACHER_CREATE_REFERENCE='1'), True),
                                 (teacher_env, False), (dict(cfg['env'], QWEN36_PREFIX_MARGIN='1'), False)):
                with self.subTest(env=sorted(env), teacher=teacher):
                    self.assertEqual(case(env, teacher, profile='teacher-x' if teacher else 'x'), 3)

    def test_summary_screen_never_releases_and_release_needs_every_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d); steps = root / 'steps'
            lines = ['CONFIG candidate=x', 'A PASS t=1s', 'B PASS q', 'D.pilot PASS', 'D.matrix PASS', 'C PASS', 'E.pilot PASS',
                     'E.soak PASS']
            write(steps, '\n'.join(lines + ['D.compare SCREEN s']) + '\n')
            self.assertEqual(self.run_helper('summary', '--steps', str(steps), '--run', 'r', '--out', str(root / 's.json')), 0)
            s = json.loads((root / 's.json').read_text())
            self.assertTrue(s['gates_pass']); self.assertFalse(s['releasable'])
            self.assertEqual(s['blocking'], ['screen_mode_never_releasable'])
            self.run_helper('summary', '--steps', str(steps), '--run', 'r', '--out', str(root / 's.json'), '--release')
            self.assertEqual(json.loads((root / 's.json').read_text())['blocking'], ['D.compare'])
            write(steps, '\n'.join(lines + ['D.compare PASS c', 'E.soak FAIL later']) + '\n')
            self.run_helper('summary', '--steps', str(steps), '--run', 'r', '--out', str(root / 's.json'), '--release')
            self.assertEqual(json.loads((root / 's.json').read_text())['blocking'], ['E.soak'])

    def test_soak_check_modes(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / 'a.json'
            def check(doc, *flags):
                write(path, doc); return self.run_helper('soak-check', '--analysis', str(path), *flags)
            short = {'verdict': 'FAIL', 'two_hour_execution_gate_pass': False,
                     'pilot_view': {'all_case_types_passed': True, 'failures_excluding_duration': [],
                                    'failures_excluding_duration_and_reference': []}}
            self.assertEqual(check(short), 0)
            self.assertEqual(check(short, '--release'), 3)
            slower = copy.deepcopy(short)
            slower['pilot_view'].update(failures_excluding_duration=['ttft_regression'],
                                        failures_excluding_duration_and_reference=[])
            self.assertEqual(check(slower), 3)
            self.assertEqual(check(slower, '--pilot'), 0)
            self.assertEqual(check({'verdict': 'PASS', 'two_hour_execution_gate_pass': True, 'pilot_view': {}}, '--release'), 0)

    def test_context_check_binds_the_context_run_to_the_qualified_teacher_run(self):
        f = quality.fixtures()
        teacher = dict(f[1], run=dict(f[1]['run'], profile='teacher-candidate'))
        base = quality.functional_report('candidate', 'qualify-functional')
        ctx = quality.functional_report('candidate', 'context-run')
        with tempfile.TemporaryDirectory() as d:
            reports = pathlib.Path(d)
            def check(ctx_doc, cancel_run='context-run', boundary=True):
                for name, doc in (('t', teacher), ('fb', base), ('c-functional', ctx_doc), ('c-boundary', {'pass': boundary}),
                                  ('c-backend', {'all_pass': True}), ('c-cancel', {'pass': True, 'run': {'run_id': cancel_run}})):
                    write(reports / f'{name}.json', doc)
                return self.run_helper('context-check', '--reports', str(reports), '--name', 'c', '--profile', 'candidate',
                                       '--teacher-report', 't', '--functional-report', 'fb')
            self.assertEqual(check(ctx), 0)
            failed = copy.deepcopy(ctx); failed['rows'][0]['pass'] = False
            self.assertEqual(check(failed), 3)
            same_as_base = copy.deepcopy(failed); base['rows'][0]['pass'] = False
            self.assertEqual(check(same_as_base), 0)
            base['rows'][0]['pass'] = True
            self.assertEqual(check(ctx, cancel_run='other'), 3)
            self.assertEqual(check(ctx, boundary=False), 3)
            drift = copy.deepcopy(ctx); drift['run']['packages']['torch'] = 'other'
            self.assertEqual(check(drift), 3)

    def test_screen_reports_medians_for_any_pair_count(self):
        with tempfile.TemporaryDirectory() as d:
            reports = pathlib.Path(d); arms = []
            for repeat in range(3):
                for profile, elapsed in ((BASELINE, 2.0 + repeat), (CANDIDATE, 1.0 + repeat)):
                    name = f'm-{repeat}-{profile}.json'
                    arms.append({'repeat': repeat, 'profile': profile, 'status': 'complete', 'report': name})
                    write(reports / name, {'rows': [{'case': 'fresh', 'length': 64, 'status': 200, 'elapsed_s': elapsed,
                                                     'ttft_content_s': .1, 'decode_client_tps': 100.0},
                                                    {'case': 'warmup', 'status': 200, 'elapsed_s': 99.0}]})
            write(reports / 'm-matrix.json', arms)
            out = reports / 'screen.json'
            self.assertEqual(self.run_helper('screen', '--reports', str(reports), '--matrix', 'm', '--baseline', BASELINE,
                                             '--candidate', CANDIDATE, '--out', str(out)), 0)
            row = json.loads(out.read_text())['rows'][0]
            self.assertEqual((row['length'], row['elapsed_s']['pairs'], row['elapsed_speedup']), (64, 3, 1.5))
            self.assertFalse(json.loads(out.read_text())['gate'])

    def test_env_parsing_takes_quoted_and_commented_values_and_refuses_placeholders(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / 'local.env'
            write(path, "QWEN36_ROOT=/srv/q   # comment\nQWEN36_NATIVE='/srv/q/optimization/'\n")
            import contextlib, io
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.assertEqual(self.run_helper('env', str(path)), 0)
            self.assertEqual(buffer.getvalue().split(), ['QWEN36_ROOT=/srv/q', 'QWEN36_NATIVE=/srv/q/optimization'])
            for bad in ('QWEN36_ROOT=/home/USER/qwen36-api\nQWEN36_NATIVE=/srv/n\n', 'QWEN36_NATIVE=/srv/n\n',
                        'QWEN36_ROOT=relative\nQWEN36_NATIVE=/srv/n\n'):
                write(path, bad)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(self.run_helper('env', str(path)), 3)


class VramFloor(unittest.TestCase):
    def test_floor_is_0_8_gib_everywhere(self):
        import bench
        self.assertEqual(bench.VRAM_FLOOR_BYTES, int(0.8 * GIB))
        soak_analysis = load('soak_analysis_floor', HERE / 'soak-analysis.py')
        self.assertEqual(soak_analysis.VRAM_FLOOR_BYTES, bench.VRAM_FLOOR_BYTES)
        for name in ('matrix.py', 'compare-ab.py', 'bench.py'):
            self.assertNotIn("owner's 0.5 GiB", (HERE / name).read_text(encoding='utf-8'))
        self.assertNotIn('GIB//2', (HERE / 'bench.py').read_text(encoding='utf-8'))

    def test_matrix_marks_an_arm_between_the_old_and_new_floor_invalid(self):
        import matrix
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d); plan = {'repeat': 0, 'profile': CANDIDATE}
            bench_path, task_path = matrix.arm_files(root, 'm', plan)
            write(bench_path, {'rows': [{'case': 'fresh', 'peaks': {'min_gpu_free_bytes': int(0.7 * GIB)}}]})
            write(task_path, {'steps': []})
            vram = matrix.arm_vram(root, 'm', plan, {'used_bytes': 0, 'free_bytes': 0, 'total_bytes': 0})
            self.assertEqual((vram['gate'], vram['code']), ('INVALID', 'gpu_min_free_below_floor'))


class ContextGatesScript(unittest.TestCase):
    def test_named_run_refuses_existing_reports_before_starting(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            write(root / 'start-experiment.sh', 'echo started >> "$1.started"\n')
            write(root / 'reports/ctx-boundary.json', '{}')
            env = dict(os.environ, QWEN36_DATA_ROOT=str(root), QWEN36_GPU_PYTHON=sys.executable)
            script = HERE / 'run-context-gates.sh'
            r = subprocess.run(['bash', str(script), CANDIDATE, '131072', '120000', 'ctx'], capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 2); self.assertIn('ctx-boundary.json', r.stderr)
            r = subprocess.run(['bash', str(script), CANDIDATE, '131072', '120000', 'Bad Name'], capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 2)
            self.assertFalse(list(root.glob('*.started')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
