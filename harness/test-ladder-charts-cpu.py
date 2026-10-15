"""CPU tests for tools/ladder-charts.py: fixture render, valid SVG, determinism, guards."""
import contextlib
import copy
import importlib.util
import io
import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from repo_paths import CHECKOUT

TOOL = CHECKOUT / 'tools' / 'ladder-charts.py'
FIXTURE = CHECKOUT / 'results' / 'ladder' / 'fixture-summary.json'
SVG_NS = '{http://www.w3.org/2000/svg}'
CHARTS = ['ttft-steps', 'decode-steps', 'waterfall-ttft-short', 'waterfall-decode16k', 'nsys-decode16k',
          'nsys-prefill-short', 'nsys-prefill16k', 'memory-steps', 'teacher-steps']


def load_tool():
    spec = importlib.util.spec_from_file_location('ladder_charts', TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lc = load_tool()


def fixture():
    return json.loads(FIXTURE.read_text(encoding='utf-8'))


def tiny(**over):
    doc = {'schema': 'qwen36-ladder-summary', 'schema_version': 1, 'lengths': [64, 16384],
           'steps': [{'id': 'L0', 'parent': None,
                      'ab': {'ttft_s': {'64': [1.0, 1.0, 1.0]}, 'decode_tps': {'16384': [100.0]}}},
                     {'id': 'L1', 'parent': 'L0',
                      'ab': {'reference': 'L0', 'pairs': 3,
                             'ttft_s': {'64': [0.8, 0.9, 0.7]}, 'ref_ttft_s': {'64': [1.0, 1.0, 1.0]},
                             'decode_tps': {'16384': [110.0]}}}]}
    doc.update(over)
    return doc


def render(doc, outdir):
    with contextlib.redirect_stdout(io.StringIO()):
        return lc.render(lc.Ladder(doc), str(outdir))


class FixtureRender(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fixture_is_marked_synthetic(self):
        doc = fixture()
        self.assertIs(doc['synthetic'], True)
        self.assertIn('СИНТЕТИКА', doc['fixture_note'])
        lc.validate(doc)

    def test_all_charts_both_themes_valid_svg(self):
        written = render(fixture(), self.out)
        names = sorted(os.path.basename(p) for p in written)
        expected = sorted([f'{c}.svg' for c in CHARTS] + [f'{c}-dark.svg' for c in CHARTS]
                          + ['ladder-table.md', 'ladder-table.en.md', 'DRAFT.md'])
        self.assertEqual(names, expected)
        for c in CHARTS:
            for suffix in ('', '-dark'):
                path = self.out / f'{c}{suffix}.svg'
                raw = path.read_bytes()
                self.assertNotIn(b'\r\n', raw)
                root = ET.fromstring(raw)
                self.assertEqual(root.tag, SVG_NS + 'svg')
                self.assertTrue(root.get('viewBox'))
                self.assertTrue(root.find(SVG_NS + 'title').text)
                text = raw.decode('utf-8')
                for bad in ('nan', 'inf', 'None'):
                    self.assertNotIn(f'"{bad}', text, f'{path.name}: {bad} in an attribute')
                self.assertIn('СИНТЕТИКА · ЧЕРНОВИК', text)

    def test_deterministic(self):
        a, b = self.out / 'a', self.out / 'b'
        render(fixture(), a)
        render(fixture(), b)
        files = sorted(os.listdir(a))
        self.assertEqual(files, sorted(os.listdir(b)))
        for f in files:
            self.assertEqual((a / f).read_bytes(), (b / f).read_bytes(), f)

    def test_tables_mark_draft(self):
        render(fixture(), self.out)
        self.assertTrue((self.out / 'ladder-table.md').read_text(encoding='utf-8').startswith('> **СИНТЕТИКА'))
        self.assertTrue((self.out / 'ladder-table.en.md').read_text(encoding='utf-8').startswith('> **SYNTHETIC'))
        self.assertNotIn('н/д', (self.out / 'ladder-table.en.md').read_text(encoding='utf-8'))


class Guards(unittest.TestCase):
    def test_synthetic_never_written_to_final_charts(self):
        final = CHECKOUT / 'results' / 'ladder' / 'charts'
        existed = final.exists()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = lc.main(['--input', str(FIXTURE), '--output', str(final)])
        self.assertEqual(rc, 2)
        self.assertIn('synthetic', err.getvalue())
        self.assertEqual(final.exists(), existed)

    def test_default_output_for_synthetic_is_draft(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'in.json'
            src.write_text(json.dumps(tiny(synthetic=True)), encoding='utf-8')
            out = Path(d) / 'out'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(lc.main(['--input', str(src), '--output', str(out)]), 0)
            self.assertTrue((out / 'DRAFT.md').exists())

    def test_invalid_documents(self):
        bad = [
            {'schema': 'other', 'schema_version': 1, 'steps': [{'id': 'L0'}]},
            tiny(schema_version=2),
            tiny(steps=[]),
            tiny(steps=[{'id': 'L0'}, {'id': 'L0'}]),
            tiny(steps=[{'id': 'L0', 'parent': 'L9'}]),
            tiny(steps=[{'id': 'L0', 'ab': {'reference': 'X'}}]),
            tiny(steps=[{'id': 'L0', 'cumulative': False}]),
            tiny(lengths=['64']),
        ]
        for doc in bad:
            with self.assertRaises(lc.LadderError, msg=json.dumps(doc)[:120]):
                lc.validate(doc)

    def test_main_rejects_bad_input(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'in.json'
            src.write_text('{not json', encoding='utf-8')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(lc.main(['--input', str(src), '--output', d]), 2)
                self.assertEqual(lc.main(['--input', str(Path(d) / 'missing.json'), '--output', d]), 2)


class Math(unittest.TestCase):
    def test_paired_ratio_is_median_of_pairs(self):
        lad = lc.Ladder(tiny())
        r, how = lad.ratio(lad.by_id['L1'], 'ttft_s', '64')
        self.assertEqual(how, 'pairs')
        self.assertAlmostEqual(r, 0.8)
        base, chain = lad.chain('ttft_s', '64')
        self.assertEqual(base, 1.0)
        self.assertAlmostEqual(chain[-1][2], 0.8)

    def test_fallback_to_medians_without_pairs(self):
        lad = lc.Ladder(tiny())
        r, how = lad.ratio(lad.by_id['L1'], 'decode_tps', '16384')
        self.assertEqual(how, 'medians')
        self.assertAlmostEqual(r, 1.1)

    def test_pairs_against_another_arm_are_not_used(self):
        doc = tiny()
        doc['steps'].insert(1, {'id': 'X', 'parent': 'L0', 'cumulative': False,
                                'ab': {'ttft_s': {'64': [0.5]}}})
        doc['steps'][2]['ab']['reference'] = 'X'
        lad = lc.Ladder(doc)
        r, how = lad.ratio(lad.by_id['L1'], 'ttft_s', '64')
        self.assertEqual(how, 'medians')
        self.assertAlmostEqual(r, 0.8)

    def test_missing_step_keeps_chain(self):
        doc = tiny()
        doc['steps'].append({'id': 'L2', 'parent': 'L1'})
        lad = lc.Ladder(doc)
        _, chain = lad.chain('ttft_s', '64')
        self.assertIsNone(chain[-1][3])
        self.assertAlmostEqual(chain[-1][2], 0.8)

    def test_median(self):
        self.assertEqual(lc.median([3, 1, 2]), 2)
        self.assertEqual(lc.median([4, 1, 2, 3]), 2.5)
        self.assertIsNone(lc.median([None, 'x']))
        self.assertIsNone(lc.median(None))

    def test_release_ratio(self):
        doc = tiny(release_ab={'candidate': 'L1', 'reference': 'L0', 'pairs': 2,
                               'ttft_s': {'64': [0.5, 0.6]}, 'ref_ttft_s': {'64': [1.0, 1.0]}})
        self.assertAlmostEqual(lc.Ladder(doc).release_ratio('ttft_s', '64'), 0.55)


class Rendering(unittest.TestCase):
    def test_sparse_document_renders(self):
        doc = {'schema': 'qwen36-ladder-summary', 'schema_version': 1,
               'steps': [{'id': 'L0'}, {'id': 'L1', 'parent': 'L0'}]}
        with tempfile.TemporaryDirectory() as d:
            render(doc, d)
            for c in CHARTS:
                ET.parse(os.path.join(d, c + '.svg'))
            self.assertFalse(os.path.exists(os.path.join(d, 'DRAFT.md')))
            self.assertNotIn('СИНТЕТИКА', Path(d, 'ttft-steps.svg').read_text(encoding='utf-8'))

    def test_teacher_pass_needs_bitexact(self):
        doc = tiny()
        doc['steps'][0]['teacher'] = {'status': 'pass', 'passed': 125, 'total': 125, 'max_abs': 0.0}
        doc['steps'][1]['teacher'] = {'status': 'pass', 'passed': 125, 'total': 125, 'max_abs': 0.5}
        lad = lc.Ladder(doc)
        svg = '\n'.join(lc.teacher_chart(lad, lc.THEMES['light']).parts)
        self.assertEqual(svg.count('>PASS<'), 1)
        self.assertEqual(svg.count('>FAIL<'), 1)

    def test_release_ab_shown_in_waterfall(self):
        doc = tiny(release_ab={'candidate': 'L1', 'reference': 'L0', 'pairs': 5,
                               'ttft_s': {'64': [0.75] * 5}, 'ref_ttft_s': {'64': [1.0] * 5}})
        lad = lc.Ladder(doc)
        svg = '\n'.join(lc.waterfall_chart(lad, lc.THEMES['dark'], 'ttft_s', '64').parts)
        self.assertIn('5 пар: −25.0 %', svg)
        self.assertIn('−200 мс', svg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
