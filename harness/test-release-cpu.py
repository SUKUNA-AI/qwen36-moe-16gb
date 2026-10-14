"""CPU-only release guards; temporary files and mocked systemctl, never live ROOT."""
import contextlib
import importlib.util
import io
import json
import pathlib
import re
import subprocess
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('release_under_test', HERE / 'release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def state(active='inactive', load='loaded', sub='dead', rc=0):
    return subprocess.CompletedProcess([], rc,
        f'LoadState={load}\nActiveState={active}\nSubState={sub}\n', 'private-error-not-printed')


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='qwen-release-cpu-')
        self.root = pathlib.Path(self.tmp.name)
        self.root_patch = mock.patch.object(release, 'ROOT', self.root)
        self.root_patch.start()
        self.backup = self.root / 'backups/release-fixture'
        self.backup.mkdir(parents=True)
        self.manifest = self.backup / 'manifest.json'
        self.rows = []
        for name in ('first.py', 'last.py'):
            target = self.root / name
            target.write_bytes(('after-' + name).encode())
            saved = self.backup / 'files' / name
            saved.parent.mkdir(exist_ok=True)
            saved.write_bytes(('before-' + name).encode())
            self.rows.append({'path': name, 'before': release.digest(saved), 'after': release.digest(target)})
        self.save_manifest()

    def tearDown(self):
        self.root_patch.stop()
        self.tmp.cleanup()

    def save_manifest(self):
        self.manifest.write_text(json.dumps({'files': self.rows}))

    def targets(self):
        return {row['path']: (self.root / row['path']).read_bytes() for row in self.rows}

    def test_verified_stopped_states_both_units(self):
        for response in (state(), state('failed', sub='failed'), state(load='masked'), state(load='not-found')):
            with self.subTest(response=response.stdout), mock.patch.object(release.subprocess, 'run', return_value=response) as run:
                release.stopped()
                self.assertEqual(run.call_count, 2)
                self.assertEqual([c.args[0][3] for c in run.call_args_list],
                    ['qwen36-inference.service', 'qwen36-experiment.service'])
                for call in run.call_args_list:
                    self.assertEqual(call.kwargs['timeout'], 15)
                    self.assertTrue(call.kwargs['capture_output'])

    def test_transitional_active_unknown_and_unverified_missing_refused(self):
        bad = [state(x) for x in ('active', 'activating', 'deactivating', 'reloading', 'refreshing', 'maintenance', '')]
        bad += [state(load='error'), state(load='not-found', sub='running'), state('failed', load='not-found')]
        bad += [state(rc=1), state(rc=3), subprocess.CompletedProcess([], 0, '', ''),
                subprocess.CompletedProcess([], 0, 'LoadState=loaded\nActiveState=inactive\n', ''),
                subprocess.CompletedProcess([], 0, state().stdout + 'ActiveState=active\n', '')]
        for response in bad:
            with self.subTest(response=response.stdout, rc=response.returncode), mock.patch.object(release.subprocess, 'run', return_value=response):
                with self.assertRaises(RuntimeError): release.stopped()

    def test_unavailable_and_timeout_refused(self):
        for error in (FileNotFoundError(), subprocess.TimeoutExpired('systemctl', 15)):
            with mock.patch.object(release.subprocess, 'run', side_effect=error):
                with self.assertRaises(type(error)): release.stopped()

    def test_experiment_transition_blocks_after_inference_stopped(self):
        with mock.patch.object(release.subprocess, 'run', side_effect=[state(), state('activating', sub='start')]):
            with self.assertRaises(RuntimeError): release.stopped()

    def test_corrupt_later_backup_no_partial_rollback(self):
        # Old rollback iterated reversed rows: last.py was already changed before
        # discovering first.py's corrupt backup. Verify neither file changes.
        (self.backup / 'files/first.py').write_bytes(b'corrupt')
        before = self.targets()
        with mock.patch.object(release, 'stopped') as stopped, mock.patch.object(release, 'atomic') as atomic:
            with self.assertRaises(RuntimeError): release.rollback(self.manifest, False)
            stopped.assert_not_called(); atomic.assert_not_called()
        self.assertEqual(self.targets(), before)

    def test_missing_later_backup_and_check_mode_refused(self):
        (self.backup / 'files/first.py').unlink()
        before = self.targets()
        for check in (False, True):
            with self.assertRaises(RuntimeError): release.rollback(self.manifest, check)
        self.assertEqual(self.targets(), before)

    def test_later_edit_refused_before_any_mutation(self):
        (self.root / 'first.py').write_bytes(b'owner later edit')
        before = self.targets()
        with mock.patch.object(release, 'atomic') as atomic:
            with self.assertRaises(RuntimeError): release.rollback(self.manifest, False)
            atomic.assert_not_called()
        self.assertEqual(self.targets(), before)

    def test_revalidation_after_state_query(self):
        before = self.targets()
        def corrupt(): (self.backup / 'files/first.py').write_bytes(b'changed during state query')
        with mock.patch.object(release, 'stopped', side_effect=corrupt), mock.patch.object(release, 'atomic') as atomic:
            with self.assertRaises(RuntimeError): release.rollback(self.manifest, False)
            atomic.assert_not_called()
        self.assertEqual(self.targets(), before)

    def test_active_rollback_refused_without_changes(self):
        before = self.targets()
        with mock.patch.object(release.subprocess, 'run', return_value=state('deactivating')):
            with self.assertRaises(RuntimeError): release.rollback(self.manifest, False)
        self.assertEqual(self.targets(), before)

    def test_success_check_then_rollback_and_idempotent(self):
        before = self.targets()
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(release, 'stopped') as stopped:
            release.rollback(self.manifest, True)
            stopped.assert_not_called(); self.assertEqual(self.targets(), before)
            release.rollback(self.manifest, False)
            for row in self.rows: self.assertEqual(release.digest(self.root / row['path']), row['before'])
            release.rollback(self.manifest, False)

    def test_manifest_duplicates_bad_hash_escape_refused(self):
        variants = [[self.rows[0], self.rows[0]], [{**self.rows[0], 'after':'bad'}],
                    [{**self.rows[0], 'path':'../escape.py'}], []]
        for rows in variants:
            self.manifest.write_text(json.dumps({'files': rows}))
            with self.assertRaises(RuntimeError): release.validate_rollback(self.manifest)


class ReleaseLibraryTests(unittest.TestCase):
    """Each validator library switch stages exactly its library, from the file the qualification configs load."""

    def setUp(self):
        self.validator = release.load_validator(HERE.parent / 'profile-runtime.py')
        self.tmp = tempfile.TemporaryDirectory(prefix='qwen-release-lib-cpu-')
        self.patches = [mock.patch.object(release, 'RUNTIME_SOURCE', pathlib.Path(self.tmp.name)),
                        mock.patch.object(release, 'verify_qualified_runtime', lambda validator, sources: None)]
        for patcher in self.patches: patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patches): patcher.stop()
        self.tmp.cleanup()

    @staticmethod
    def profiles(**switches):
        plain = {'chunk': 2048, 'memory_ratio': 0.75, 'kv_dtype': 'bf16', 'flags': [], 'qualified': False}
        return {'schema': 1, 'default': 'main128', 'profiles': {
            'baseline64': dict(plain, context=65536), 'long256': dict(plain, context=262144, **switches),
            'main128': dict(plain, context=131072, qualified=True, **switches)}}

    def libraries(self, profiles):
        return {k: v for k, v in release.staged_sources(self.validator, profiles).items() if k.startswith('release/lib/')}

    def test_each_library_switch_stages_its_pinned_source(self):
        self.assertEqual(self.libraries(self.profiles()), {})
        for switch, (_, library) in self.validator.NATIVE_LIBRARIES.items():
            with self.subTest(switch):
                self.assertEqual(set(self.libraries(self.profiles(**{switch: True}))), {library})
        root = release.ROOT / 'optimization'
        self.assertEqual(self.libraries(self.profiles(**{switch: True for switch in self.validator.NATIVE_LIBRARIES})), {
            'release/lib/libqwen_attention_gate.so': root / 'private-libraries/libqwen_attention_gate.so',
            'release/lib/libqwen_nvfp4_bitexact.so': root / 'private-libraries/libqwen_nvfp4_bitexact.so',
            'release/lib/libqwen36_gdn.so': root / 'private-cuda-gdn/pinned/libqwen36_gdn-a8a806eda3a90a65.so',
            'release/lib/libqwen_fp8_gemv.so': root / 'private-fp8gemv/libqwen_fp8_gemv-685c78cf89e60d02.so'})
        # A library request the release does not know (validator/release drift) is refused.
        with mock.patch.object(self.validator, 'required_libraries', lambda profiles: {'release/lib/libqwen36_kv.so'}):
            with self.assertRaises(RuntimeError): self.libraries(self.profiles())

    def test_fp8_source_is_the_build_the_runtime_module_pins(self):
        module = release.PROJECT / 'work/FreeToken-opt/python/freetoken/kernel/cuda_fp8_gemv.py'
        if not module.is_file(): self.skipTest('fp8-gemv patch not applied to the FreeToken runtime tree')
        pinned = re.search(r'^LIBRARY_SHA256 = "([0-9a-f]{64})"', module.read_text(), re.M).group(1)
        self.assertEqual(pinned, '685c78cf89e60d0205f0db3ac2040203dec27e510a1e543cb7a7d3097414093b')
        self.assertIn(pinned[:16], release.FP8_GEMV_LIBRARY_SOURCE.name)
        self.assertIn('a8a806eda3a90a65', release.GDN_LIBRARY_SOURCE.name)


if __name__ == '__main__': unittest.main(verbosity=2)
