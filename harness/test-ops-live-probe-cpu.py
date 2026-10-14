"""CPU-only tests for ops-live-probe.py. No sockets, SSH, services, key files or GPU."""
import importlib.util
import io
import json
import pathlib
import socket
import subprocess
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

path = pathlib.Path(__file__).with_name('ops-live-probe.py')
spec = importlib.util.spec_from_file_location('ops_live_probe_fixture', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
KEY = 'SYNTHETIC-NOT-A-CREDENTIAL-0123456789abcdef0123456789'
GEN = 'a' * 32
LEASE = 'b' * 32


def healthy(instance='inst-1', ctx=131072, active=0):
    health = {'status': 'ok', 'model': 'qwen36-35b', 'instance_id': instance, 'maintenance': 'serving'}
    stats = {'instance_id': instance, 'model': {'id': 'qwen36-35b', 'ctx': ctx}, 'requests': {'active': active}}
    return lambda key, path: dict(health) if path == '/health' else json.loads(json.dumps(stats))


def units(inference='active', experiment='inactive'):
    return lambda: {m.UNITS[0]: {'ActiveState': inference, 'SubState': 'running', 'MainPID': '42', 'NRestarts': '0',
                                 'InvocationID': 'inv'},
                    m.UNITS[1]: {'LoadState': 'not-found', 'ActiveState': experiment, 'SubState': 'dead'}}


class Reply(io.BytesIO):
    def __init__(self, status, data=b'', content_type='application/json', headers=None):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Type': content_type, **(headers or {})}


class IdentityTests(unittest.TestCase):
    def run_identity(self, **kw):
        args = dict(json_get=healthy(), states=units(), selected=lambda: 'main128')
        args.update(kw)
        return m.identity(KEY, 'main128', **{k: v for k, v in args.items() if k in ('expect_instance', 'different_from', 'json_get', 'states', 'selected')})

    def test_matching_identity_passes_without_key_in_report(self):
        report = self.run_identity()
        self.assertTrue(report['pass'])
        self.assertEqual(report['instance_id'], 'inst-1')
        self.assertNotIn(KEY, json.dumps(report))

    def test_wrong_served_context_fails(self):
        for ctx in (65536, 262144, None, '131072'):
            self.assertFalse(self.run_identity(json_get=healthy(ctx=ctx))['pass'])

    def test_missing_or_mismatched_instance_fails(self):
        self.assertFalse(self.run_identity(json_get=healthy(instance=None))['pass'])
        self.assertFalse(self.run_identity(json_get=healthy(instance=''))['pass'])
        def split(key, path):
            return healthy('inst-1')(key, path) if path == '/health' else healthy('inst-2')(key, path)
        self.assertFalse(self.run_identity(json_get=split)['pass'])

    def test_wrong_model_or_loading_fails(self):
        def wrong(key, path):
            value = healthy()(key, path)
            if path == '/health':
                value['model'] = 'other'
            return value
        self.assertFalse(self.run_identity(json_get=wrong)['pass'])
        def loading(key, path):
            value = healthy()(key, path)
            if path == '/health':
                value['status'] = 'loading'
            return value
        self.assertFalse(self.run_identity(json_get=loading)['pass'])

    def test_selected_profile_file_must_match(self):
        for chosen in (None, 'long256', 'baseline64', 'garbage'):
            report = self.run_identity(selected=lambda chosen=chosen: chosen)
            self.assertFalse(report['pass'])
            self.assertNotIn('garbage', json.dumps(report))

    def test_units_must_be_exclusive(self):
        self.assertFalse(self.run_identity(states=units(experiment='active'))['pass'])
        self.assertFalse(self.run_identity(states=units(inference='activating'))['pass'])

    def test_expected_and_previous_instance_guards(self):
        self.assertTrue(self.run_identity(expect_instance='inst-1')['pass'])
        self.assertFalse(self.run_identity(expect_instance='inst-0')['pass'])
        self.assertTrue(self.run_identity(different_from='inst-0')['pass'])
        self.assertFalse(self.run_identity(different_from='inst-1')['pass'])

    def test_unit_state_parser_fails_closed(self):
        done = types.SimpleNamespace(returncode=0, stdout='LoadState=loaded\nActiveState=active\nActiveState=inactive\nSubState=running\n')
        with self.assertRaises(m.Refused):
            m.unit_states(run=lambda *a, **k: done)
        done = types.SimpleNamespace(returncode=0, stdout='LoadState=loaded\n')
        with self.assertRaises(m.Refused):
            m.unit_states(run=lambda *a, **k: done)
        with self.assertRaises(m.Refused):
            m.unit_states(run=lambda *a, **k: types.SimpleNamespace(returncode=1, stdout=''))

    def test_local_opener_ignores_proxy_and_redirect(self):
        proxies = {name: 'http://third-party.invalid:8888' for name in
                   ('HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy', 'ALL_PROXY', 'all_proxy')}
        with mock.patch.dict(m.os.environ, {**proxies, 'NO_PROXY': '', 'no_proxy': ''}):
            handlers = m.direct_opener().handlers
        self.assertFalse(any(isinstance(h, m.urllib.request.ProxyHandler) and h.proxies for h in handlers))
        self.assertTrue(any(isinstance(h, m.NoRedirect) for h in handlers))
        self.assertIsNone(m.NoRedirect().redirect_request(None, None, 302, '', {}, 'http://third-party.invalid/'))


class RemoteTests(unittest.TestCase):
    """Executes the REMOTE source with all transport functions replaced by fixtures."""

    def setUp(self):
        self.ns = {}
        exec(compile(m.REMOTE, 'synthetic-remote-overload', 'exec'), self.ns)
        # Long pauses yield briefly so the nonstream holder thread can start;
        # every other wait is instantaneous. The clock advances per reading.
        self.ns['pause'] = lambda seconds: time.sleep(0.05) if seconds >= 2 else None
        self.t = [0.0]
        def clock():
            self.t[0] += 0.5
            return self.t[0]
        self.ns['clock'] = clock
        self.own_active = False
        self.holding = False
        self.nonstream_seen_busy = threading.Event()
        self.nonstream_admitted = True
        self.nonstream_calls = 0
        self.release_after = 0
        self.burst_template = [401] * 20 + [429] * 20
        self.burst_codes = []
        self.oversize_status = 413
        self.busy_headers = {'Retry-After': '2'}
        self.busy_status = 429
        self.ns['call'] = self.call
        self.ns['raw_exchange'] = self.raw_exchange
        self.ns['open_stream'] = self.open_stream
        self.ns['close_stream'] = self.close_stream

    def open_stream(self, key, base, payload, timeout=120):
        self.assertEqual(key, KEY)
        self.assert_cannot_finish_early(payload)
        self.own_active = True
        return Reply(200, b'', 'text/event-stream'), 1.25

    def close_stream(self, response):
        self.own_active = False

    def raw_exchange(self, key, base, body_size, payload=None, hold_seconds=0.0, timeout=30.0):
        self.assertEqual(key, KEY)
        if payload is not None:
            self.assertIs(payload['stream'], False)
            self.assert_cannot_finish_early(payload)
            self.nonstream_calls += 1
            if self.nonstream_admitted:
                self.holding = True
                self.own_active = True
                self.nonstream_seen_busy.wait(5)
                self.own_active = False  # client closed its own connection
                self.holding = False
            return {'sent': 10, 'closed_by_client': True}
        self.assertGreater(body_size, 2 * 1024 * 1024)
        return {'status': self.oversize_status, 'body_bytes_declared': body_size, 'body_bytes_sent': 65536, 'send_error_type': None}

    def call(self, key, base, path, body=None, auth=True, timeout=30):
        if not auth:
            code = self.burst_codes.pop(0) if self.burst_codes else 401
            data = json.dumps({'error': {'message': 'Request rate exceeded' if code == 429 else 'Invalid or missing Bearer token'}}).encode()
            return Reply(code, data)
        self.assertEqual(key, KEY)
        if body.get('max_tokens') == 8:
            if self.own_active:
                if self.holding and not body.get('stream'):
                    self.nonstream_seen_busy.set()
                return Reply(self.busy_status, b'{"error":{"message":"One inference request is already active; no waiting queue"}}',
                             headers=self.busy_headers)
            return Reply(200, json.dumps({'model': 'qwen36-35b'}).encode())
        if body.get('max_tokens') == 16:
            if self.own_active or self.release_after > 0:
                self.release_after -= 1
                return Reply(429, b'{"error":{"message":"busy"}}', headers={'Retry-After': '2'})
            return Reply(200, json.dumps({'model': 'qwen36-35b', 'choices': [{'message': {'content': 'PRIVATE_ANSWER'}}]}).encode())
        raise AssertionError('unexpected request')

    def collect(self):
        self.burst_codes = list(self.burst_template)
        self.nonstream_seen_busy.clear()
        self.own_active = False
        return self.ns['collect'](KEY)

    def assert_cannot_finish_early(self, payload):
        # Own long generations use the guard's maximum budget and ignore EOS.
        self.assertEqual(payload['max_tokens'], 8192)
        self.assertIs(payload['ignore_eos'], True)

    def row(self, report, case):
        return next(r for r in report['rows'] if r['case'] == case)

    def test_fast_release_is_attributed_to_the_client_close(self):
        report = self.collect()
        for case in ('sse-client-close-releases-slot', 'nonstream-client-close-releases-slot'):
            row = self.row(report, case)
            self.assertTrue(row['pass'], row)
            self.assertIs(row['release_attributable_to_cancel'], True)
            self.assertLessEqual(row['release_s'], row['release_deadline_s'])
            self.assertLess(row['own_request_age_s'], row['natural_completion_floor_s'])
        self.assertEqual(self.nonstream_calls, 1)

    def test_release_only_by_natural_completion_fails_and_stacks_nothing(self):
        # Cancel ignored: the slot frees itself only when the own request ends by
        # itself (~15 s of busy 429 here). The 5 s criterion never sees it.
        self.release_after = 30
        report = self.collect()
        sse = self.row(report, 'sse-client-close-releases-slot')
        self.assertFalse(sse['pass'])
        self.assertEqual(sse['status'], 429)
        self.assertIs(sse['release_attributable_to_cancel'], False)
        nonstream = self.row(report, 'nonstream-client-close-releases-slot')
        self.assertFalse(nonstream['pass'])
        self.assertEqual(nonstream['skipped'], 'previous_release_failed_slot_may_be_held')
        self.assertEqual(self.nonstream_calls, 0)  # no second long generation stacked
        self.assertFalse(report['pass'])

    def test_widened_deadline_still_rejects_release_after_natural_floor(self):
        # Even with a (hypothetically) widened deadline, a 200 that arrives after the
        # earliest possible natural completion is not attributable to the close.
        self.ns['RELEASE_DEADLINE'] = 10.0 ** 4
        natural_s = self.ns['OWN_MAX_TOKENS'] / 131.4  # fastest measured decode, CLAUDE-HANDOFF section 8
        self.release_after = int(natural_s / 0.5) + 1   # fake clock advances 0.5 s per reading
        report = self.collect()
        sse = self.row(report, 'sse-client-close-releases-slot')
        self.assertEqual(sse['status'], 200)
        self.assertIs(sse['release_attributable_to_cancel'], False)
        self.assertGreaterEqual(sse['own_request_age_s'], sse['natural_completion_floor_s'])
        self.assertFalse(sse['pass'])
        self.assertFalse(report['pass'])

    def test_release_window_is_far_below_natural_completion(self):
        ns = self.ns
        self.assert_cannot_finish_early(ns['LONG_PAYLOAD'])
        self.assertEqual(ns['RELEASE_DEADLINE'], 5.0)
        self.assertGreaterEqual(ns['DECODE_CEILING_TPS'], 2 * 131.4)
        self.assertAlmostEqual(ns['NATURAL_FLOOR_S'], 8192 / ns['DECODE_CEILING_TPS'])
        # nonstream: hold + full release window must stay well inside the floor
        self.assertLess(2 * (ns['NONSTREAM_HOLD_S'] + ns['RELEASE_DEADLINE']), ns['NATURAL_FLOOR_S'])
        # at the fastest measured decode the own request would run >= 60 s
        self.assertGreater(8192 / 131.4, 60)

    def test_guard_accepts_own_long_payload_at_its_budget_cap(self):
        guard_path = path.parent.parent / 'guard.py'
        if not guard_path.exists():
            self.skipTest('guard.py not present next to optimization/')
        guard_spec = importlib.util.spec_from_file_location('ops_live_guard_fixture', guard_path)
        guard = importlib.util.module_from_spec(guard_spec)
        guard_spec.loader.exec_module(guard)
        for stream in (True, False):
            guard.validate_payload({**self.ns['LONG_PAYLOAD'], 'stream': stream})
        with self.assertRaises(ValueError):
            guard.validate_payload({**self.ns['LONG_PAYLOAD'], 'max_tokens': self.ns['OWN_MAX_TOKENS'] + 1})

    def test_all_cases_pass_in_order_and_nothing_private_escapes(self):
        report = self.collect()
        self.assertTrue(report['pass'], [r for r in report['rows'] if not r['pass']])
        self.assertEqual([row['case'] for row in report['rows']],
                         ['unauthenticated-burst-ingress-429', 'body-over-2MiB-413', 'body-over-2MiB-413',
                          'own-sse-stream-started', 'busy-429-nonstream', 'busy-429-stream-before-sse',
                          'sse-client-close-releases-slot', 'nonstream-client-close-releases-slot'])
        self.assertEqual({row['base'] for row in report['rows'] if row['case'] == 'body-over-2MiB-413'},
                         {'http://127.0.0.1:1919', 'https://llm.example.com'})
        dumped = json.dumps(report)
        self.assertNotIn(KEY, dumped)
        self.assertNotIn('PRIVATE_ANSWER', dumped)
        self.assertNotIn('already active', dumped)
        self.assertEqual(report['external_e2e'], 'NOT_RUN')

    def test_nonstream_cancel_requires_observed_busy(self):
        self.nonstream_admitted = False  # own nonstream request never became active
        report = self.collect()
        row = next(r for r in report['rows'] if r['case'] == 'nonstream-client-close-releases-slot')
        self.assertFalse(row['pass'])
        self.assertFalse(row['busy_while_own_nonstream_active'])
        self.assertFalse(report['pass'])

    def test_wrong_413_status_or_transport_failure_fails(self):
        for status in (200, 400, 502, None):
            self.oversize_status = status
            report = self.collect()
            rows = [row for row in report['rows'] if row['case'] == 'body-over-2MiB-413']
            self.assertTrue(rows and all(not row['pass'] for row in rows))
            self.assertFalse(report['pass'])
        def broken(*args, **kwargs):
            raise ConnectionResetError('PRIVATE_TRANSPORT_DETAIL')
        self.ns['raw_exchange'] = broken
        report = self.collect()
        self.assertFalse(report['pass'])
        self.assertNotIn('PRIVATE_TRANSPORT_DETAIL', json.dumps(report))

    def test_queued_or_unlabelled_busy_response_fails(self):
        self.busy_status = 200
        report = self.collect()
        self.assertFalse(next(r for r in report['rows'] if r['case'] == 'busy-429-nonstream')['pass'])
        self.busy_status = 429
        self.busy_headers = {}
        report = self.collect()
        self.assertFalse(next(r for r in report['rows'] if r['case'] == 'busy-429-nonstream')['pass'])

    def test_busy_sse_instead_of_json_fails(self):
        def call(key, base, path, body=None, auth=True, timeout=30):
            if auth and body and body.get('max_tokens') == 8 and body.get('stream'):
                return Reply(429, b'data: {}\n\n', 'text/event-stream', {'Retry-After': '2'})
            return self.call(key, base, path, body, auth, timeout)
        self.ns['call'] = call
        report = self.collect()
        self.assertFalse(next(r for r in report['rows'] if r['case'] == 'busy-429-stream-before-sse')['pass'])

    def test_slot_never_released_fails_bounded(self):
        self.release_after = 10 ** 6
        report = self.collect()
        row = next(r for r in report['rows'] if r['case'] == 'sse-client-close-releases-slot')
        self.assertFalse(row['pass'])
        self.assertLess(row['attempts'], 100)

    def test_burst_requires_ingress_429_and_only_401_or_429(self):
        for template in ([401] * 40, [401] * 10 + [502] + [429] * 29, [429] * 40):
            self.burst_template = template
            row = next(r for r in self.collect()['rows'] if r['case'].startswith('unauthenticated'))
            self.assertFalse(row['pass'], template)

    def test_stream_not_started_fails_whole_report(self):
        def failing(*args, **kwargs):
            raise ValueError('own_stream_not_started')
        self.ns['open_stream'] = failing
        report = self.collect()
        self.assertFalse(report['pass'])
        self.assertFalse(any(r['case'].startswith('busy-429') for r in report['rows']))

    def test_remote_raw_head_never_logged_and_tls_verified(self):
        ns = {}
        exec(compile(m.REMOTE, 'synthetic-remote-overload', 'exec'), ns)
        self.assertEqual(ns['CONTEXT'].verify_mode, __import__('ssl').CERT_REQUIRED)
        self.assertTrue(ns['CONTEXT'].check_hostname)
        self.assertIsNone(ns['NoRedirect']().redirect_request(None, None, 307, '', {}, 'https://third-party.invalid/'))


class RawExchangeLoopbackTests(unittest.TestCase):
    """Own in-process loopback server on an ephemeral port; never the service port 1919."""

    def serve(self, behaviour):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        seen = {}
        def run():
            conn, _ = listener.accept()
            conn.settimeout(10)
            try:
                head = b''
                while b'\r\n\r\n' not in head:
                    data = conn.recv(4096)
                    if not data:
                        break
                    head += data
                seen['head_has_bearer'] = b'Authorization: Bearer ' in head
                length = int([l.split(b':', 1)[1] for l in head.split(b'\r\n') if l.lower().startswith(b'content-length:')][0])
                seen['declared'] = length
                behaviour(conn, head.split(b'\r\n\r\n', 1)[1], length, seen)
            finally:
                conn.close()
                listener.close()
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        ns = {}
        exec(compile(m.REMOTE, 'synthetic-remote-overload', 'exec'), ns)
        return ns, 'http://127.0.0.1:%d' % port, thread, seen

    def test_early_413_without_reading_body_is_observed(self):
        def early(conn, already, length, seen):
            conn.sendall(b'HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
        ns, base, thread, seen = self.serve(early)
        result = ns['raw_exchange'](KEY, base, 2 * 1024 * 1024 + 1, timeout=10)
        thread.join(10)
        self.assertEqual(result['status'], 413)
        self.assertEqual(seen['declared'], 2 * 1024 * 1024 + 1)
        self.assertNotIn(KEY, json.dumps(result))

    def test_413_after_full_body_read_is_observed(self):
        def drain(conn, already, length, seen):
            got = len(already)
            while got < length:
                data = conn.recv(65536)
                if not data:
                    break
                got += len(data)
            seen['received'] = got
            conn.sendall(b'HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\n\r\n')
        ns, base, thread, seen = self.serve(drain)
        result = ns['raw_exchange'](KEY, base, 2 * 1024 * 1024 + 1, timeout=20)
        thread.join(10)
        self.assertEqual(result['status'], 413)
        self.assertEqual(seen['received'], 2 * 1024 * 1024 + 1)

    def test_silent_server_is_not_a_pass(self):
        def silent(conn, already, length, seen):
            time.sleep(3)
        ns, base, thread, seen = self.serve(silent)
        result = ns['raw_exchange'](KEY, base, 2 * 1024 * 1024 + 1, timeout=2)
        thread.join(10)
        self.assertIsNone(result['status'])

    def test_nonstream_hold_sends_body_then_client_closes(self):
        def read_until_close(conn, already, length, seen):
            got = len(already)
            while True:
                data = conn.recv(65536)
                if not data:
                    break
                got += len(data)
            seen['received'] = got
            seen['client_closed'] = True
        ns, base, thread, seen = self.serve(read_until_close)
        ns['pause'] = lambda seconds: None
        result = ns['raw_exchange'](KEY, base, 0, payload={'model': 'qwen36-35b', 'stream': False}, hold_seconds=0)
        thread.join(10)
        self.assertTrue(result['closed_by_client'] and seen['client_closed'])
        self.assertEqual(seen['received'], seen['declared'])
        self.assertTrue(seen['head_has_bearer'])


class OverloadWrapperTests(unittest.TestCase):
    def test_refuses_when_not_idle_and_never_contacts_edge(self):
        remote = mock.Mock()
        report = m.overload(KEY, 'main128', json_get=healthy(active=1), states=units(), selected=lambda: 'main128', remote=remote)
        self.assertFalse(report['pass'])
        self.assertEqual(report['scope'], 'SERVICE_NOT_IDLE')
        remote.assert_not_called()

    def test_identity_change_after_overload_fails(self):
        instances = iter(['inst-1', 'inst-1', 'inst-2', 'inst-2'])
        def json_get(key, path):
            return healthy(next(instances))(key, path)
        report = m.overload(KEY, 'main128', json_get=json_get, states=units(), selected=lambda: 'main128',
                            remote=lambda key: {'pass': True, 'rows': [{'pass': True}]})
        self.assertFalse(report['pass'])
        self.assertFalse(report['same_instance_after'])

    def test_remote_failure_is_sanitized(self):
        def remote(key):
            raise RuntimeError('PRIVATE_SSH_DETAIL ' + key)
        report = m.overload(KEY, 'main128', json_get=healthy(), states=units(), selected=lambda: 'main128', remote=remote)
        self.assertFalse(report['pass'])
        self.assertNotIn('PRIVATE_SSH_DETAIL', json.dumps(report))
        self.assertNotIn(KEY, json.dumps(report))

    def test_ssh_key_only_on_stdin_with_strict_host_keys(self):
        seen = {}
        def run(argv, **kwargs):
            seen.update(argv=argv, kwargs=kwargs)
            return types.SimpleNamespace(returncode=0, stdout=json.dumps({'pass': True, 'rows': []}))
        m.remote_overload(KEY, run=run)
        self.assertNotIn(KEY, ' '.join(seen['argv']))
        self.assertEqual(seen['kwargs']['input'], KEY)
        self.assertIn('StrictHostKeyChecking=yes', seen['argv'])
        self.assertIn('BatchMode=yes', seen['argv'])
        bad = lambda argv, **kwargs: types.SimpleNamespace(returncode=0, stdout=json.dumps({'pass': 'yes'}))
        with self.assertRaises(m.Refused):
            m.remote_overload(KEY, run=bad)

    def test_remote_command_fits_windows_command_line(self):
        # ssh.exe is a Windows binary started from WSL: its whole command line
        # (including the quoted remote program) must stay below 32767 chars.
        seen = {}
        def run(argv, **kwargs):
            seen['argv'] = argv
            return types.SimpleNamespace(returncode=0, stdout=json.dumps({'pass': True, 'rows': []}))
        m.remote_overload(KEY, run=run)
        self.assertLess(len(subprocess.list2cmdline(seen['argv'])), 30000)


class KeepaliveGoneTests(unittest.TestCase):
    def inspector(self, alive=False):
        calls = []
        def verify(generation, lease, expect, wait, root):
            calls.append((generation, lease, expect, wait))
            return {'pass': not alive, 'exact_identity_alive': alive, 'linux_pid': 77}
        return types.SimpleNamespace(private=lambda path, directory: None, verify=verify, calls=calls)

    def write(self, root, generation=GEN, lease=LEASE, state='exited', reason='owner_state_removed'):
        directory = root / generation
        directory.mkdir(parents=True, exist_ok=True)
        record = {'generation': generation, 'lease_id': lease, 'linux_pid': 77, 'linux_starttime_ticks': 5,
                  'state': state, 'reason': reason, 'started_utc': 's', 'finished_utc': 'f'}
        (directory / '77-5.json').write_text(json.dumps(record))

    def test_gone_helpers_pass_with_sanitized_reason(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            self.write(root)
            inspector = self.inspector()
            report = m.keepalive_gone([GEN], 0, root, inspector)
            self.assertTrue(report['pass'])
            self.assertEqual(report['rows'][0]['exit_reason'], 'owner_state_removed')
            self.assertEqual(inspector.calls, [(GEN, LEASE, 'gone', 0)])
            self.assertEqual(report['signals_sent'], 0)

    def test_alive_missing_or_invalid_receipts_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            self.write(root)
            self.assertFalse(m.keepalive_gone([GEN], 0, root, self.inspector(alive=True))['pass'])
            self.assertFalse(m.keepalive_gone(['c' * 32], 0, root, self.inspector())['pass'])
            self.write(root, generation='d' * 32, lease='not-a-lease')
            self.assertFalse(m.keepalive_gone(['d' * 32], 0, root, self.inspector())['pass'])
            with self.assertRaises(m.Refused):
                m.keepalive_gone(['../../etc'], 0, root, self.inspector())
            with self.assertRaises(m.Refused):
                m.keepalive_gone([], 0, root, self.inspector())

    def test_generations_from_ops_report(self):
        with tempfile.TemporaryDirectory() as temp:
            report = pathlib.Path(temp) / 'ops.json'
            report.write_text('﻿' + json.dumps({'FinalProcesses': [{'Generation': GEN}, {'Generation': GEN},
                                                                          {'Generation': 'e' * 32}, {'Generation': 'bad'}]}), encoding='utf-8')
            self.assertEqual(m.generations_from_ops_report(report), [GEN, 'e' * 32])
            report.write_text(json.dumps({'FinalProcesses': []}))
            with self.assertRaises(m.Refused):
                m.generations_from_ops_report(report)


class CliTests(unittest.TestCase):
    def test_default_is_plan_only_without_key_network_or_report(self):
        with mock.patch.object(m, 'load_key', side_effect=AssertionError('key read')), \
             mock.patch.object(socket, 'create_connection', side_effect=AssertionError('socket')), \
             mock.patch.object(subprocess, 'run', side_effect=AssertionError('subprocess')), \
             mock.patch.object(m, 'write_report', side_effect=AssertionError('report')), \
             mock.patch('sys.stdout', new_callable=io.StringIO) as out:
            self.assertEqual(m.main(['identity', '--profile', 'main128']), 0)
            self.assertEqual(m.main([]), 0)
        self.assertEqual(json.loads(out.getvalue().splitlines()[0])['result'], 'PLAN_ONLY')

    def test_report_is_exclusive_and_named_safely(self):
        with tempfile.TemporaryDirectory() as temp:
            target = m.write_report({'pass': True}, 'identity', report_dir=temp, name='fixture-1')
            self.assertTrue(target.exists())
            with self.assertRaises(FileExistsError):
                m.write_report({'pass': True}, 'identity', report_dir=temp, name='fixture-1')
            with self.assertRaises(m.Refused):
                m.write_report({'pass': True}, 'identity', report_dir=temp, name='../escape')


if __name__ == '__main__':
    unittest.main(verbosity=2)
