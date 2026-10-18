"""Admission queue and per-user keys of serving/guard.py, exercised as ASGI without a model.

Covers QWEN36_QUEUE_DEPTH/QWEN36_QUEUE_WAIT (FIFO, full queue, wait timeout,
disconnect, shared MAX_SECONDS budget, cleanup failure), the depth-0 default that
keeps the historical 429, and secrets/users.env reload/fail-closed rules.
"""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

SERVING = Path(__file__).resolve().parent.parent / 'serving'
OWNER = 'o' * 43
ALICE = 'A' * 43
BOB = 'B-' * 22


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SERVING / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


os.environ['QWEN36_API_KEY'] = OWNER
os.environ.pop('QWEN36_DATA_ROOT', None)
guard = load('guard', 'guard.py')
ingress = load('ingress', 'ingress.py')
BODY = json.dumps({'model': 'qwen36-35b', 'messages': [{'role': 'user', 'content': 'x'}], 'max_tokens': 4}).encode()


class App:
    """Generation app whose requests finish only when the test releases them."""
    def __init__(self):
        self.started = []
        self.release = {}
        self.cancelled = []

    async def __call__(self, scope, receive, send):
        await receive()
        tag = dict(scope['headers']).get(b'x-tag', b'').decode()
        if scope['path'] == '/v1/chat/completions':
            self.release[tag] = asyncio.Event()
            self.started.append(tag)
            try: await self.release[tag].wait()
            except asyncio.CancelledError:
                self.cancelled.append(tag)
                raise
        await send({'type': 'http.response.start', 'status': 200, 'headers': []})
        await send({'type': 'http.response.body', 'body': b'{}'})


async def call(asgi, tag='', key=OWNER, disconnect=None, path='/v1/chat/completions', method='POST'):
    events, first = [], True

    async def receive():
        nonlocal first
        if first:
            first = False
            return {'type': 'http.request', 'body': BODY if method == 'POST' else b''}
        if disconnect is not None: await disconnect.wait()
        else: await asyncio.sleep(3600)
        return {'type': 'http.disconnect'}

    async def send(event): events.append(event)
    headers = [(b'authorization', b'Bearer ' + key.encode()), (b'x-tag', tag.encode())]
    await asyncio.wait_for(asyncio.shield(asgi({'type': 'http', 'method': method, 'path': path, 'headers': headers},
                                               receive, send)), 30)
    start = next((e for e in events if e['type'] == 'http.response.start'), None)
    body = b''.join(e.get('body', b'') for e in events if e['type'] == 'http.response.body')
    return (start['status'] if start else None), (json.loads(body)['error']['message'] if start and start['status'] >= 400 else None)


async def until(condition, seconds=5):
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline: raise AssertionError('condition not reached')
        await asyncio.sleep(.01)


class QueueTests(unittest.TestCase):
    def setUp(self):
        guard._KEYRING = None
        os.environ.pop('QWEN36_DATA_ROOT', None)

    def test_default_depth_zero_keeps_immediate_429(self):
        async def scenario():
            app = App()
            asgi = guard.APIGuard(app)
            self.assertEqual((asgi.queue_depth, asgi.queue_wait), (0, 600))
            first = asyncio.create_task(call(asgi, 'a'))
            await until(lambda: app.started == ['a'])
            self.assertEqual(await call(asgi, 'b'), (429, 'One inference request is already active; no waiting queue'))
            app.release['a'].set()
            self.assertEqual(await first, (200, None))
            self.assertFalse(asgi.busy)
        asyncio.run(scenario())

    def test_fifo_order_and_full_queue(self):
        async def scenario():
            app = App()
            asgi = guard.APIGuard(app, queue_depth=2, queue_wait=60)
            tasks = [asyncio.create_task(call(asgi, 'a'))]
            await until(lambda: app.started == ['a'])
            for tag in ('b', 'c'):
                tasks.append(asyncio.create_task(call(asgi, tag)))
                await until(lambda: len(asgi.waiters) == len(tasks) - 1)
            self.assertEqual(await call(asgi, 'd'), (429, 'Inference queue is full'))
            for index, tag in enumerate('abc'):
                await until(lambda: app.started[-1:] == [tag])
                app.release[tag].set()
                self.assertEqual(await tasks[index], (200, None))
            self.assertEqual(app.started, ['a', 'b', 'c'])
            self.assertFalse(asgi.busy or asgi.waiters)
        asyncio.run(scenario())

    def test_wait_timeout_returns_429(self):
        async def scenario():
            app = App()
            asgi = guard.APIGuard(app, queue_depth=1, queue_wait=1)
            first = asyncio.create_task(call(asgi, 'a'))
            await until(lambda: app.started == ['a'])
            begin = time.monotonic()
            self.assertEqual(await call(asgi, 'b'), (429, 'Inference queue wait exceeded'))
            self.assertGreaterEqual(time.monotonic() - begin, .9)
            self.assertFalse(asgi.waiters)
            app.release['a'].set()
            await first
            self.assertEqual(app.started, ['a'])
        asyncio.run(scenario())

    def test_disconnect_while_waiting_frees_place(self):
        async def scenario():
            app = App()
            asgi = guard.APIGuard(app, queue_depth=2, queue_wait=60)
            first = asyncio.create_task(call(asgi, 'a'))
            await until(lambda: app.started == ['a'])
            gone = asyncio.Event()
            leaver = asyncio.create_task(call(asgi, 'b', disconnect=gone))
            await until(lambda: len(asgi.waiters) == 1)
            stayer = asyncio.create_task(call(asgi, 'c'))
            await until(lambda: len(asgi.waiters) == 2)
            gone.set()
            self.assertEqual(await leaver, (None, None))
            self.assertEqual(len(asgi.waiters), 1)
            app.release['a'].set()
            await first
            await until(lambda: app.started == ['a', 'c'])
            app.release['c'].set()
            self.assertEqual(await stayer, (200, None))
        asyncio.run(scenario())

    def test_disconnect_after_admission_still_cancels_app(self):
        async def scenario():
            app = App()
            asgi = guard.APIGuard(app, queue_depth=1, queue_wait=60)
            first = asyncio.create_task(call(asgi, 'a'))
            await until(lambda: app.started == ['a'])
            gone = asyncio.Event()
            queued = asyncio.create_task(call(asgi, 'b', disconnect=gone))
            await until(lambda: len(asgi.waiters) == 1)
            app.release['a'].set()
            await first
            await until(lambda: app.started == ['a', 'b'])
            gone.set()
            await queued
            self.assertEqual(app.cancelled, ['b'])
            self.assertFalse(asgi.busy)
        asyncio.run(scenario())

    def test_queue_wait_counts_toward_max_seconds(self):
        async def scenario():
            app = App()
            asgi = guard.APIGuard(app, queue_depth=1, queue_wait=60)
            saved = guard.MAX_SECONDS
            guard.MAX_SECONDS = 2
            try:
                first = asyncio.create_task(call(asgi, 'a'))
                await until(lambda: app.started == ['a'])
                second = asyncio.create_task(call(asgi, 'b'))
                await asyncio.sleep(1.2)
                app.release['a'].set()
                await first
                begin = time.monotonic()
                self.assertEqual(await second, (504, 'Maximum request duration exceeded'))
                self.assertLess(time.monotonic() - begin, 1.5)
            finally:
                guard.MAX_SECONDS = saved
        asyncio.run(scenario())

    def test_engine_still_active_without_holder_is_polled(self):
        async def scenario():
            app = App()
            stats = SimpleNamespace(active=1, inflight_uids=set())
            asgi = guard.APIGuard(app, state_getter=lambda: SimpleNamespace(stats=stats), queue_depth=1, queue_wait=60)
            waiting = asyncio.create_task(call(asgi, 'a'))
            await until(lambda: len(asgi.waiters) == 1)
            await asyncio.sleep(.3)
            self.assertEqual(app.started, [])
            stats.active = 0
            await until(lambda: app.started == ['a'])
            app.release['a'].set()
            self.assertEqual(await waiting, (200, None))
        asyncio.run(scenario())

    def test_cleanup_failure_rejects_waiters(self):
        async def scenario():
            app = App()
            asgi = guard.APIGuard(app, queue_depth=2, queue_wait=60)
            asgi.busy = True
            waiting = asyncio.create_task(call(asgi, 'a'))
            await until(lambda: len(asgi.waiters) == 1)
            asgi.cleanup_failed = True
            asgi.dispatch()
            self.assertEqual(await waiting, (503, 'Protocol cleanup failed; service recovery pending'))
            self.assertEqual(app.started, [])
        asyncio.run(scenario())

    def test_settings_validation(self):
        good = {'': None, '0': 0, '4': 4, '6': 6}
        for raw, value in good.items():
            depth, wait = guard.queue_settings({'QWEN36_QUEUE_DEPTH': raw, 'QWEN36_QUEUE_WAIT': '600'})
            self.assertEqual(depth, 0 if value is None else value)
            self.assertEqual(wait, 600)
        self.assertEqual(guard.queue_settings({}), (0, 600))
        self.assertEqual(guard.queue_settings({'QWEN36_QUEUE_WAIT': '840'})[1], 840)
        for raw in ('7', '-1', 'x', '4.0', ' 4', 'пј”', '00000'):
            with self.assertRaises(RuntimeError): guard.queue_settings({'QWEN36_QUEUE_DEPTH': raw})
        for raw in ('0', '841', '1e3'):
            with self.assertRaises(RuntimeError): guard.queue_settings({'QWEN36_QUEUE_WAIT': raw})

    def test_profile_runtime_preserves_only_queue_settings(self):
        runtime = load('profile_runtime_for_queue', 'profile-runtime.py')
        self.assertEqual(runtime.PRESERVED_ENV, {'QWEN36_API_KEY', 'QWEN36_QUEUE_DEPTH', 'QWEN36_QUEUE_WAIT'})


@unittest.skipUnless(os.name == 'posix', 'users.env needs POSIX owner-only permissions')
class KeyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'secrets').mkdir()
        self.file = self.root / 'secrets' / 'users.env'
        os.environ['QWEN36_DATA_ROOT'] = str(self.root)
        guard._KEYRING = None

    def tearDown(self):
        os.environ.pop('QWEN36_DATA_ROOT', None)
        guard._KEYRING = None
        self.temp.cleanup()

    def write(self, text, mode=0o600):
        temp = self.file.with_suffix('.tmp')
        temp.write_text(text)
        os.chmod(temp, mode)
        os.replace(temp, self.file)

    def status(self, asgi, key):
        return asyncio.run(call(asgi, key=key, path='/v1/models', method='GET'))[0]

    def test_user_keys_issue_and_revoke_without_restart(self):
        asgi = guard.APIGuard(App())
        self.assertEqual(self.status(asgi, ALICE), 401)
        self.write('# comment\n\nalice=' + ALICE + '\nbob=' + BOB + '\n')
        self.assertEqual([self.status(asgi, key) for key in (OWNER, ALICE, BOB, 'C' * 43)], [200, 200, 200, 401])
        self.write('\ufeffalice=' + ALICE + '\r\n\r\n')
        self.assertEqual(self.status(asgi, ALICE), 200)
        self.write('alice=' + ALICE + '\nbob=' + BOB + '\n')
        self.assertEqual(guard.keyring().lookup(b'Bearer ' + BOB.encode()), 'bob')
        self.write('bob=' + BOB + '\n')
        self.assertEqual([self.status(asgi, key) for key in (ALICE, BOB)], [401, 200])
        self.file.unlink()
        self.assertEqual([self.status(asgi, key) for key in (OWNER, BOB)], [200, 401])

    def test_ingress_accepts_user_keys(self):
        self.write('alice=' + ALICE + '\n')
        asgi = ingress.Ingress(App())
        self.assertEqual([self.status(asgi, key) for key in (OWNER, ALICE, BOB)], [200, 200, 401])

    def test_invalid_files_fail_closed_to_owner(self):
        asgi = guard.APIGuard(App())
        bad = ['alice=' + ALICE + '\nalice=' + BOB, 'alice=' + ALICE + '\nbob=' + ALICE, 'owner=' + ALICE,
               'alice=' + OWNER, 'Alice=' + ALICE, 'alice=short', 'alice ' + ALICE, 'alice=' + ALICE + '!']
        for text in bad:
            self.write(text + '\n')
            self.assertEqual([self.status(asgi, key) for key in (OWNER, ALICE)], [200, 401], text)
        self.write('alice=' + ALICE + '\n')
        self.assertEqual(self.status(asgi, ALICE), 200)

    def test_shared_or_linked_file_is_refused(self):
        asgi = guard.APIGuard(App())
        self.write('alice=' + ALICE + '\n', mode=0o640)
        self.assertEqual(self.status(asgi, ALICE), 401)
        target = self.root / 'elsewhere.env'
        target.write_text('alice=' + ALICE + '\n')
        os.chmod(target, 0o600)
        self.file.unlink()
        self.file.symlink_to(target)
        self.assertEqual([self.status(asgi, key) for key in (OWNER, ALICE)], [200, 401])


if __name__ == '__main__':
    unittest.main(verbosity=2)
