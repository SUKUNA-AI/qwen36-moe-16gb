"""tools/longdoc.py against a fake OpenAI-compatible server on loopback: cutting, map -> reduce order,
hierarchical reduce, context-length refusals, dry run, retries. No GPU, model, key or network."""
import contextlib
import http.server
import importlib.util
import io
import json
import math
import os
import re
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import repo_paths

SOURCE = repo_paths.CHECKOUT / 'tools' / 'longdoc.py'
spec = importlib.util.spec_from_file_location('longdoc', SOURCE)
L = importlib.util.module_from_spec(spec)
sys.modules['longdoc'] = L
spec.loader.exec_module(L)

KEY = 'synthetic-key-1'
FRAGMENT = re.compile(r'=== FRAGMENT (\S+) OF (\d+) ===\n(.{0,12})', re.S)
NOTES = re.compile(r'=== NOTES \(([^)]*)\) ===')


def default_reply(payload):
    user = payload['messages'][1]['content']
    m = FRAGMENT.search(user)
    if m:
        return f'NOTE[{m.group(1)}] ' + m.group(3).replace('\n', ' ')
    if '=== DOCUMENT ===' in user:
        return 'SINGLE-ANSWER'
    if '=== NOTES (' in user:
        final = L.INSTR_REDUCE_QA in user or L.INSTR_REDUCE_SUMMARY in user
        return ('FINAL-ANSWER[' if final else 'MERGED[') + '|'.join(NOTES.findall(user)) + ']'
    return 'UNKNOWN'


class Fake:
    """Records every request; `faults` maps a 1-based request number to a failure kind."""

    def __init__(self, density=1.0, window=10 ** 9, key=KEY, faults=None, replier=default_reply):
        self.density, self.window, self.key = density, window, key
        self.faults, self.replier = faults or {}, replier
        self.log = []
        self.lock = threading.Lock()
        self.count = self.active = self.max_active = 0
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'

            def log_message(self, *args):
                pass

            def reply_json(self, status, obj, headers=()):
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                for k, v in headers:
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
                with fake.lock:
                    fake.count += 1
                    number = fake.count
                    fake.active += 1
                    fake.max_active = max(fake.max_active, fake.active)
                try:
                    self.serve(number, raw)
                finally:
                    with fake.lock:
                        fake.active -= 1

            def serve(self, number, raw):
                entry = {'n': number, 'path': self.path, 'auth': self.headers.get('Authorization'),
                         'payload': json.loads(raw), 'status': None, 'body_bytes': len(raw)}
                fake.log.append(entry)

                def status(code, obj, headers=()):
                    entry['status'] = code
                    self.reply_json(code, obj, headers)

                if self.path != '/v1/chat/completions':
                    return status(404, {'error': {'message': 'Route unavailable'}})
                if entry['auth'] != 'Bearer ' + fake.key:
                    return status(401, {'error': {'message': 'Invalid or missing Bearer token'}})
                fault = fake.faults.get(number)
                if fault == 'drop':
                    entry['status'] = 'dropped'
                    return
                if fault == 'busy':
                    return status(429, {'error': {'message': 'One inference request is already active'}},
                                  [('Retry-After', '0')])
                if fault == 'http500':
                    return status(500, {'error': {'message': 'engine fault'}})
                if fault == 'bad400':
                    return status(400, {'error': {'message': 'Only n=1 is supported'}})
                if fault == 'redirect':
                    return status(302, {}, [('Location', 'http://127.0.0.1:1/v1/chat/completions')])
                payload = entry['payload']
                chars = sum(len(m['content']) for m in payload['messages'])
                tokens = math.ceil(chars / fake.density) + 8
                if tokens + payload['max_tokens'] > fake.window:
                    return status(400, {'error': {
                        'message': f"prompt ({tokens}) + requested output ({payload['max_tokens']}) "
                                   f'exceeds context limit {fake.window}',
                        'type': 'invalid_request_error', 'param': None, 'code': 'context_length_exceeded'}})
                reply = fake.replier(payload)
                usage = {'prompt_tokens': tokens, 'completion_tokens': len(reply), 'total_tokens': tokens + len(reply)}
                if not payload.get('stream'):
                    return status(200, {'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': reply},
                                                     'finish_reason': 'stop'}], 'usage': usage})
                entry['status'] = 200
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()

                def sse(obj):
                    self.wfile.write(b'data: ' + json.dumps(obj).encode() + b'\n\n')
                    self.wfile.flush()

                sse({'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})
                third = max(1, len(reply) // 3)
                pieces = [reply[:third], reply[third:2 * third], reply[2 * third:]]
                sse({'choices': [{'index': 0, 'delta': {'reasoning_content': 'thinking is ignored'}, 'finish_reason': None}]})
                for piece in pieces[:2] if fault == 'truncate' else pieces:
                    sse({'choices': [{'index': 0, 'delta': {'content': piece}, 'finish_reason': None}]})
                if fault == 'truncate':
                    return
                sse({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
                if (payload.get('stream_options') or {}).get('include_usage'):
                    sse({'choices': [], 'usage': usage})
                self.wfile.write(b'data: [DONE]\n\n')

        self.httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.base_url = f'http://127.0.0.1:{self.httpd.server_address[1]}/v1'

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    def maps(self):
        return [e for e in self.log if e['status'] == 200 and FRAGMENT.search(e['payload']['messages'][1]['content'])]

    def reduces(self):
        return [e for e in self.log if e['status'] == 200 and '=== NOTES (' in e['payload']['messages'][1]['content']]

    def user(self, entry):
        return entry['payload']['messages'][1]['content']


def para(i):
    return f'P{i:02d} Alpha beta gamma. Delta epsilon zeta. Eta theta iota.'


def document(n):
    return '\n\n'.join(para(i) for i in range(n))


# one character is one token for the estimate, so sizes below are plain character counts
STD = ['--chars-per-token', '1', '--context-window', '1000000', '--max-tokens', '64', '--retry-delay', '0',
       '--busy-wait', '5', '--timeout', '10']


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def write(self, text, name='doc.txt'):
        path = self.dir / name
        path.write_bytes(text.encode('utf-8'))
        return str(path)

    def cli(self, fake, argv, env=None, stdin=None):
        out, err = io.StringIO(), io.StringIO()
        environ = {'QWEN36_API_KEY': fake.key}
        environ.update(env or {})
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.dict(os.environ, environ))
        stack.enter_context(contextlib.redirect_stdout(out))
        stack.enter_context(contextlib.redirect_stderr(err))
        if stdin is not None:
            stack.enter_context(mock.patch.object(sys, 'stdin', types.SimpleNamespace(
                buffer=io.BytesIO(stdin.encode('utf-8')), isatty=lambda: False)))
        with stack:
            try:
                rc = L.main(['--base-url', fake.base_url] + argv)
            except SystemExit as exc:
                rc = exc.code
        return rc, out.getvalue(), err.getvalue()


class Chunker(unittest.TestCase):
    def test_paragraph_boundaries_overlap_and_limit(self):
        text = document(60)
        chunks = L.plan_chunks(text, 400, 80)
        self.assertGreater(len(chunks), 5)
        self.assertEqual((chunks[0].start, chunks[-1].end), (0, len(text)))
        for prev, cur in zip(chunks, chunks[1:]):
            self.assertLessEqual(cur.start, prev.end)                 # no gap
            self.assertGreater(cur.start, prev.start)                 # progress
            self.assertGreater(cur.end, prev.end)
            self.assertTrue(0 < prev.end - cur.start <= 80, (prev, cur))
            self.assertEqual(cur.overlap, prev.end - cur.start)
        for c in chunks:
            self.assertLessEqual(c.length, 400)
            self.assertTrue(text[c.start:].startswith('P'), c)        # starts at a paragraph
            if c.end < len(text):
                self.assertTrue(text[c.end:].startswith('P'), c)      # ends at a paragraph border
        self.assertEqual(chunks[0].overlap, 0)

    def test_without_overlap_the_chunks_tile_the_text(self):
        text = document(40)
        chunks = L.plan_chunks(text, 300, 0)
        self.assertEqual(''.join(text[c.start:c.end] for c in chunks), text)
        self.assertTrue(all(c.overlap == 0 for c in chunks))

    def test_sentence_boundaries_when_there_are_no_paragraphs(self):
        text = ' '.join(f'Sentence number {i} is short.' for i in range(300))
        chunks = L.plan_chunks(text, 1000, 200)
        self.assertGreater(len(chunks), 5)
        for c in chunks:
            self.assertLessEqual(c.length, 1000)
            if c.end < len(text):
                self.assertTrue(text[:c.end].endswith('. '), repr(text[c.end - 6:c.end]))
            if c.index > 1:
                self.assertTrue(text[c.start:].startswith('Sentence'), c)
                self.assertTrue(0 < c.overlap <= 200)

    def test_hard_cut_only_for_text_without_any_space(self):
        chunks = L.plan_chunks('x' * 5000, 1000, 200)
        self.assertEqual([(c.start, c.end) for c in chunks], [(i * 1000, (i + 1) * 1000) for i in range(5)])

    def test_russian_text_and_token_limit(self):
        text = '\n\n'.join(f'Абзац {i}. Это первое предложение. А это второе, подлиннее. И третье!' for i in range(200))
        cpt = 2.8
        chunk_tokens, overlap_tokens = 500, 40
        chunks = L.plan_chunks(text, int(chunk_tokens * cpt), int(overlap_tokens * cpt))
        self.assertTrue(all(L.est_tokens(c.length, cpt) <= chunk_tokens for c in chunks))
        self.assertEqual(chunks[-1].end, len(text))
        self.assertGreater(len(chunks), 5)

    def test_short_text_is_one_chunk(self):
        chunks = L.plan_chunks('short text', 1000, 100)
        self.assertEqual([(c.start, c.end) for c in chunks], [(0, 10)])

    def test_split_half_covers_the_text_and_shrinks(self):
        text = document(30)
        first, second = L.split_half(text, 100)
        self.assertTrue(text.startswith(first) and text.endswith(second))
        self.assertLess(len(first), len(text))
        self.assertLess(len(second), len(text))
        self.assertGreaterEqual(len(first) + len(second), len(text))   # overlap only adds
        self.assertTrue(second.startswith('P'))

    def test_group_notes_has_at_least_two_per_group(self):
        groups = L.group_notes(list(range(10)), 5, lambda item: 100)
        self.assertTrue(all(len(g) >= 2 for g in groups[:-1]))
        self.assertEqual([x for g in groups for x in g], list(range(10)))

    def test_context_error_is_parsed(self):
        self.assertEqual(L.parse_context_error('prompt (140000) + requested output (2048) exceeds context limit 131072'),
                         (140000, 131072))
        self.assertEqual(L.parse_context_error('boom', 'context_length_exceeded'), (None, None))
        self.assertIsNone(L.parse_context_error('Only n=1 is supported'))


class MapReduce(Base):
    ARGS = STD + ['--reduce-tokens', '10000']

    def test_map_then_one_reduce_in_order_one_request_at_a_time(self):
        text = document(12)
        path = self.write(text)
        chunks = L.plan_chunks(text, 200, 30)
        self.assertGreaterEqual(len(chunks), 3)
        with Fake() as fake:
            rc, out, err = self.cli(fake, [path, '-q', 'Which letters?', '--chunk-tokens', '200',
                                           '--overlap-tokens', '30'] + self.ARGS)
            log = fake.log
            self.assertEqual(rc, 0, err)
            self.assertEqual(len(log), len(chunks) + 1)
            maps, reduces = fake.maps(), fake.reduces()
            self.assertEqual(len(maps), len(chunks))
            self.assertEqual(len(reduces), 1)
            self.assertIs(log[-1], reduces[0])                         # reduce comes last
            labels = [FRAGMENT.search(fake.user(e)).group(1) for e in maps]
            self.assertEqual(labels, [str(i) for i in range(1, len(chunks) + 1)])
            self.assertTrue(all(FRAGMENT.search(fake.user(e)).group(2) == str(len(chunks)) for e in maps))
            self.assertEqual(fake.max_active, 1)                      # strictly sequential
            final = fake.user(reduces[0])
            self.assertIn(L.INSTR_REDUCE_QA, final)
            positions = [final.index(f'=== NOTES (fragment {i}) ===') for i in range(1, len(chunks) + 1)]
            self.assertEqual(positions, sorted(positions))
            self.assertIn('NOTE[1] P00', final)
            self.assertEqual(out, 'FINAL-ANSWER[' + '|'.join(f'fragment {i}' for i in range(1, len(chunks) + 1)) + ']\n')
            self.assertIn(f'map 1/{len(chunks)}', err)                 # progress goes to stderr
            self.assertEqual(out.count('map '), 0)

    def test_request_shape_auth_and_cache_friendly_prefix(self):
        path = self.write(document(12))
        with Fake() as fake:
            rc, _, err = self.cli(fake, [path, '-q', 'Which letters?', '--chunk-tokens', '200',
                                         '--overlap-tokens', '30'] + self.ARGS)
            self.assertEqual(rc, 0, err)
            for e in fake.log:
                p = e['payload']
                self.assertEqual(e['auth'], 'Bearer ' + KEY)
                self.assertEqual(p['model'], 'qwen36-35b')
                self.assertEqual(p['max_tokens'], 64)
                self.assertIs(p['stream'], True)
                self.assertEqual(p['stream_options'], {'include_usage': True})
                self.assertEqual(p['chat_template_kwargs'], {'enable_thinking': False})
                self.assertNotIn('temperature', p)
                self.assertEqual([m['role'] for m in p['messages']], ['system', 'user'])
            systems = {e['payload']['messages'][0]['content'] for e in fake.log}
            self.assertEqual(systems, {L.SYSTEM_PROMPT})
            users = [fake.user(e) for e in fake.maps()]
            head = f'Question: Which letters?\n\n{L.INSTR_MAP_QA}\n\n=== FRAGMENT '
            self.assertTrue(os.path.commonprefix(users).startswith(head))   # identical up to the fragment
            reduce_user = fake.user(fake.reduces()[0])
            self.assertTrue(reduce_user.startswith('Question: Which letters?\n\n'))

    def test_thinking_temperature_and_seed(self):
        path = self.write(document(2))
        with Fake() as fake:
            rc, _, err = self.cli(fake, [path, '-q', 'q', '--thinking', 'on', '--temperature', '0.2', '--seed', '7'] + STD)
            self.assertEqual(rc, 0, err)
            p = fake.log[0]['payload']
            self.assertEqual(p['chat_template_kwargs'], {'enable_thinking': True})
            self.assertEqual((p['temperature'], p['seed']), (0.2, 7))

    def test_short_document_is_a_single_request_without_reduce(self):
        path = self.write(document(2))
        with Fake() as fake:
            rc, out, err = self.cli(fake, [path, '-q', 'q'] + STD)
            self.assertEqual((rc, out), (0, 'SINGLE-ANSWER\n'), err)
            self.assertEqual(fake.count, 1)
            self.assertIn(L.INSTR_SINGLE_QA, fake.user(fake.log[0]))

    def test_no_stream_uses_plain_json(self):
        path = self.write(document(2))
        with Fake() as fake:
            rc, out, err = self.cli(fake, [path, '-q', 'q', '--no-stream'] + STD)
            self.assertEqual((rc, out), (0, 'SINGLE-ANSWER\n'), err)
            self.assertIs(fake.log[0]['payload']['stream'], False)
            self.assertNotIn('stream_options', fake.log[0]['payload'])

    def test_summary_mode_needs_no_question(self):
        path = self.write(document(12))
        with Fake() as fake:
            rc, out, err = self.cli(fake, [path, '--mode', 'summary', '--chunk-tokens', '200',
                                           '--overlap-tokens', '30'] + MapReduce.ARGS)
            self.assertEqual(rc, 0, err)
            self.assertTrue(out.startswith('FINAL-ANSWER['))
            self.assertTrue(all(fake.user(e).startswith('Task: summarize the whole document.') for e in fake.log))
            self.assertIn(L.INSTR_MAP_SUMMARY, fake.user(fake.maps()[0]))
            self.assertIn(L.INSTR_REDUCE_SUMMARY, fake.user(fake.reduces()[0]))

    def test_hierarchical_reduce_when_the_answers_do_not_fit_one_request(self):
        text = document(40)
        path = self.write(text)
        n = len(L.plan_chunks(text, 120, 20))
        self.assertGreater(n, 12)
        with Fake() as fake:
            rc, out, err = self.cli(fake, [path, '-q', 'q', '--chunk-tokens', '120', '--overlap-tokens', '20',
                                           '--reduce-tokens', '90', '--save-parts', str(self.dir / 'parts')] + STD)
            self.assertEqual(rc, 0, err)
            reduces = fake.reduces()
            self.assertGreater(len(reduces), 2)
            self.assertIs(fake.log[-1], reduces[-1])
            finals = [e for e in reduces if L.INSTR_REDUCE_QA in fake.user(e)]
            merges = [e for e in reduces if L.INSTR_MERGE in fake.user(e)]
            self.assertEqual(len(finals), 1)
            self.assertIs(finals[0], reduces[-1])
            self.assertEqual(len(merges), len(reduces) - 1)
            self.assertEqual(len(fake.maps()), n)
            # the last request sees contiguous ranges that cover fragments 1..n and are fewer than n
            titles = NOTES.findall(fake.user(finals[0]))
            self.assertLess(len(titles), n)
            spans = []
            for title in titles:
                nums = re.findall(r'\d+', title)
                spans.append((int(nums[0]), int(nums[-1])))
            self.assertEqual(spans[0][0], 1)
            self.assertEqual(spans[-1][1], n)
            for (a, b), (c, d) in zip(spans, spans[1:]):
                self.assertEqual(c, b + 1)
            # no request except the first level carries raw map notes together with merged ones
            for e in merges:
                self.assertLessEqual(sum(len(x) for x in NOTES.findall(fake.user(e))), 400)
            self.assertTrue(out.startswith('FINAL-ANSWER['))
            saved = sorted(p.name for p in (self.dir / 'parts').iterdir())
            self.assertIn('reduce-L1-G01.md', saved)
            self.assertIn('final.md', saved)

    def test_not_found_everywhere_skips_reduce(self):
        path = self.write(document(12))
        with Fake(replier=lambda payload: 'NOT FOUND in this fragment.') as fake:
            rc, out, err = self.cli(fake, [path, '-q', 'q', '--chunk-tokens', '200', '--overlap-tokens', '30'] + MapReduce.ARGS)
            self.assertEqual(rc, 0, err)
            self.assertEqual(len(fake.reduces()), 0)
            self.assertTrue(out.startswith('NOT FOUND'))

    def test_not_found_notes_are_left_out_of_the_reduce_step(self):
        path = self.write(document(12))

        def reply(payload):
            user = payload['messages'][1]['content']
            m = FRAGMENT.search(user)
            if m and m.group(1) != '2':
                return 'NOT FOUND'
            return default_reply(payload)

        with Fake(replier=reply) as fake:
            rc, out, err = self.cli(fake, [path, '-q', 'q', '--chunk-tokens', '200', '--overlap-tokens', '30'] + MapReduce.ARGS)
            self.assertEqual(rc, 0, err)
            final = fake.user(fake.reduces()[0])
            self.assertEqual(NOTES.findall(final), ['fragment 2'])
            self.assertRegex(final, r'; \d+ of them had nothing relevant')

    def test_save_parts(self):
        text = document(12)
        path = self.write(text)
        chunks = L.plan_chunks(text, 200, 30)
        parts = self.dir / 'parts'
        with Fake() as fake:
            rc, out, err = self.cli(fake, [path, '-q', 'q', '--chunk-tokens', '200', '--overlap-tokens', '30',
                                           '--save-parts', str(parts)] + MapReduce.ARGS)
            self.assertEqual(rc, 0, err)
        names = sorted(p.name for p in parts.iterdir())
        self.assertEqual(names, sorted([f'map-{i:04d}.md' for i in range(1, len(chunks) + 1)] + ['final.md', 'parts.json']))
        self.assertEqual((parts / 'final.md').read_text(encoding='utf-8'), out)
        self.assertTrue((parts / 'map-0001.md').read_text(encoding='utf-8').startswith('NOTE[1]'))
        meta = json.loads((parts / 'parts.json').read_text(encoding='utf-8'))
        self.assertEqual([p['kind'] for p in meta['parts']], ['map'] * len(chunks) + ['final'])
        self.assertTrue(all(p['prompt_tokens'] > 0 for p in meta['parts']))

    def test_output_file_is_utf8_and_stdout_still_gets_the_answer(self):
        path = self.write(document(2))
        target = self.dir / 'answer.txt'
        with Fake(replier=lambda payload: 'Ответ: да') as fake:
            rc, out, err = self.cli(fake, [path, '-q', 'q', '-o', str(target)] + STD)
            self.assertEqual((rc, out), (0, 'Ответ: да\n'), err)
            self.assertEqual(target.read_bytes(), 'Ответ: да\n'.encode('utf-8'))

    def test_empty_final_answer_is_an_error(self):
        path = self.write(document(2))
        with Fake(replier=lambda payload: '') as fake:
            rc, out, err = self.cli(fake, [path, '-q', 'q'] + STD)
            self.assertEqual((rc, out), (1, ''))
            self.assertIn('empty final answer', err)

    def test_stdin_crlf_and_bom(self):
        text = document(2).replace('\n', '\r\n')
        with Fake() as fake:
            rc, out, err = self.cli(fake, ['-q', 'q'] + STD, stdin='﻿' + text)
            self.assertEqual((rc, out), (0, 'SINGLE-ANSWER\n'), err)
            sent = fake.user(fake.log[0])
            self.assertNotIn('\r', sent)
            self.assertNotIn('﻿', sent)
            self.assertIn('P00 Alpha', sent)

    def test_api_key_file_beats_the_environment(self):
        path = self.write(document(2))
        bare = self.write('file-key-bare\n', 'bare.key')
        assign = self.write('# comment\nQWEN36_API_KEY="file-key-assign"\n', 'api.env')
        with Fake(key='file-key-bare') as fake:
            rc, out, err = self.cli(fake, [path, '-q', 'q', '--api-key-file', bare] + STD, env={'QWEN36_API_KEY': 'wrong'})
            self.assertEqual(rc, 0, err)
        with Fake(key='file-key-assign') as fake:
            rc, out, err = self.cli(fake, [path, '-q', 'q', '--api-key-file', assign] + STD, env={'QWEN36_API_KEY': 'wrong'})
            self.assertEqual(rc, 0, err)


class ContextLength(Base):
    def test_refused_document_is_cut_in_half_and_reduced(self):
        text = document(50)                     # about 2950 characters
        path = self.write(text)
        with Fake(density=1.0, window=3000) as fake:
            rc, out, err = self.cli(fake, [path, '-q', 'q', '--chars-per-token', '2', '--chunk-tokens', '2000',
                                           '--context-window', '1000000', '--max-tokens', '64', '--retry-delay', '0',
                                           '--overlap-tokens', '40'])
            self.assertEqual(rc, 0, err)
            statuses = [e['status'] for e in fake.log]
            self.assertEqual(statuses, [400, 200, 200, 200])           # refused once, no retries of the same
            self.assertIn('splitting', err)
            sent = ''.join(fake.user(e) for e in fake.maps())
            for i in range(50):
                self.assertIn(f'P{i:02d} Alpha', sent)                 # nothing was lost
            self.assertEqual(len(fake.reduces()), 1)
            self.assertTrue(out.startswith('FINAL-ANSWER['))
            self.assertIn('context limit', err)                        # learned from the server's message

    def test_the_estimate_is_corrected_so_the_next_chunks_are_not_sent_too_big(self):
        text = '\n\n'.join(para(i) for i in range(100))               # about 6000 characters
        path = self.write(text)
        args = [path, '-q', 'q', '--chars-per-token', '2', '--chunk-tokens', '1500', '--overlap-tokens', '30',
                '--context-window', '100000', '--max-tokens', '64', '--retry-delay', '0', '--reduce-tokens', '10000']
        n = len(L.plan_chunks(text, 3000, 60))
        self.assertGreaterEqual(n, 2)
        with Fake(density=1.0, window=3000) as fake:
            rc, out, err = self.cli(fake, args)
            self.assertEqual(rc, 0, err)
            refused = [e for e in fake.log if e['status'] == 400]
            self.assertEqual(len(refused), 1)                          # only the first chunk was refused
            self.assertGreater(len(fake.maps()), n)                    # and the chunks were halved
            self.assertIn('estimate corrected', err)
            sent = ''.join(fake.user(e) for e in fake.maps())
            for i in range(100):
                self.assertIn(f'P{i:02d} Alpha', sent)

    def test_oversized_body_is_split_before_sending(self):
        path = self.write(document(30))                                # about 1800 characters
        with Fake() as fake:
            rc, out, err = self.cli(fake, [path, '-q', 'q', '--chunk-tokens', '3000', '--max-body-bytes', '2500'] + STD)
            self.assertEqual(rc, 0, err)
            self.assertTrue(all(e['status'] == 200 for e in fake.log))
            self.assertTrue(all(e['body_bytes'] <= 2500 for e in fake.log))
            self.assertEqual(len(fake.maps()), 2)
            self.assertEqual(len(fake.reduces()), 1)

    def test_a_fragment_that_cannot_shrink_is_an_error(self):
        path = self.write(document(2))
        with Fake(window=10) as fake:
            rc, out, err = self.cli(fake, [path, '-q', 'q'] + STD)
            self.assertEqual(rc, 1)
            self.assertIn('error', err)
            self.assertEqual(out, '')


class DryRun(Base):
    def test_plan_without_any_request_and_without_a_key(self):
        text = document(40)
        path = self.write(text)
        chunks = L.plan_chunks(text, 400, 50)
        with Fake() as fake:
            rc, out, err = self.cli(fake, [path, '-q', 'q', '--chunk-tokens', '400', '--overlap-tokens', '50',
                                           '--dry-run'] + STD, env={'QWEN36_API_KEY': ''})
            self.assertEqual(rc, 0, err)
            self.assertEqual(fake.count, 0)
            self.assertRegex(out, rf'chunks:\s+{len(chunks)}\b')
            self.assertIn(f'{len(text):,} characters', out)
            self.assertIn('reduce', out)
            self.assertEqual(len(re.findall(r'^\s+#\d+\s', out, re.M)), len(chunks))
            self.assertEqual(err, '')

    def test_single_chunk_plan(self):
        path = self.write(document(2))
        with Fake() as fake:
            rc, out, err = self.cli(fake, [path, '--dry-run'] + STD)
            self.assertEqual(rc, 0, err)
            self.assertRegex(out, r'chunks:\s+1\b')
            self.assertIn('no reduce step', out)
            self.assertEqual(fake.count, 0)


class Faults(Base):
    def run_single(self, faults, extra=()):
        path = self.write(document(2))
        with Fake(faults=faults) as fake:
            rc, out, err = self.cli(fake, [path, '-q', 'q'] + STD + list(extra))
            return rc, out, err, fake

    def test_retries_network_drops_then_succeeds(self):
        rc, out, err, fake = self.run_single({1: 'drop', 2: 'drop'})
        self.assertEqual((rc, out), (0, 'SINGLE-ANSWER\n'), err)
        self.assertEqual(fake.count, 3)
        self.assertIn('retry 1/3', err)
        self.assertIn('retry 2/3', err)

    def test_gives_up_after_three_retries(self):
        rc, out, err, fake = self.run_single({1: 'drop', 2: 'drop', 3: 'drop', 4: 'drop', 5: 'drop'})
        self.assertEqual(rc, 1)
        self.assertEqual(fake.count, 4)                                # one attempt + three retries
        self.assertIn('after 3 retries', err)
        self.assertEqual(out, '')

    def test_retries_server_errors(self):
        rc, out, err, fake = self.run_single({1: 'http500'})
        self.assertEqual((rc, out, fake.count), (0, 'SINGLE-ANSWER\n', 2), err)

    def test_retries_a_stream_that_ends_early(self):
        rc, out, err, fake = self.run_single({1: 'truncate'})
        self.assertEqual((rc, out, fake.count), (0, 'SINGLE-ANSWER\n', 2), err)

    def test_waits_while_the_server_is_busy(self):
        rc, out, err, fake = self.run_single({1: 'busy', 2: 'busy'})
        self.assertEqual((rc, out, fake.count), (0, 'SINGLE-ANSWER\n', 3), err)
        self.assertIn('busy', err)

    def test_wrong_key_is_a_clear_error_without_retries(self):
        path = self.write(document(2))
        with Fake() as fake:
            rc, out, err = self.cli(fake, [path, '-q', 'q'] + STD, env={'QWEN36_API_KEY': 'wrong-key'})
            self.assertEqual(rc, 1)
            self.assertEqual(fake.count, 1)
            self.assertIn('API key', err)
            self.assertNotIn('wrong-key', err)

    def test_missing_route_points_at_the_base_url(self):
        path = self.write(document(2))
        with Fake() as fake:
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.dict(os.environ, {'QWEN36_API_KEY': KEY}), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = L.main(['--base-url', fake.base_url.replace('/v1', '/api'), path, '-q', 'q'] + STD)
            self.assertEqual(rc, 1)
            self.assertIn('--base-url', err.getvalue())

    def test_other_client_errors_are_shown_and_not_retried(self):
        rc, out, err, fake = self.run_single({1: 'bad400'})
        self.assertEqual((rc, fake.count), (1, 1))
        self.assertIn('Only n=1 is supported', err)

    def test_redirects_are_not_followed(self):
        rc, out, err, fake = self.run_single({1: 'redirect'})
        self.assertEqual((rc, fake.count), (1, 1))
        self.assertIn('redirect', err)

    def test_connection_refused_is_retried_then_reported(self):
        path = self.write(document(2))
        with Fake() as fake:
            dead = fake.base_url
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {'QWEN36_API_KEY': KEY}), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = L.main(['--base-url', dead, path, '-q', 'q', '--retries', '1'] + STD)
        self.assertEqual(rc, 1)
        self.assertIn('after 1 retry', err.getvalue())


class Inputs(Base):
    def test_input_errors(self):
        with Fake() as fake:
            cases = {
                'missing': [str(self.dir / 'nope.txt'), '-q', 'q'],
                'empty': [self.write('  \n\n', 'empty.txt'), '-q', 'q'],
                'binary': [self.write('abc\x00def', 'bin.txt'), '-q', 'q'],
            }
            for name, argv in cases.items():
                rc, out, err = self.cli(fake, argv + STD)
                self.assertEqual(rc, 1, name)
                self.assertIn('error', err, name)
            (self.dir / 'latin.txt').write_bytes('привет'.encode('cp1251'))
            rc, out, err = self.cli(fake, [str(self.dir / 'latin.txt'), '-q', 'q'] + STD)
            self.assertEqual(rc, 1)
            self.assertIn('not valid utf-8', err)
            self.assertEqual(fake.count, 0)

    def test_argument_errors_exit_with_status_two(self):
        path = self.write(document(2))
        with Fake() as fake:
            self.assertEqual(self.cli(fake, [path] + STD)[0], 2)                            # no question
            self.assertEqual(self.cli(fake, [path, '-q', 'q', '--chunk-tokens', '200000'])[0], 2)   # exceeds the window
            self.assertEqual(self.cli(fake, [path, '-q', 'q', '--chunk-tokens', '100', '--overlap-tokens', '60'])[0], 2)
            self.assertEqual(fake.count, 0)


if __name__ == '__main__':
    unittest.main()
