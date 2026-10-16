#!/usr/bin/env python3
"""Map-reduce over a document that does not fit into the model's context window.

The server (Qwen3.6-35B-A3B, profile main128, 131 072-token window) is not changed: this client
cuts the document into pieces of about --chunk-tokens tokens, asks the same question about every
piece (map, one request at a time), and then asks the model to merge the partial answers (reduce,
hierarchical when the partial answers do not fit into one request either).

    python tools/longdoc.py report.md -q "What deadlines does the contract set?" > answer.txt
    python tools/longdoc.py book.txt --mode summary --save-parts parts/ > summary.txt
    cat log.txt | python tools/longdoc.py -q "When did the first error appear?" --thinking on
    python tools/longdoc.py report.md -q "..." --dry-run          # plan only, no requests

Only the Python standard library is used (Python 3.10+, Windows and Linux).

Configuration: --base-url / QWEN36_BASE_URL (default http://127.0.0.1:1919/v1, the local tunnel or
edge loopback; a public name looks like https://llm.example.com/v1), the key from QWEN36_API_KEY or
--api-key-file (a file with the bare key or with a QWEN36_API_KEY=... line).

Token counting. FreeToken has no /tokenize endpoint. It has the Anthropic-style
POST /v1/messages/count_tokens (exact, with the chat template), but the guarded ingress of this
release (serving/guard.py, ALLOWED) admits only /v1/models, /v1/chat/completions and three
diagnostics routes, so the count is not reachable. The client therefore estimates tokens as
characters / --chars-per-token (default 2.8, conservative for mixed Russian/English text) and
corrects itself from what the server reports: when a request is rejected as too long, the error
text carries the exact prompt size and the context limit; the fragment is cut in half, retried,
and the chars-per-token estimate is lowered for the remaining fragments. Real counts of every
answered request (usage.prompt_tokens) are logged to stderr.

The answer goes to stdout, progress and warnings to stderr. Exit status: 0 success, 1 error
(network, server, input), 2 bad arguments, 130 interrupted.
"""
import argparse
import http.client
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_BASE_URL = 'http://127.0.0.1:1919/v1'
DEFAULT_MODEL = 'qwen36-35b'
TEMPLATE_TOKENS = 48       # chat-template tokens around the messages (estimate)
MIN_SPLIT_CHARS = 200      # a fragment shorter than this is never split again
MAX_SPLIT_DEPTH = 12
NOT_FOUND = 'NOT FOUND'
MAX_BODY_BYTES = 2_000_000  # the guarded ingress refuses bodies over 2 MiB (2 097 152)


class LongdocError(Exception):
    """A fatal, user-facing error."""


class SplitError(LongdocError):
    """The server cannot take this request as it is; the caller may send a smaller one."""

    def __init__(self, message, prompt_tokens=None, limit=None, reason='context'):
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.limit = limit
        self.reason = reason


class _Busy(Exception):
    def __init__(self, retry_after):
        super().__init__('server busy')
        self.retry_after = retry_after


class _Transient(Exception):
    """Network or server fault worth retrying."""


# --------------------------------------------------------------------------------------------
# Prompts. System prompt and the head of the user message (task header + instructions) are
# identical for every request of a kind, and the part that differs (fragment label, text) comes
# after them, so repeated requests reuse the server's prefix cache.
# --------------------------------------------------------------------------------------------
SYSTEM_PROMPT = (
    'You are a precise assistant for reading very long documents that are processed one fragment '
    'at a time. Rely only on the text given in the message. Never invent facts, numbers, names or '
    'quotes. Quote evidence verbatim, in double quotes, when it matters. Write the reply in the '
    'language of the question.')

INSTR_MAP_QA = (
    'Instructions: below is ONE fragment of a long document; neighbouring fragments overlap '
    'slightly at the borders. Answer the question using only this fragment. If only a part of '
    'the answer can be here, list the facts of this fragment that are relevant to the question, '
    'each with a short verbatim quote and, when visible, its location (section, heading, page, '
    'line). Do not guess about parts of the document you cannot see. If the fragment contains '
    f'nothing relevant to the question, reply with exactly: {NOT_FOUND}')
INSTR_MAP_SUMMARY = (
    'Instructions: below is ONE fragment of a long document; neighbouring fragments overlap '
    'slightly at the borders. Write a dense summary of this fragment in the order of the text: '
    'key points, facts, names, numbers, dates, decisions, open questions. Keep exact figures and '
    'names. Do not comment on and do not guess about parts of the document you cannot see. If '
    f'the fragment has no content, reply with exactly: {NOT_FOUND}')
INSTR_SINGLE_QA = (
    'Instructions: answer the question using only the document below. Support the answer with '
    'short verbatim quotes. If the document does not contain the answer, say so.')
INSTR_SINGLE_SUMMARY = (
    'Instructions: write a clear, well-structured summary of the document below: key points, '
    'facts, names, numbers, dates, decisions. Keep exact figures and names.')
INSTR_REDUCE_QA = (
    'Instructions: below are notes made independently from consecutive fragments of one long '
    'document, in document order. Neighbouring fragments overlap, so the same fact can appear '
    'twice. Combine the notes into ONE final answer to the question. Use only the notes. Merge '
    'duplicates, keep the important verbatim quotes, and do not fill gaps by guessing. If notes '
    'contradict each other, say so and name the fragments. If the notes do not answer the '
    'question, say that the document does not contain the answer.')
INSTR_REDUCE_SUMMARY = (
    'Instructions: below are summaries of consecutive fragments of one long document, in '
    'document order. Neighbouring fragments overlap, so some content can appear twice. Combine '
    'them into ONE coherent, well-structured summary of the whole document. Use only the '
    'summaries. Merge duplicates, keep exact figures and names.')
INSTR_MERGE = (
    'Instructions: below are notes made independently from consecutive fragments of one long '
    'document, in document order. Neighbouring fragments overlap, so the same fact can appear '
    'twice. Merge them into ONE shorter consolidated set of notes for the later final answer. '
    'Keep every relevant fact, number, name and verbatim quote, remove duplicates, keep the '
    'document order. Do not write the final answer yet.')


def task_header(mode, question):
    if mode == 'summary':
        return 'Task: summarize the whole document.' + (f' Focus on: {question}' if question else '')
    return f'Question: {question}'


def _reminder(mode, question, what):
    if mode == 'summary':
        return f'Now write the summary of {what}.'
    return f'Reminder, the question: {question}'


def map_prompt(mode, question, label, total, text):
    instr = INSTR_MAP_QA if mode == 'qa' else INSTR_MAP_SUMMARY
    return (f'{task_header(mode, question)}\n\n{instr}\n\n'
            f'=== FRAGMENT {label} OF {total} ===\n{text}\n=== END OF FRAGMENT {label} OF {total} ===\n\n'
            f'{_reminder(mode, question, "this fragment")}')


def single_prompt(mode, question, text):
    instr = INSTR_SINGLE_QA if mode == 'qa' else INSTR_SINGLE_SUMMARY
    return (f'{task_header(mode, question)}\n\n{instr}\n\n'
            f'=== DOCUMENT ===\n{text}\n=== END OF DOCUMENT ===\n\n'
            f'{_reminder(mode, question, "the document")}')


@dataclass
class Note:
    title: str   # 'fragment 3', 'fragment 3.1', 'fragments 1..8'
    label: str   # '3', '3.1', '1..8'
    text: str


def reduce_prompt(mode, question, notes, final, total, omitted):
    if not final:
        instr = INSTR_MERGE
    else:
        instr = INSTR_REDUCE_QA if mode == 'qa' else INSTR_REDUCE_SUMMARY
    info = f'The document was split into {total} fragments'
    info += (f'; {omitted} of them had nothing relevant and are left out.' if omitted else '.')
    parts = [task_header(mode, question), '', instr, '', info, '']
    for note in notes:
        parts += [f'=== NOTES ({note.title}) ===', note.text, '=== END OF NOTES ===', '']
    if final:
        parts.append(_reminder(mode, question, 'the whole document'))
    else:
        parts.append('Write the consolidated notes now.')
    return '\n'.join(parts)


def clean_answer(text):
    """Drop a leaked reasoning block (when the server did not split it off) and edge whitespace."""
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    return text.strip()


def is_not_found(text):
    norm = re.sub(r'[\W_]+', ' ', text).strip().upper()
    return norm.startswith(NOT_FOUND) and len(norm) <= 80


# --------------------------------------------------------------------------------------------
# Cutting. Boundaries by priority: blank line (paragraph), sentence end, line break, any space.
# Every cut position is the END of a separator run, so the next piece starts at the next text.
# --------------------------------------------------------------------------------------------
_PARA = re.compile(r'\n[ \t]*\n\s*')
_SENT = re.compile(r'(?:[.!?…]+["\'»”’)\]]*\s+)|(?:[。！？]+[」』»”’)\]]*\s*)')
_LINE = re.compile(r'\n+')
_WS = re.compile(r'\s+')
_LEVELS = (_PARA, _SENT, _LINE, _WS)


def find_break(text, lo, hi, min_end):
    """The last cut position e (lo < e <= hi, e >= min_end) at the coarsest separator level that
    has one, or None. Pieces text[lo:e] then end with a separator run."""
    scan_from = max(lo, min_end - 256)
    for rx in _LEVELS:
        best = None
        for m in rx.finditer(text, scan_from, hi):
            if m.end() >= min_end:
                best = m.end()
        if best is not None and best > lo:
            return best
    return None


def overlap_start(text, pos, end, overlap_chars):
    """Where the piece after text[pos:end] starts so that it repeats the tail of this piece:
    the earliest clean boundary within the last overlap_chars characters (>= a quarter of the
    requested overlap); `end` itself (no overlap) if there is none."""
    if overlap_chars <= 0:
        return end
    lo = max(pos + 1, end - overlap_chars)
    for rx in _LEVELS:
        m = rx.search(text, lo, end)
        if m and m.end() < end and end - m.end() >= overlap_chars / 4:
            return m.end()
    return end


@dataclass
class Chunk:
    index: int     # 1-based
    start: int
    end: int
    overlap: int   # characters shared with the previous chunk

    @property
    def length(self):
        return self.end - self.start


def plan_chunks(text, limit_chars, overlap_chars):
    """Cut text into chunks of at most limit_chars characters at paragraph, then sentence, then
    line, then word boundaries (a hard cut only for text without any space), consecutive chunks
    overlapping by up to overlap_chars characters that start at a clean boundary."""
    if limit_chars < 1:
        raise ValueError('limit_chars must be positive')
    n = len(text)
    spans, pos = [], 0
    while True:
        hard = pos + limit_chars
        if hard >= n:
            spans.append((pos, n))
            break
        end = find_break(text, pos, hard, pos + limit_chars // 2)
        if end is None:
            end = find_break(text, pos, hard, pos + 1)
        if end is None or (spans and end <= spans[-1][1]):
            end = hard   # no clean boundary, or no progress past the previous chunk
        spans.append((pos, end))
        pos = overlap_start(text, pos, end, overlap_chars)
    chunks = []
    for i, (a, b) in enumerate(spans, 1):
        shared = max(0, spans[i - 2][1] - a) if i > 1 else 0
        chunks.append(Chunk(i, a, b, shared))
    return chunks


def split_half(text, overlap_chars):
    """Two pieces for a fragment the server refused: cut near the middle at a clean boundary; the
    second piece repeats a short tail of the first."""
    n = len(text)
    if n < 2:
        raise ValueError('cannot split')
    mid = n // 2
    end = find_break(text, 0, mid, mid // 2) or find_break(text, 0, mid, 1) or mid
    ov = min(overlap_chars, end // 2)
    start2 = overlap_start(text, 0, end, ov)
    return text[:end], text[start2:]


def group_notes(items, budget, estimate):
    """Consecutive groups with estimated size <= budget, at least two items per group (so every
    reduce level shrinks the list); the last group may be a single leftover."""
    groups, cur, size = [], [], 0
    for item in items:
        t = estimate(item)
        if len(cur) >= 2 and size + t > budget:
            groups.append(cur)
            cur, size = [], 0
        cur.append(item)
        size += t
    if cur:
        groups.append(cur)
    return groups


def estimate_reduce(n_notes, note_tokens, budget):
    """(levels, requests) of the reduce phase if every note has note_tokens tokens."""
    levels = requests = 0
    n = n_notes
    while n > 1 and n * note_tokens > budget:
        per = max(2, budget // max(1, note_tokens))
        full, rem = divmod(n, per)
        requests += full + (1 if rem > 1 else 0)
        n = full + (1 if rem else 0)
        levels += 1
    return levels, requests + 1


# --------------------------------------------------------------------------------------------
# HTTP client
# --------------------------------------------------------------------------------------------
_CTX_EXACT = re.compile(r'prompt \((\d+)\)\s*\+\s*requested output \((\d+)\)\s*exceeds context limit (\d+)', re.I)
_CTX_WORDS = re.compile(r'context (?:length|limit|window)|maximum context|too many tokens|prompt is too long', re.I)


def parse_context_error(message, code=None):
    """None if the error is not about the request length, else (prompt_tokens, limit); the numbers
    are None when the message does not carry them."""
    m = _CTX_EXACT.search(message or '')
    if m:
        return int(m.group(1)), int(m.group(3))
    if code == 'context_length_exceeded' or _CTX_WORDS.search(message or ''):
        return None, None
    return None


@dataclass
class Completion:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    seconds: float


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None   # never forward the Authorization header to another address


def is_loopback(host):
    return host in ('localhost', '::1') or host.startswith('127.')


class Client:
    def __init__(self, base_url, api_key, model, *, timeout=600.0, retries=3, retry_delay=2.0,
                 busy_wait=300.0, stream=True, thinking=False, temperature=None, seed=None,
                 max_body=MAX_BODY_BYTES, log=None, sleep=time.sleep):
        parsed = urlparse(base_url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            raise LongdocError(f'bad --base-url {base_url!r}: expected http(s)://host[:port]/v1')
        base = base_url.rstrip('/')
        if parsed.path in ('', '/'):
            base += '/v1'
        self.url = base + '/chat/completions'
        self.model = model
        self.timeout, self.retries, self.retry_delay = timeout, retries, retry_delay
        self.busy_wait = busy_wait
        self.stream, self.thinking = stream, thinking
        self.temperature, self.seed = temperature, seed
        self.max_body = max_body
        self.log = log or (lambda msg: None)
        self.sleep = sleep
        self.headers = {'Content-Type': 'application/json; charset=utf-8', 'Accept': 'text/event-stream, application/json'}
        if api_key:
            self.headers['Authorization'] = 'Bearer ' + api_key
        loopback = is_loopback(parsed.hostname)
        if api_key and parsed.scheme == 'http' and not loopback:
            self.log('warning: the API key is sent over plain HTTP to a non-local address')
        proxy = urllib.request.ProxyHandler({}) if loopback else urllib.request.ProxyHandler()
        self.opener = urllib.request.build_opener(proxy, _NoRedirect)

    # -- public ------------------------------------------------------------------------------
    def chat(self, system, user, max_tokens):
        payload = {
            'model': self.model,
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            'max_tokens': max_tokens,
            'stream': self.stream,
            'chat_template_kwargs': {'enable_thinking': self.thinking},
        }
        if self.stream:
            payload['stream_options'] = {'include_usage': True}
        if self.temperature is not None:
            payload['temperature'] = self.temperature
        if self.seed is not None:
            payload['seed'] = self.seed
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        if len(body) > self.max_body:
            raise SplitError(f'request body is {len(body):,} bytes, over the {self.max_body:,}-byte limit',
                             reason='body')
        attempt = 0
        busy_deadline = None
        while True:
            try:
                return self._once(body)
            except _Busy as busy:
                now = time.monotonic()
                if busy_deadline is None:
                    busy_deadline = now + self.busy_wait
                if now >= busy_deadline:
                    raise LongdocError('the server stayed busy (another request is active) for '
                                       f'{self.busy_wait:.0f} s; try later or raise --busy-wait')
                wait = max(busy.retry_after, self.retry_delay * 2)
                self.log(f'server busy (one request at a time), waiting {wait:.0f} s')
                self.sleep(wait)
            except _Transient as exc:
                attempt += 1
                if attempt > self.retries:
                    plural = 'retry' if self.retries == 1 else 'retries'
                    raise LongdocError(f'request failed after {self.retries} {plural}: {exc}') from None
                delay = self.retry_delay * 2 ** (attempt - 1)
                self.log(f'network/server fault ({exc}); retry {attempt}/{self.retries} in {delay:.0f} s')
                self.sleep(delay)

    # -- one attempt -------------------------------------------------------------------------
    def _once(self, body):
        request = urllib.request.Request(self.url, data=body, headers=self.headers, method='POST')
        started = time.monotonic()
        try:
            response = self.opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            self._raise_http(exc)
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            raise _Transient(_describe(exc)) from None
        try:
            with response:
                if response.headers.get_content_type() == 'text/event-stream':
                    result = self._read_sse(response)
                else:
                    result = self._parse_full(json.loads(response.read().decode('utf-8')))
        except (OSError, http.client.HTTPException, ValueError) as exc:
            raise _Transient(_describe(exc)) from None
        result.seconds = time.monotonic() - started
        return result

    def _raise_http(self, exc):
        try:
            raw = exc.read(65536).decode('utf-8', 'replace')
        except Exception:   # noqa: BLE001 - the status line is enough
            raw = ''
        message, code = raw.strip()[:500] or str(exc.reason), None
        try:
            err = json.loads(raw).get('error')
            if isinstance(err, dict):
                message, code = err.get('message') or message, err.get('code')
            elif isinstance(err, str):
                message = err
        except (ValueError, AttributeError):
            pass
        status = exc.code
        ctx = parse_context_error(message, code)
        if ctx is not None:
            raise SplitError(message, *ctx)
        if status == 413:
            raise SplitError(message, reason='body')
        if status == 504:
            raise SplitError(f'{message} (the server cut the request at its time limit)', reason='time')
        if status == 429:
            try:
                retry_after = float(exc.headers.get('Retry-After', 0))
            except (TypeError, ValueError):
                retry_after = 0.0
            raise _Busy(retry_after)
        if status in (401, 403):
            raise LongdocError(f'HTTP {status}: the server rejected the API key '
                               '(set QWEN36_API_KEY or pass --api-key-file)')
        if status == 404:
            raise LongdocError(f'HTTP 404 at {self.url}: route not found; check --base-url '
                               '(it must end with /v1)')
        if status in (408, 500, 502, 503):
            raise _Transient(f'HTTP {status}: {message}')
        if 300 <= status < 400:
            raise LongdocError(f'HTTP {status}: redirects are not followed (they would carry the API key '
                               f'to another address); use the final URL in --base-url')
        raise LongdocError(f'HTTP {status}: {message}')

    def _read_sse(self, response):
        parts, finish, usage, done = [], None, None, False
        for raw in response:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:'):
                continue   # blank separators and ': keep-alive' comments
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            try:
                obj = json.loads(data)
            except ValueError:
                raise _Transient('malformed stream chunk') from None
            err = obj.get('error') if isinstance(obj, dict) else None
            if err:
                message = err.get('message', str(err)) if isinstance(err, dict) else str(err)
                ctx = parse_context_error(message, err.get('code') if isinstance(err, dict) else None)
                if ctx is not None:
                    raise SplitError(message, *ctx)
                raise LongdocError(f'server error in the stream: {message}')
            for choice in obj.get('choices') or ():
                content = (choice.get('delta') or {}).get('content')
                if content:
                    parts.append(content)
                if choice.get('finish_reason'):
                    finish = choice['finish_reason']
            if obj.get('usage'):
                usage = obj['usage']
        if not done and finish is None:
            raise _Transient('the stream ended before the answer was complete')
        return _completion(''.join(parts), finish or 'stop', usage)

    def _parse_full(self, obj):
        choice = (obj.get('choices') or [{}])[0]
        message = choice.get('message') or {}
        return _completion(message.get('content') or '', choice.get('finish_reason') or 'stop', obj.get('usage'))


def _completion(text, finish, usage):
    usage = usage or {}
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
    return Completion(text, finish, int(usage.get('prompt_tokens') or 0),
                      int(usage.get('completion_tokens') or 0), int(cached or 0), 0.0)


def _describe(exc):
    if isinstance(exc, urllib.error.URLError) and not isinstance(exc.reason, str):
        exc = exc.reason
    text = str(exc) or type(exc).__name__
    return f'{type(exc).__name__}: {text}'


# --------------------------------------------------------------------------------------------
# Map-reduce driver
# --------------------------------------------------------------------------------------------
def fmt(n):
    return f'{n:,}'


def est_tokens(chars, cpt):
    """Estimated tokens of `chars` characters (the epsilon keeps 280000 / 2.8 at exactly 100000)."""
    return math.ceil(chars / cpt - 1e-6)


def part_name(label):
    head, *rest = label.split('.')
    return '.'.join([head.zfill(4)] + rest) if head.isdigit() else label.replace('..', '-')


class Runner:
    def __init__(self, args, client, log, estimate_cpt):
        self.mode, self.question = args.mode, args.question or ''
        self.client, self.log = client, log
        self.max_tokens = args.max_tokens
        self.final_max_tokens = args.final_max_tokens or args.max_tokens
        self.chunk_tokens, self.overlap_tokens = args.chunk_tokens, args.overlap_tokens
        self.reduce_tokens = args.reduce_tokens or args.chunk_tokens
        self.cpt, self.window = estimate_cpt, args.context_window
        self.configured_cpt = estimate_cpt
        self.save_dir = Path(args.save_parts) if args.save_parts else None
        self.parts = []
        self.sent_chars = self.sent_tokens = 0
        self.requests = 0
        self.prompt_tokens = self.completion_tokens = self.cached_tokens = 0

    # -- helpers -----------------------------------------------------------------------------
    def _est(self, chars):
        return est_tokens(chars, self.cpt)

    @property
    def overlap_chars(self):
        return int(self.overlap_tokens * self.cpt)

    def _send(self, user, max_tokens, desc, document=True):
        chars = len(SYSTEM_PROMPT) + len(user)
        est = self._est(chars) + TEMPLATE_TOKENS
        if est + max_tokens > self.window:
            raise SplitError(f'estimated {fmt(est)} prompt + {fmt(max_tokens)} answer tokens do not fit the '
                             f'{fmt(self.window)}-token window', reason='estimate')
        self.log(f'{desc}: sending ~{fmt(est)} prompt tokens (estimate), answer up to {fmt(max_tokens)}')
        try:
            result = self.client.chat(SYSTEM_PROMPT, user, max_tokens)
        except SplitError as exc:
            self._learn(exc, chars)
            raise
        self.requests += 1
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        self.cached_tokens += result.cached_tokens
        cpt_note = ''
        if result.prompt_tokens:
            observed = chars / result.prompt_tokens
            if document:   # the tip at the end is about the document text, not the generated notes
                self.sent_chars += chars
                self.sent_tokens += result.prompt_tokens
            cpt_note = f', {observed:.2f} chars/token'
            self._lower_cpt(observed)
        rate = f', {result.completion_tokens / result.seconds:.1f} tok/s' if result.seconds > 0 and result.completion_tokens else ''
        self.log(f'{desc}: done in {result.seconds:.1f} s: prompt {fmt(result.prompt_tokens)} tokens '
                 f'(cached {fmt(result.cached_tokens)}), answer {fmt(result.completion_tokens)} tokens{cpt_note}{rate}')
        if result.finish_reason == 'length':
            self.log(f'warning: {desc}: the answer was cut at max_tokens; raise --max-tokens'
                     + (' (with --thinking on the reasoning also uses this budget)' if self.client.thinking else ''))
        return result

    def _nonempty(self, answer, result):
        if not answer:
            raise LongdocError(f'the model returned an empty final answer (finish_reason {result.finish_reason}); '
                               'with --thinking on raise --max-tokens / --final-max-tokens')
        return answer

    def _lower_cpt(self, observed):
        observed = max(observed, 0.5)
        if observed < self.cpt * 0.99:   # ignore rounding-level differences
            self.log(f'estimate corrected: {self.cpt:.2f} -> {observed:.2f} chars/token for the rest of the run')
            self.cpt = observed

    def _learn(self, exc, request_chars):
        if exc.prompt_tokens:
            self._lower_cpt(request_chars / exc.prompt_tokens)
        if exc.limit and exc.limit != self.window:
            self.log(f'the server reports a {fmt(exc.limit)}-token context limit (was assuming {fmt(self.window)})')
            self.window = exc.limit

    def _save(self, name, text, meta):
        meta = dict(meta, file=f'{name}.md')
        self.parts.append(meta)
        if self.save_dir is None:
            return
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            (self.save_dir / f'{name}.md').write_text(text + '\n', encoding='utf-8', newline='\n')
            (self.save_dir / 'parts.json').write_text(
                json.dumps({'mode': self.mode, 'question': self.question, 'parts': self.parts},
                           ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
        except OSError as exc:
            raise LongdocError(f'cannot write to --save-parts {self.save_dir}: {exc}') from None

    def _meta(self, kind, label, result, chars):
        return {'kind': kind, 'label': label, 'chars': chars, 'prompt_tokens': result.prompt_tokens,
                'completion_tokens': result.completion_tokens, 'cached_tokens': result.cached_tokens,
                'finish_reason': result.finish_reason, 'seconds': round(result.seconds, 2)}

    # -- phases ------------------------------------------------------------------------------
    def run(self, text, chunks):
        t0 = time.monotonic()
        pieces = [(str(c.index), text[c.start:c.end]) for c in chunks]
        total = len(pieces)
        if total == 1:
            user = single_prompt(self.mode, self.question, pieces[0][1])
            try:
                result = self._send(user, self.final_max_tokens, 'single request')
            except SplitError as exc:
                self.log(f'the whole document was refused ({exc.reason}): splitting it in half and using map-reduce')
                first, second = split_half(pieces[0][1], self.overlap_chars)
                pieces, total = [('1', first), ('2', second)], 2
            else:
                answer = self._nonempty(clean_answer(result.text), result)
                self._save('final', answer, self._meta('single', '1', result, len(pieces[0][1])))
                self._summary(t0)
                return answer
        notes = []
        for label, piece in pieces:
            notes += self._map(piece, label, total, 0)
        answer = self._reduce_all(notes, total)
        self._summary(t0)
        return answer

    def _map(self, piece, label, total, depth):
        desc = f'map {label}/{total}'
        user = map_prompt(self.mode, self.question, label, total, piece)
        try:
            result = self._send(user, self.max_tokens, desc)
        except SplitError as exc:
            if len(piece) < MIN_SPLIT_CHARS or depth >= MAX_SPLIT_DEPTH:
                raise LongdocError(f'{desc}: the server refuses even a small fragment: {exc}') from None
            self.log(f'{desc}: too large for the server ({exc.reason}: {exc}); splitting in half')
            first, second = split_half(piece, self.overlap_chars)
            return (self._map(first, f'{label}.1', total, depth + 1)
                    + self._map(second, f'{label}.2', total, depth + 1))
        answer = clean_answer(result.text)
        if not answer:
            self.log(f'warning: {desc}: empty answer (finish_reason {result.finish_reason}); '
                     'with --thinking on raise --max-tokens')
        self._save('map-' + part_name(label), answer, self._meta('map', label, result, len(piece)))
        return [Note(f'fragment {label}', label, answer)]

    def _reduce_all(self, notes, total):
        useful = [n for n in notes if n.text and not is_not_found(n.text)]
        omitted = len(notes) - len(useful)
        if not useful:
            self.log('no fragment reported anything relevant')
            return f'{NOT_FOUND}: none of the {total} fragments of the document contained information relevant to the question.'
        if omitted:
            self.log(f'{omitted} of {len(notes)} partial answers are "{NOT_FOUND}" and are left out of the reduce step')
        level = 0
        estimate = lambda note: self._est(len(note.text))  # noqa: E731 - follows the learned cpt
        while len(useful) > 1 and sum(map(estimate, useful)) > self.reduce_tokens:
            groups = group_notes(useful, self.reduce_tokens, estimate)
            if len(groups) >= len(useful):
                raise LongdocError('hierarchical reduce makes no progress; raise --reduce-tokens')
            level += 1
            self.log(f'reduce level {level}: {len(useful)} partial answers do not fit one request, '
                     f'merging them in {len(groups)} groups')
            merged = []
            for gi, group in enumerate(groups, 1):
                merged.append(group[0] if len(group) == 1 else
                              self._reduce_group(group, False, f'L{level}-G{gi:02d}', total, omitted))
            useful = merged
        return self._reduce_group(useful, True, 'final', total, omitted).text

    def _reduce_group(self, group, final, tag, total, omitted):
        """One reduce request over a group of notes -> a Note (final: the final answer, as a Note)."""
        title = ('fragments ' + f'{group[0].label}..{group[-1].label}') if len(group) > 1 else group[0].title
        label = f'{group[0].label}..{group[-1].label}' if len(group) > 1 else group[0].label
        if len(group) == 1 and not final:
            return group[0]
        user = reduce_prompt(self.mode, self.question, group, final, total, omitted)
        desc = f'reduce {tag} ({len(group)} notes)'
        try:
            result = self._send(user, self.final_max_tokens if final else self.max_tokens, desc, document=False)
        except SplitError as exc:
            if len(group) <= 2:
                raise LongdocError(f'{desc}: the server refuses a request of {len(group)} partial answers: {exc}') from None
            self.log(f'{desc}: too large for the server ({exc.reason}: {exc}); reducing the two halves first')
            mid = len(group) // 2
            left = self._reduce_group(group[:mid], False, tag + 'a', total, omitted)
            right = self._reduce_group(group[mid:], False, tag + 'b', total, omitted)
            return self._reduce_group([left, right], final, tag + 'm', total, omitted)
        answer = clean_answer(result.text)
        if final:
            answer = self._nonempty(answer, result)
        self._save('final' if final else f'reduce-{tag}', answer,
                   self._meta('final' if final else 'reduce', label, result, sum(len(n.text) for n in group)))
        return Note(title, label, answer)

    def _summary(self, t0):
        elapsed = time.monotonic() - t0
        self.log(f'finished: {self.requests} requests in {elapsed:.0f} s; prompt {fmt(self.prompt_tokens)} tokens '
                 f'(cached {fmt(self.cached_tokens)}), answers {fmt(self.completion_tokens)} tokens')
        if self.sent_tokens:
            observed = self.sent_chars / self.sent_tokens
            assumed = self.configured_cpt
            if observed < assumed * 0.9:
                self.log(f'tip: this text has only {observed:.2f} chars/token against the assumed {assumed:.2f}, so '
                         f'chunks were larger than planned (and some were split); next time use --chars-per-token {observed * 0.95:.1f}')
            elif observed > assumed * 1.15:
                self.log(f'tip: this text has {observed:.2f} chars/token against the assumed {assumed:.2f}, so chunks '
                         f'used only {assumed / observed:.0%} of the window they were planned for; --chars-per-token '
                         f'{observed * 0.9:.1f} would give fewer, larger chunks')


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def read_input(path, encoding):
    try:
        if path in (None, '-'):
            if sys.stdin is None or sys.stdin.isatty():
                raise LongdocError('no input: pass a file path or pipe the text into stdin')
            raw = sys.stdin.buffer.read()
        else:
            raw = Path(path).read_bytes()
    except OSError as exc:
        raise LongdocError(f'cannot read {path}: {exc}') from None
    utf8 = encoding.lower().replace('_', '-') in ('utf-8', 'utf8')
    if utf8 and b'\x00' in raw[:65536]:
        raise LongdocError('the input looks binary (NUL bytes); convert PDF/DOCX to UTF-8 text first')
    try:
        text = raw.decode('utf-8-sig' if utf8 else encoding)
    except (UnicodeDecodeError, LookupError) as exc:
        raise LongdocError(f'the input is not valid {encoding} text ({exc}); convert it to UTF-8 '
                           'or pass --encoding') from None
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    if not text.strip():
        raise LongdocError('the input is empty')
    return text


def load_api_key(path):
    if path:
        try:
            raw = Path(path).read_text(encoding='utf-8-sig')
        except OSError as exc:
            raise LongdocError(f'cannot read --api-key-file: {exc}') from None
        for line in raw.splitlines():
            m = re.match(r'\s*(?:export\s+)?QWEN36_API_KEY\s*=\s*(.*?)\s*$', line)
            if m:
                key = m.group(1).strip('\'"')
                break
        else:
            key = raw.strip()
    else:
        key = os.environ.get('QWEN36_API_KEY', '').strip()
    if key and (not key.isascii() or any(c.isspace() for c in key)):
        raise LongdocError('the API key must be a single ASCII token without spaces')
    return key or None


def build_parser():
    p = argparse.ArgumentParser(
        prog='longdoc', description='Question answering / summary for documents larger than the model window '
        '(client-side map-reduce over the OpenAI-compatible API).',
        epilog='Examples:\n'
               '  longdoc.py report.md -q "What deadlines are set?"\n'
               '  longdoc.py book.txt --mode summary --save-parts parts/ > summary.txt\n'
               '  longdoc.py report.md -q "..." --base-url https://llm.example.com/v1 --api-key-file key.txt\n'
               '  longdoc.py report.md -q "..." --dry-run',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input', nargs='?', default='-', help='UTF-8 text or markdown file; "-" or omitted: stdin')
    p.add_argument('-q', '--question', help='the question (required in the default qa mode); in summary mode an optional focus')
    p.add_argument('--mode', choices=('qa', 'summary'), default='qa', help='qa (default): answer the question; summary: condense the document')
    p.add_argument('--chunk-tokens', type=int, default=100000, help='upper bound of document tokens per request (default 100000)')
    p.add_argument('--overlap-tokens', type=int, default=1000, help='overlap between neighbouring chunks (default 1000)')
    p.add_argument('--chars-per-token', type=float, default=2.8, help='token estimate: characters per token (default 2.8, conservative for mixed Russian/English)')
    p.add_argument('--context-window', type=int, default=131072, help='server context window in tokens (default 131072, profile main128)')
    p.add_argument('--max-tokens', type=int, default=2048, help='answer budget of every map and intermediate request (default 2048)')
    p.add_argument('--final-max-tokens', type=int, help='answer budget of the final request (default: --max-tokens)')
    p.add_argument('--reduce-tokens', type=int, help='size of one reduce group in tokens (default: --chunk-tokens)')
    p.add_argument('--thinking', choices=('on', 'off'), default='off', help='Qwen thinking mode (default off); the reasoning shares the max-tokens budget')
    p.add_argument('--base-url', default=os.environ.get('QWEN36_BASE_URL') or DEFAULT_BASE_URL, help=f'API base URL (env QWEN36_BASE_URL, default {DEFAULT_BASE_URL})')
    p.add_argument('--api-key-file', help='file with the key (bare, or a QWEN36_API_KEY=... line); otherwise env QWEN36_API_KEY')
    p.add_argument('--model', default=DEFAULT_MODEL, help=f'served model name (default {DEFAULT_MODEL})')
    p.add_argument('--temperature', type=float, help='sampling temperature (default: the server default)')
    p.add_argument('--seed', type=int, help='sampling seed')
    p.add_argument('--timeout', type=float, default=600.0, help='socket timeout, seconds (default 600; the guarded server cuts requests at 900)')
    p.add_argument('--retries', type=int, default=3, help='retries after a network fault (default 3)')
    p.add_argument('--retry-delay', type=float, default=2.0, help='first retry delay, seconds; doubles each time (default 2)')
    p.add_argument('--busy-wait', type=float, default=300.0, help='how long to wait while the server serves another request, seconds (default 300)')
    p.add_argument('--max-body-bytes', type=int, default=MAX_BODY_BYTES, help='split a request whose JSON body is larger than this (default 2000000)')
    p.add_argument('--no-stream', action='store_true', help='do not stream the answer (default: stream)')
    p.add_argument('--encoding', default='utf-8', help='input encoding (default utf-8)')
    p.add_argument('-o', '--output', metavar='FILE', help='also write the final answer to FILE as UTF-8 (on Windows PowerShell prefer this to ">" redirection, which re-encodes)')
    p.add_argument('--save-parts', metavar='DIR', help='write every partial answer, the reduce steps, the final answer and parts.json into DIR')
    p.add_argument('--dry-run', action='store_true', help='print the cutting plan and the request count; send nothing')
    p.add_argument('--quiet', action='store_true', help='no progress on stderr')
    return p


def validate(args, parser):
    if args.mode == 'qa' and not args.question and not args.dry_run:
        parser.error('--question is required in qa mode (or use --mode summary)')
    if args.chars_per_token <= 0:
        parser.error('--chars-per-token must be positive')
    if args.chunk_tokens < 1 or args.overlap_tokens < 0 or args.overlap_tokens * 2 > args.chunk_tokens:
        parser.error('--chunk-tokens must be positive and --overlap-tokens at most half of it')
    for name in ('max_tokens', 'context_window', 'retries'):
        if getattr(args, name) < (0 if name == 'retries' else 1):
            parser.error(f'--{name.replace("_", "-")} is out of range')
    for name in ('final_max_tokens', 'reduce_tokens'):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')


def prompt_overhead(args):
    """Estimated tokens of everything in a request except the document text / the notes."""
    cpt = args.chars_per_token
    q = args.question or ''
    sizes = (len(map_prompt(args.mode, q, '999', 999, '')),
             len(single_prompt(args.mode, q, '')),
             len(reduce_prompt(args.mode, q, [], True, 999, 999)))
    return est_tokens(len(SYSTEM_PROMPT) + max(sizes), cpt) + TEMPLATE_TOKENS


def check_budget(args, parser, overhead):
    final_max = args.final_max_tokens or args.max_tokens
    reduce_tokens = args.reduce_tokens or args.chunk_tokens
    worst = max(args.chunk_tokens, reduce_tokens) + overhead + max(args.max_tokens, final_max)
    if worst > args.context_window:
        room = args.context_window - overhead - max(args.max_tokens, final_max)
        parser.error(f'chunk/reduce size {max(args.chunk_tokens, reduce_tokens):,} + prompt ~{overhead:,} + answer '
                     f'{max(args.max_tokens, final_max):,} exceeds --context-window {args.context_window:,}; '
                     f'use --chunk-tokens (and --reduce-tokens) of at most {max(room, 0):,}')


def print_plan(args, text, chunks, overhead):
    cpt = args.chars_per_token
    total = est_tokens(len(text), cpt)
    reduce_tokens = args.reduce_tokens or args.chunk_tokens
    out = ['longdoc plan (dry run: no request is sent)',
           f'  input:         {fmt(len(text))} characters, ~{fmt(total)} tokens (estimate at {cpt:g} chars/token; '
           'the guarded server has no token-count route)',
           f'  mode:          {args.mode}',
           f'  chunk limit:   {fmt(args.chunk_tokens)} tokens = {fmt(int(args.chunk_tokens * cpt))} characters; '
           f'overlap {fmt(args.overlap_tokens)} tokens = {fmt(int(args.overlap_tokens * cpt))} characters',
           f'  one request:   chunk + ~{fmt(overhead)} prompt + {fmt(args.max_tokens)} answer tokens, '
           f'window {fmt(args.context_window)}',
           f'  chunks:        {len(chunks)}']
    shown = chunks if len(chunks) <= 40 else chunks[:20] + chunks[-10:]
    for c in shown:
        if shown is not chunks and c is chunks[-10]:
            out.append('    ...')
        out.append(f'    #{c.index:<4} chars {fmt(c.start):>12} - {fmt(c.end):<12} ~{fmt(est_tokens(c.length, cpt)):>9} tokens'
                   f'   overlap with previous {fmt(c.overlap)} characters')
    if len(chunks) == 1:
        out.append('  requests:      1 (the document fits one request; no reduce step)')
    else:
        levels, reqs = estimate_reduce(len(chunks), args.max_tokens, reduce_tokens)
        out.append(f'  requests:      {len(chunks)} map (one at a time) + {reqs} reduce '
                   f'({levels} intermediate level{"" if levels == 1 else "s"}) = {len(chunks) + reqs} '
                   f'(reduce is a worst case: every partial answer uses all {fmt(args.max_tokens)} tokens)')
    out.append('  on a context-length refusal the fragment is cut in half and retried')
    print('\n'.join(out))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate(args, parser)
    overhead = prompt_overhead(args)
    check_budget(args, parser, overhead)
    log_enabled = not args.quiet

    def log(message):
        if log_enabled:
            print(f'[{time.strftime("%H:%M:%S")}] {message}', file=sys.stderr, flush=True)

    try:
        text = read_input(args.input, args.encoding)
        limit_chars = max(1, int(args.chunk_tokens * args.chars_per_token))
        overlap_chars = int(args.overlap_tokens * args.chars_per_token)
        chunks = plan_chunks(text, limit_chars, overlap_chars)
        if args.dry_run:
            print_plan(args, text, chunks, overhead)
            return 0
        log(f'{fmt(len(text))} characters, ~{fmt(est_tokens(len(text), args.chars_per_token))} tokens (estimate), '
            f'{len(chunks)} chunk{"" if len(chunks) == 1 else "s"} of up to {fmt(args.chunk_tokens)} tokens')
        client = Client(args.base_url, load_api_key(args.api_key_file), args.model, timeout=args.timeout,
                        retries=args.retries, retry_delay=args.retry_delay, busy_wait=args.busy_wait,
                        stream=not args.no_stream, thinking=args.thinking == 'on',
                        temperature=args.temperature, seed=args.seed, max_body=args.max_body_bytes, log=log)
        runner = Runner(args, client, log, args.chars_per_token)
        answer = runner.run(text, chunks)
    except LongdocError as exc:
        print(f'longdoc: error: {exc}', file=sys.stderr)
        if args.save_parts:
            print(f'longdoc: partial answers already written to {args.save_parts}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('longdoc: interrupted', file=sys.stderr)
        return 130
    answer = answer.rstrip('\n') + '\n'
    if args.output:
        try:
            Path(args.output).write_text(answer, encoding='utf-8', newline='\n')
        except OSError as exc:
            print(f'longdoc: error: cannot write --output {args.output}: {exc}', file=sys.stderr)
            sys.stdout.write(answer)   # the answer is not lost
            return 1
    sys.stdout.write(answer)
    sys.stdout.flush()
    return 0


def _cli():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')   # Russian text on a cp1251/cp866 console
        except (AttributeError, ValueError):
            pass
    sys.exit(main())


if __name__ == '__main__':
    _cli()
