"""CPU tests for eval-v1 (fixture, sandbox, checkers, runner, managed mode, compare). No GPU, no service.

Run (WSL): CUDA_VISIBLE_DEVICES= /home/USER/qwen36-api/.venv/bin/python test-eval-v1-cpu.py [--out FILE]
Every test uses a temporary root/freeze/lock; nothing under /home/USER/qwen36-api is touched.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import types

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'eval-v1'))
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = load('eval_v1', 'eval-v1.py')
cmp_ = load('eval_v1_compare', 'eval-v1-compare.py')
cases, sandbox = ev.cases, ev.sandbox
import bench  # noqa: E402

REF = {
    'rotate_right': 'def rotate_right(a, k):\n    if not a:\n        return []\n    k %= len(a)\n    return a[-k:] + a[:-k] if k else list(a)\n',
    'is_palindrome': 'def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]\n',
    'rle': ('def rle(s):\n    out = []\n    i = 0\n    while i < len(s):\n        j = i\n        while j < len(s) and s[j] == s[i]:\n'
            '            j += 1\n        out.append(s[i] + str(j - i))\n        i = j\n    return "".join(out)\n'),
    'merge_intervals': ('def merge_intervals(intervals):\n    res = []\n    for a, b in sorted(intervals):\n'
                        '        if res and a <= res[-1][1]:\n            res[-1][1] = max(res[-1][1], b)\n'
                        '        else:\n            res.append([a, b])\n    return res\n'),
    'top_words': ('def top_words(text, n):\n    words, cur = [], ""\n    for ch in text.lower():\n        if ch.isalpha():\n'
                  '            cur += ch\n        else:\n            if cur:\n                words.append(cur)\n            cur = ""\n'
                  '    if cur:\n        words.append(cur)\n    counts = {}\n    for w in words:\n        counts[w] = counts.get(w, 0) + 1\n'
                  '    return sorted(counts, key=lambda w: (-counts[w], w))[:n]\n'),
    'roman_to_int': ('def roman_to_int(s):\n    v = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}\n'
                     '    total = 0\n    for i, ch in enumerate(s):\n        if i + 1 < len(s) and v[ch] < v[s[i + 1]]:\n'
                     '            total -= v[ch]\n        else:\n            total += v[ch]\n    return total\n'),
    'flatten': ('def flatten(x):\n    out = []\n    for item in x:\n        if isinstance(item, list):\n            out.extend(flatten(item))\n'
                '        else:\n            out.append(item)\n    return out\n'),
    'balanced': ('def balanced(s):\n    pairs = {")": "(", "]": "[", "}": "{"}\n    stack = []\n    for ch in s:\n        if ch in "([{":\n'
                 '            stack.append(ch)\n        elif ch in pairs:\n            if not stack or stack.pop() != pairs[ch]:\n'
                 '                return False\n    return not stack\n'),
    'second_largest': 'def second_largest(a):\n    u = sorted(set(a))\n    return u[-2] if len(u) >= 2 else None\n',
    'parse_duration': ('def parse_duration(s):\n    total, num = 0, ""\n    for ch in s:\n        if ch.isdigit():\n            num += ch\n'
                       '        else:\n            total += int(num) * {"h": 3600, "m": 60, "s": 1}[ch]\n            num = ""\n    return total\n'),
    'transpose': 'def transpose(m):\n    return [list(r) for r in zip(*m)]\n',
    'caesar': ('def caesar(s, k):\n    A = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"\n    out = []\n    for ch in s:\n        low = ch.lower()\n'
               '        if low in A:\n            c = A[(A.index(low) + k) % 33]\n            out.append(c.upper() if ch != low else c)\n'
               '        else:\n            out.append(ch)\n    return "".join(out)\n'),
    'clamp-orig': 'def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n',
    'clamp-edit': 'def clamp(x, lo, hi):\n    if lo > hi:\n        lo, hi = hi, lo\n    return max(lo, min(x, hi))\n',
}
ORACLE_FINAL = {
    ('tool-01', 'main'): 'В Казани сейчас 17 °C.', ('tool-02', 'main'): 'Получится 231,75 евро.',
    ('tool-03', 'main'): 'Посылка в городе Нижний Новгород, доставка ожидается 17 октября 2026 года.',
    ('tool-04', 'main'): 'Результат: 87.', ('tool-05', 'main'): 'Событие создано, идентификатор EV-7731.',
    ('tool-06', 'main'): 'Сумма равна 50.', ('tool-07', 'main'): 'Cat', ('tool-08', 'main'): 'Отдел — логистика, в Томске −3 °C.',
    ('tool-09', 'main'): 'В Омске 41 °F.', ('tool-10', 'main'): 'Это 11 397,15 тенге.',
    ('tool-11', 'main'): 'Найдены: Звук-3 и Тишина Pro.', ('tool-12', 'main'): 'В Перми 8 °C, в Сочи 19 °C — в Сочи теплее.',
    ('agent-03', 'orig'): 'Вторник, 14:00', ('agent-03', 'edit'): 'Среда, 16:30', ('agent-04', 'orig'): 'В Казани 17 °C.',
    ('agent-04', 'edit'): 'В Перми сейчас 9 °C.', ('agent-08', 'orig'): 'ПАРИЖ', ('agent-08', 'edit'): 'Париж',
    ('agent-10', 'orig'): '1390', ('agent-10', 'edit'): '5560',
}


def text_answer(check):
    kind = check['type']
    if kind == 'number':
        return str(check['value'])
    if kind == 'words':
        return check['accept'][0].capitalize() + '.'
    if kind == 'sequence':
        return ', '.join(check['value'])
    if kind == 'json':
        return '```json\n' + json.dumps(check['value'], ensure_ascii=False) + '\n```'
    raise KeyError(kind)


class Oracle:
    """Fake /v1/chat/completions: a 'perfect model' (or a mutated one) keyed by the request's initial messages."""

    def __init__(self, suite, mutate=None):
        self.index = []
        for case in suite:
            for request in case['requests']:
                self.index.append((request['messages'], case['id'], request))
        self.mutate = mutate or (lambda case_id, label, message, finish: (message, finish))
        self.calls = 0

    def find(self, messages):
        best = None
        for initial, case_id, request in self.index:
            if messages[:len(initial)] == initial and (best is None or len(initial) > len(best[0])):
                best = (initial, case_id, request)
        return best

    def __call__(self, payload):
        self.calls += 1
        initial, case_id, request = self.find(payload['messages'])
        assert payload['temperature'] == 0 and payload['chat_template_kwargs'] == {'enable_thinking': False}
        assert payload['stream'] is False and payload['seed'] == cases.SAMPLING['seed']
        label = request['label']
        if request['kind'] == 'text':
            message, finish = {'role': 'assistant', 'content': ORACLE_FINAL.get((case_id, label)) or text_answer(request['check'])}, 'stop'
        elif request['kind'] == 'code':
            key = request['function'] if request['function'] != 'clamp' else f'clamp-{label}'
            message, finish = {'role': 'assistant', 'content': f'```python\n{REF[key]}```'}, 'stop'
        else:
            step = sum(1 for m in payload['messages'][len(initial):] if m['role'] == 'assistant')
            calls = request['calls']
            if case_id == 'tool-12' and step == 0:
                batch = calls
            else:
                done = sum(1 for m in payload['messages'][len(initial):] if m['role'] == 'tool')
                batch = calls[done:done + 1]
            if batch:
                message = {'role': 'assistant', 'content': None, 'tool_calls': [
                    {'id': f'call_{case_id}_{label}_{step}_{i}', 'type': 'function',
                     'function': {'name': c['name'], 'arguments': json.dumps(c['arguments'], ensure_ascii=False)}}
                    for i, c in enumerate(batch)]}
                finish = 'tool_calls'
            else:
                message, finish = {'role': 'assistant', 'content': ORACLE_FINAL[(case_id, label)]}, 'stop'
        message, finish = self.mutate(case_id, label, message, finish)
        if isinstance(message, tuple):  # (status, None) injected failure
            return message[0], None, None
        return 200, {'choices': [{'message': message, 'finish_reason': finish}],
                     'usage': {'prompt_tokens': 11, 'completion_tokens': 7}}, 'rid'


# ----------------------------------------------------------------------------- helpers
def temp_freeze(tmp):
    path = tmp / 'freeze.json'
    path.write_text(json.dumps(ev.build_freeze(None)), encoding='utf-8')
    return path


def with_freeze(path):
    ev.FREEZE = path


def receipt(profile, run_id='a' * 32, runtime_ok=True, config=None):
    files = {'x.py': 'b' * 64}
    expected = json.loads(ev.FREEZE.read_text())['runtime_expectations']['source_files']
    for rel, sha in expected.items():
        files[rel] = sha if runtime_ok else 'c' * 64
    return {'profile': profile, 'run_id': run_id, 'checkpoint': bench.CHECKPOINT,
            'checkpoint_revision': bench.CHECKPOINT_REVISION, 'source_files': files,
            'source_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            'service_source_files': {n: 'd' * 64 for n in bench.SERVICE_FILES},
            'packages': {n: '1.0' for n in bench.PACKAGES},
            'configuration': copy.deepcopy(config if config is not None else ev.PROFILES[profile]),
            'native_libraries': {}}


def make_root(tmp, profile, **kw):
    root = tmp / 'root'
    (root / 'reports').mkdir(parents=True, exist_ok=True)
    (root / 'active-run.json').write_text(json.dumps(receipt(profile, **kw)))
    return root


def args(**kw):
    return types.SimpleNamespace(**kw)


TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ----------------------------------------------------------------------------- fixture
@test
def fixture_shape_and_determinism():
    a, b = cases.materialize(), cases.materialize()
    assert a == b and cases.suite_sha256(a) == cases.suite_sha256(b)
    assert len(a) == 60 and len({c['id'] for c in a}) == 60
    for cat in cases.CATEGORIES:
        assert sum(c['category'] == cat for c in a) == 12, cat
    assert [c['category'] for c in a] == sorted((c['category'] for c in a), key=cases.CATEGORIES.index)
    json.dumps(a, ensure_ascii=False)
    for c in a:
        for r in c['requests']:
            assert r['kind'] in ev.RUNNERS and r['max_tokens'] <= 512
    assert all(len(c['requests']) == 2 and [r['label'] for r in c['requests']] == ['orig', 'edit']
               for c in a if c['category'] == 'agent')


@test
def long_answers_recomputable_from_text():
    for c in [c for c in cases.materialize() if c['category'] == 'long']:
        content = c['requests'][0]['messages'][0]['content']
        expected = c['requests'][0]['check']['value']
        journal = content.split('<журнал>\n', 1)[1].split('</журнал>', 1)[0]
        lines = journal.splitlines()
        hay = [l for l in lines if l.startswith('Запись ')]
        assert len(hay) == cases.LONG_LINES[c['length_target']]
        for word in ('кг', 'тариф', 'табельн', 'ВАЖНО', 'контейнер', 'ключ'):
            assert not any(word.lower() in l.lower() for l in hay), (c['id'], word)
        kind = c['id'].split('-')[-1]
        if kind == 'facts':
            code = re.search(r'код текущей задачи — (\S+)\.', journal).group(1)
            count = int(re.search(rf'для задачи {code} требуется (\d+) деталей', journal).group(1))
            owner = re.search(rf'ответственный за задачу {code} — (.+?)\.\n', journal).group(1)
            price = int(re.search(r'Цена одной детали — (\d+) рублей', content).group(1))
            got = {'code': code, 'count': count, 'owner': owner, 'total': count * price}
        elif kind == 'countsum':
            hits = re.findall(r'контейнер К-(\d+) весит (\d+) кг', journal)
            got = {'count': len(hits), 'total_kg': sum(int(w) for _, w in hits)}
        elif kind == 'hop':
            n = re.search(r'табельным номером (\d+)', journal).group(1)
            got = {'tab_number': int(n), 'name': re.search(rf'Владелец табельного номера {n} — (.+?)\.\n', journal).group(1)}
            assert len(re.findall(r'Владелец табельного номера \d+ — ', journal)) == 3
        else:
            values = re.findall(r'(?:установлен: |теперь )(\d+) рублей', journal)
            days = int(re.search(r'за (\d+) суток', content).group(1))
            got = {'tariff': int(values[-1]), 'cost': int(values[-1]) * days}
        assert got == expected, (c['id'], got, expected)
        # needle positions follow the frozen fractions
        _, _, positions = next(t for t in cases.LONG_TASKS if t[0] == kind)
        needle_idx = [i for i, l in enumerate(lines) if not l.startswith('Запись ')]
        fracs = sorted(positions[c['length_target']])
        for i, f in zip(needle_idx, fracs):
            assert abs(i / len(hay) - f) < 0.01, (c['id'], i, f)


@test
def code_references_pass_and_mutants_fail():
    suite = cases.materialize()
    for c in suite:
        for r in c['requests']:
            if r['kind'] != 'code':
                continue
            key = r['function'] if r['function'] != 'clamp' else f"clamp-{r['label']}"
            res = sandbox.run_tests(REF[key], r['function'], r['tests'])
            assert res['failure_code'] is None and res['tests_passed'] == len(r['tests']), (c['id'], res)
            wrong = REF[key].replace('return', 'return None or', 1) if 'return' in REF[key] else REF[key]
            res = sandbox.run_tests('def ' + r['function'] + '(*a):\n    return 12345\n', r['function'], r['tests'])
            assert res['failure_code'] == 'test_failed', (c['id'], res)
    # the edited clamp must fail the orig version's swap tests (edit is discriminating)
    edit = next(r for c in suite if c['id'] == 'agent-05' for r in c['requests'] if r['label'] == 'edit')
    assert sandbox.run_tests(REF['clamp-orig'], 'clamp', edit['tests'])['failure_code'] == 'test_failed'


# ----------------------------------------------------------------------------- sandbox
@test
def sandbox_policy_blocks_escapes():
    bad = {
        'import os\ndef f():\n    return 1\n': 'policy_import',
        'def f():\n    import os\n    return 1\n': 'policy_import',
        'def f():\n    return __import__("os")\n': 'policy_name',
        'def f():\n    return open("/etc/passwd")\n': 'policy_name',
        'def f():\n    return open("/etc/passwd").read()\n': 'policy_attribute',
        'def f():\n    return ().__class__\n': 'policy_attribute',
        'def f():\n    g = (i for i in [1])\n    return g.gi_frame.f_back\n': 'policy_attribute',
        'def f():\n    return getattr(1, "real")\n': 'policy_name',
        'class A:\n    pass\ndef f():\n    return 1\n': 'policy_node',
        '@staticmethod\ndef f():\n    return 1\n': 'policy_node',
        'def f():\n    with x:\n        pass\n': 'policy_node',
        'def f():\n    global y\n    return 1\n': 'policy_node',
        'def f(:\n': 'syntax_error',
        '': 'no_code',
        'def __init__():\n    return 1\n': 'policy_name',
        'def f():\n    return eval("1")\n': 'policy_name',
        'def f():\n    print(1)\n': 'policy_name',
    }
    for code, expected in bad.items():
        assert sandbox.check_policy(code)[0] == expected, (code, sandbox.check_policy(code)[0])
    ok, src, stripped = sandbox.check_policy('"""doc"""\ndef f(x):\n    return x\nprint(f(1))\nif __name__ == "__main__":\n    f(2)\n')
    assert ok is None and stripped == 3 and 'print' not in src
    assert sandbox.extract_code('text\n```python\ndef f():\n    return 1\n```\nmore') == 'def f():\n    return 1'
    assert sandbox.run_tests('x' * 20001, 'f', [])['failure_code'] == 'oversize'


@test
def sandbox_runtime_limits_defense_in_depth():
    original = sandbox.check_policy
    try:
        sandbox.check_policy = lambda code: (None, code, 0)  # bypass layer 1: layers 2-4 must still hold
        cases_ = {
            'import os\ndef f():\n    return 1\n': 'exec_error',
            'def f():\n    return __import__("os").getcwd()\n': 'test_failed',
            'def f():\n    while True:\n        pass\n': 'timeout',
            'def f():\n    return len([0] * (10 ** 10))\n': 'test_failed',
            'def f():\n    return f()\n': 'test_failed',
            'def g():\n    return 1\n': 'missing_function',
            'def f():\n    return open("x", "w")\n': 'test_failed',
        }
        for code, expected in cases_.items():
            got = sandbox.run_tests(code, 'f', [[[], 1]])
            assert got['failure_code'] == expected, (code, got)
        # the child cannot forge a verdict: it has no print/sys and does not know the nonce
        spoof = 'def f():\n    return 1\nX = \'{"nonce": "0", "results": [true]}\'\n'
        assert sandbox.run_tests(spoof, 'f', [[[], 2]])['failure_code'] == 'test_failed'
    finally:
        sandbox.check_policy = original
    res = sandbox.run_tests('def f(a):\n    return a\n', 'f', [[[1], 1], [[True], 1]])
    assert res['failure_code'] == 'test_failed' and res['tests_passed'] == 1  # bool is not int (strict types)
    # the network namespace layer, when available, really has no network
    if sandbox.network_isolation_available():
        probe = subprocess.run(sandbox._NETNS + [sys.executable, '-I', '-S', '-c',
                               'import socket\ns=socket.socket()\ns.settimeout(2)\ns.connect(("1.1.1.1", 53))'],
                               capture_output=True, timeout=20)
        assert probe.returncode != 0


# ----------------------------------------------------------------------------- checkers
@test
def text_checkers():
    ct = cases.check_text
    assert ct({'type': 'number', 'value': 9000}, '9 000') == []
    assert ct({'type': 'number', 'value': 9000}, 'Ответ: 9000 секунд.') == []
    assert ct({'type': 'number', 'value': 16}, '28-12=16') == ['number_mismatch']
    assert ct({'type': 'number', 'value': 1.65}, '1,65') == []
    assert ct({'type': 'number', 'value': 1.8}, '1,80 м') == []
    assert ct({'type': 'number', 'value': 2004}, '') == ['empty']
    w = {'type': 'words', 'accept': ['отчет'], 'reject': ['почта'], 'max_words': 4}
    assert ct(w, '**Отчёт**') == [] and ct(w, 'почта') == ['answer_missing', 'forbidden_present']
    assert ct({'type': 'words', 'accept': ['улан-удэ'], 'max_words': 5}, 'Улан‑Удэ.') == []
    assert ct({'type': 'words', 'accept': ['нет'], 'reject': ['да'], 'max_words': 6}, 'Нет, дождя не было.') == []
    assert ct({'type': 'words', 'accept': ['окон'], 'max_words': 3}, 'Слово окон в родительном падеже') == ['too_long']
    s = {'type': 'sequence', 'value': ['абрикос', 'вишня']}
    assert ct(s, 'Абрикос, вишня.') == [] and ct(s, 'вишня, абрикос') == ['sequence_mismatch']
    j = {'type': 'json', 'value': {'a': 34, 'b': 'Ё'}}
    assert ct(j, 'вот: {"b": "Ё", "a": 34}') == [] and ct(j, '{"a": 34.0, "b": "Ё"}') == ['json_mismatch']
    assert ct(j, '{"a": 34, "a": 34, "b": "Ё"}') == ['json_missing'] and ct(j, 'нет json') == ['json_missing']
    p = {'type': 'patterns', 'patterns': ['сред', '16:30'], 'forbid': ['14:00']}
    assert ct(p, 'Среда, 16:30') == [] and ct(p, 'Среда, 16:30 (было 14:00)') == ['forbidden_present']
    raw = {'type': 'patterns', 'raw': True, 'patterns': ['Париж'], 'forbid': ['ПАРИЖ']}
    assert ct(raw, 'Париж') == [] and ct(raw, 'ПАРИЖ') == ['pattern_missing', 'forbidden_present']
    assert ct({'type': 'patterns', 'patterns': [r'(?:-|минус\s*)3(?!\d)']}, 'в Томске −3 °C') == []


@test
def tool_call_validation():
    req = cases.materialize()[[c['id'] for c in cases.materialize()].index('tool-12')]['requests'][0]
    exp = req['calls']

    def msg(*calls):
        return {'role': 'assistant', 'tool_calls': [{'id': i, 'type': 'function', 'function': {'name': n, 'arguments': a}}
                                                    for i, n, a in calls]}
    good = '{"city": "Пермь", "unit": "celsius"}'
    codes, m = cases.validate_tool_calls(msg(('c1', 'get_weather', good)), 'tool_calls', 0, req, exp, set())
    assert codes == [] and len(m) == 1
    both = msg(('c1', 'get_weather', good), ('c2', 'get_weather', '{"city":"Сочи","unit":"celsius"}'))
    assert cases.validate_tool_calls(both, 'tool_calls', 0, req, exp, set())[0] == []
    dup = msg(('c1', 'get_weather', good), ('c1', 'get_weather', '{"city":"Сочи","unit":"celsius"}'))
    assert 'duplicate_id' in cases.validate_tool_calls(dup, 'tool_calls', 0, req, exp, set())[0]
    assert cases.validate_tool_calls(msg(('bad id!', 'get_weather', good)), 'tool_calls', 0, req, exp, set())[0] == ['bad_id']
    assert cases.validate_tool_calls(msg(('c1', 'get_weather', '{bad')), 'tool_calls', 0, req, exp, set())[0] == ['bad_json']
    assert cases.validate_tool_calls(msg(('c1', 'get_weather', '{"city":"Пермь","unit":"kelvin"}')), 'tool_calls', 0, req, exp,
                                     set())[0] == ['bad_args']
    assert cases.validate_tool_calls(msg(('c1', 'find_employee', '{"name":"x"}')), 'tool_calls', 0, req, exp,
                                     set())[0] == ['unexpected_call']
    assert cases.validate_tool_calls(msg(('c1', 'nope', '{}')), 'stop', 5, req, exp, set())[0] == \
        ['reasoning_present', 'bad_finish', 'bad_name']
    assert not cases.blocking(['bad_finish', 'reasoning_present']) and cases.blocking(['bad_args'])
    add = next(c for c in cases.materialize() if c['id'] == 'tool-04')['requests'][0]
    float_arg = msg(('c1', 'add_numbers', '{"a": -38.0, "b": 125}'))
    assert cases.validate_tool_calls(float_arg, 'tool_calls', 0, add, add['calls'], set())[0] == ['bad_args']
    cur = next(c for c in cases.materialize() if c['id'] == 'tool-02')['requests'][0]
    int_for_number = msg(('c1', 'convert_currency', '{"amount": 250.0, "from_currency": "USD", "to_currency": "EUR"}'))
    assert cases.validate_tool_calls(int_for_number, 'tool_calls', 0, cur, cur['calls'], set())[0] == []


# ----------------------------------------------------------------------------- runner
@test
def runner_oracle_all_pass_and_mutations():
    suite = cases.materialize()
    rows = ev.execute_suite(Oracle(suite), suite, progress=False)
    failed = [(r['id'], r['failure_codes']) for r in rows if not r['pass']]
    assert not failed, failed
    summary = ev.summarize(rows)
    assert summary['pass'] == 60 and all(v['pass'] == 12 for v in summary['by_category'].values())
    blob = json.dumps(rows, ensure_ascii=False)
    assert 'Казани' not in blob and 'def rotate_right' not in blob  # no reply text in rows

    def wrong(case_id, label, message, finish):
        if message.get('tool_calls'):
            for call in message['tool_calls']:
                call['function']['arguments'] = '{"x": 1}'
            return message, finish
        return {'role': 'assistant', 'content': 'не знаю'}, 'stop'
    rows = ev.execute_suite(Oracle(suite, wrong), suite, progress=False)
    assert sum(r['pass'] for r in rows) == 0
    codes = {r['id']: r['failure_codes'] for r in rows}
    assert codes['tool-01'] == ['bad_args'] and codes['tool-07'] == ['pattern_missing']
    assert codes['code-01'] == ['syntax_error'] and codes['long-16k-facts'] == ['json_missing']

    def no_call(case_id, label, message, finish):
        if case_id == 'tool-03' and message.get('tool_calls') and 'TRK' in json.dumps(message, ensure_ascii=False):
            return {'role': 'assistant', 'content': 'Заказ отправлен.'}, 'stop'
        if case_id == 'tool-07':
            return {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'c', 'type': 'function', 'function': {
                'name': 'get_weather', 'arguments': '{"city":"Лондон","unit":"celsius"}'}}]}, 'tool_calls'
        if case_id == 'tool-01':
            return {'role': 'assistant', 'content': 'Не могу.'}, 'stop'
        if case_id == 'tool-05' and finish == 'tool_calls':
            return message, 'stop'
        return message, finish
    rows = {r['id']: r for r in ev.execute_suite(Oracle(suite, no_call), suite, progress=False)}
    assert rows['tool-03']['failure_codes'] == ['missing_call'] and rows['tool-07']['failure_codes'] == ['unexpected_call']
    assert rows['tool-01']['failure_codes'] == ['no_call'] and rows['tool-05']['failure_codes'] == ['bad_finish']

    def infra(case_id, label, message, finish):
        if case_id == 'ru-01':
            return (503, None), None
        if case_id == 'ru-02':
            return (None, None), None
        if case_id == 'tool-03' and message.get('tool_calls') and 'TRK' in json.dumps(message, ensure_ascii=False):
            return (400, None), None
        return message, finish
    rows = {r['id']: r for r in ev.execute_suite(Oracle(suite, infra), suite, progress=False)}
    assert rows['ru-01']['failure_codes'] == ['http_5xx'] and rows['ru-01']['infra_error']
    assert rows['ru-02']['failure_codes'] == ['transport_error'] and rows['ru-02']['infra_error']
    assert rows['tool-03']['failure_codes'] == ['http_4xx_after_model_output'] and not rows['tool-03']['infra_error']


@test
def run_mode_receipts_and_validity():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix='evalv1-test-'))
    try:
        with_freeze(temp_freeze(tmp))
        suite = cases.materialize()
        root = make_root(tmp, 'bf128')
        assert ev.cmd_run(args(name='t1', expect_profile='bf128'), root=root, http=Oracle(suite)) == 0
        rep = json.loads((root / 'reports/t1-eval-v1.json').read_text())
        assert rep['status'] == 'VALID' and rep['summary']['pass'] == 60 and rep['invalid_codes'] == []
        assert oct((root / 'reports/t1-eval-v1.json').stat().st_mode & 0o777) == '0o600'
        assert ev.cmd_run(args(name='t1', expect_profile='bf128'), root=root, http=Oracle(suite)) == ev.EXIT_REFUSED
        # wrong profile configuration / runtime / receipt
        root2 = make_root(tmp / 'b', 'bf128', config=ev.PROFILES['bf128-fp16compat'])
        assert ev.cmd_run(args(name='t2', expect_profile='bf128'), root=root2, http=Oracle(suite)) == 1
        assert json.loads((root2 / 'reports/t2-eval-v1.json').read_text())['invalid_codes'] == ['profile_config_mismatch']
        root3 = make_root(tmp / 'c', 'bf128-fp16compat', runtime_ok=False)
        assert ev.cmd_run(args(name='t3', expect_profile='bf128-fp16compat'), root=root3, http=Oracle(suite)) == 1
        assert json.loads((root3 / 'reports/t3-eval-v1.json').read_text())['invalid_codes'] == ['runtime_mismatch']
        root4 = make_root(tmp / 'd', 'bf128')
        assert ev.cmd_run(args(name='t4', expect_profile='bf128-fp16compat'), root=root4, http=Oracle(suite)) == 1
        assert json.loads((root4 / 'reports/t4-eval-v1.json').read_text())['invalid_codes'] == ['receipt_invalid']
        # serving generation changes mid-run
        root5 = make_root(tmp / 'e', 'bf128')
        oracle = Oracle(suite)

        def flip(payload):
            if oracle.calls == 30:
                (root5 / 'active-run.json').write_text(json.dumps(receipt('bf128', run_id='e' * 32)))
            return oracle(payload)
        assert ev.cmd_run(args(name='t5', expect_profile='bf128'), root=root5, http=flip) == 1
        assert json.loads((root5 / 'reports/t5-eval-v1.json').read_text())['invalid_codes'] == ['serving_generation_changed']
        # fixture drift is refused before any request
        broken = json.loads(ev.FREEZE.read_text())
        broken['cases'][0]['sha256'] = '0' * 64
        bad = tmp / 'bad-freeze.json'
        bad.write_text(json.dumps(broken))
        with_freeze(bad)
        assert ev.cmd_run(args(name='t6', expect_profile='bf128'), root=root, http=Oracle(suite)) == ev.EXIT_FIXTURE
        assert not (root / 'reports/t6-eval-v1.json').exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def managed_mode_lock_start_stop():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix='evalv1-test-'))
    try:
        with_freeze(temp_freeze(tmp))
        suite = cases.materialize()
        profile = 'bf128-fp16compat'
        root = make_root(tmp, profile)
        workspace = tmp / 'ws'
        workspace.mkdir()
        (workspace / f'{profile}.json').write_text(json.dumps(ev.PROFILES[profile]))
        for rel in ev.RUNTIME_FILES:
            target = root / 'python' / 'freetoken' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ev.runtime_file(rel), target)
        lock = tmp / 'gpu.lock'
        log = []
        units = {'qwen36-inference': 'inactive', 'qwen36-experiment': 'inactive'}
        inv = {'id': ''}

        def state(prop, unit='qwen36-experiment'):
            if prop == 'ActiveState':
                return units[unit]
            return inv['id']

        def command(argv, timeout):
            log.append(argv[:3] if argv[0] != sys.executable else ['readiness'])
            if argv[0] == 'bash':
                units['qwen36-experiment'] = 'active'
                inv['id'] = 'inv-1'
            if argv[:3] == ['systemctl', '--user', 'stop']:
                units['qwen36-experiment'] = 'inactive'
                inv['id'] = ''
            return ''
        launcher = lambda p, r: {'files': {}, 'environment_sha256': 'x', 'problems': [], 'main_pid': 7, 'invocation_id': 'inv-1'}
        a = args(name='m1', profile=profile, ready_timeout=600)
        rc = ev.cmd_managed(a, root=root, workspace=workspace, lock_path=lock, http=Oracle(suite), state=state,
                            command=command, launcher=launcher)
        rep = json.loads((root / 'reports/m1-eval-v1.json').read_text())
        assert rc == 0 and rep['status'] == 'VALID' and rep['managed']['stopped'] and rep['managed']['step'] == 'done'
        assert log == [['bash', str(root / 'start-experiment.sh'), profile], ['readiness'], ['systemctl', '--user', 'stop']]
        # lock busy -> 75, nothing started
        held = bench.acquire_gpu_lock(lock)
        log.clear()
        assert ev.cmd_managed(args(name='m2', profile=profile, ready_timeout=600), root=root, workspace=workspace,
                              lock_path=lock, http=Oracle(suite), state=state, command=command, launcher=launcher) == ev.EXIT_LOCK
        os.close(held)
        assert log == [] and not (root / 'reports/m2-eval-v1.json').exists()
        # a unit is already active -> 73, no report, no stop of a foreign unit
        units['qwen36-inference'] = 'active'
        assert ev.cmd_managed(args(name='m3', profile=profile, ready_timeout=600), root=root, workspace=workspace,
                              lock_path=lock, http=Oracle(suite), state=state, command=command, launcher=launcher) == ev.EXIT_ACTIVE
        units['qwen36-inference'] = 'inactive'
        assert log == [] and not (root / 'reports/m3-eval-v1.json').exists()
        # readiness fails -> INVALID report, own unit stopped
        def failing(argv, timeout):
            if argv[0] == sys.executable:
                raise RuntimeError('not ready')
            return command(argv, timeout)
        assert ev.cmd_managed(args(name='m4', profile=profile, ready_timeout=600), root=root, workspace=workspace,
                              lock_path=lock, http=Oracle(suite), state=state, command=failing, launcher=launcher) == 1
        rep = json.loads((root / 'reports/m4-eval-v1.json').read_text())
        assert rep['invalid_codes'] == ['managed_failed'] and rep['managed']['stopped'] and units['qwen36-experiment'] == 'inactive'
        # launcher problems -> INVALID even if every case passes
        bad_launcher = lambda p, r: dict(launcher(p, r), problems=['feature_env_in_launcher'])
        assert ev.cmd_managed(args(name='m5', profile=profile, ready_timeout=600), root=root, workspace=workspace,
                              lock_path=lock, http=Oracle(suite), state=state, command=command, launcher=bad_launcher) == 1
        assert json.loads((root / 'reports/m5-eval-v1.json').read_text())['invalid_codes'] == ['launcher_problems']
        # native runtime not synced -> refused before the lock
        (root / 'python/freetoken/models/nvfp4_banks.py').write_text('old')
        assert ev.cmd_managed(args(name='m6', profile=profile, ready_timeout=600), root=root, workspace=workspace,
                              lock_path=lock, http=Oracle(suite), state=state, command=command, launcher=launcher) == ev.EXIT_FIXTURE
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------- compare
def fake_reports(tmp, lose=(), win=(), base_invalid=False, cand_env_extra=None):
    suite = cases.materialize()
    out = {}
    for name, profile, run_id in (('base', 'bf128-fp16compat', 'a' * 32), ('cand', 'bf128', 'b' * 32)):
        root = make_root(tmp / name, profile, run_id=run_id)
        oracle = Oracle(suite)
        if name == 'cand' or win:
            flip = set(lose) if name == 'cand' else set(win)

            def mutate(case_id, label, message, finish, flip=flip):
                if case_id in flip:
                    if message.get('tool_calls'):
                        message['tool_calls'][0]['function']['arguments'] = '{"x":1}'
                        return message, finish
                    return {'role': 'assistant', 'content': 'не знаю'}, 'stop'
                return message, finish
            oracle = Oracle(suite, mutate)
        ev.cmd_run(args(name=name, expect_profile=profile), root=root, http=oracle)
        out[name] = json.loads((root / f'reports/{name}-eval-v1.json').read_text())
    if base_invalid:
        out['base']['status'] = 'INVALID'
    if cand_env_extra:
        out['cand']['run']['configuration']['env'].update(cand_env_extra)
    croot = tmp / 'croot'
    (croot / 'reports').mkdir(parents=True)
    for name, rep in out.items():
        (croot / f'reports/{name}-eval-v1.json').write_text(json.dumps(rep))
    return croot


def teacher_jsonl(path, growth_per_token):
    lines = []
    for run, delta in (('r16', 0.0), ('r32', growth_per_token)):
        for i in range(20):
            nll = [1.0 + 0.01 * ((i * 7 + j) % 5) for j in range(40)]
            lines.append(json.dumps({'profile': 'teacher-bf128', 'run_id': run, 'case': f'c{i}',
                                     'nll': [v + delta + 0.002 * (((i + j) % 3) - 1) for j, v in enumerate(nll)]}))
    lines.append(json.dumps({'profile': 'teacher-fp128', 'run_id': 'zz', 'case': 'c0', 'nll': [9.0]}))
    path.write_text('\n'.join(lines) + '\n')


def run_compare(tmp, croot, growth=-0.001, name='cmp'):
    teacher_jsonl(croot / 'reports/teacher.jsonl', growth)
    rc = cmp_.main(['--name', name, '--baseline', 'base', '--candidate', 'cand', '--fp16-teacher-run-id', 'r16',
                    '--fp32-teacher-run-id', 'r32'], root=croot, freeze_path=ev.FREEZE)
    return rc, json.loads((croot / f'reports/{name}.json').read_text())


@test
def compare_decisions():
    assert cmp_.mcnemar_exact(5, 0) == 0.0625 and cmp_.mcnemar_exact(6, 0) == 0.03125
    assert cmp_.mcnemar_exact(3, 3) == 1.0 and cmp_.mcnemar_exact(0, 0) == 1.0
    assert abs(cmp_.mcnemar_exact(1, 7) - 0.0703125) < 1e-12
    tmp = pathlib.Path(tempfile.mkdtemp(prefix='evalv1-test-'))
    try:
        with_freeze(temp_freeze(tmp))
        scenarios = [
            ('equal', {}, -0.001, 'fp32', {}),
            ('one-loss', {'lose': ('ru-03',)}, -0.001, 'fp32', {}),
            ('long-net-2', {'lose': ('long-16k-facts', 'long-64k-hop')}, -0.001, 'fp16-compat', {'eval_categories': False}),
            ('long-2-1', {'lose': ('long-16k-facts', 'long-64k-hop'), 'win': ('long-120k-latest',)}, -0.001, 'fp32', {}),
            ('total-2', {'lose': ('code-01', 'ru-01')}, -0.001, 'fp16-compat', {'eval_total': False}),
            ('tool-1', {'lose': ('tool-04',)}, -0.001, 'fp16-compat', {'tool': False}),
            ('nll-worse', {}, 0.005, 'fp16-compat', {'nll': False}),
        ]
        for label, kw, growth, decision, failing in scenarios:
            croot = fake_reports(tmp / label, **kw)
            rc, rep = run_compare(tmp, croot, growth)
            assert rc == 0 and rep['decision'] == decision, (label, rep['decision'], rep['criteria'], rep['problems'])
            for key, value in failing.items():
                assert rep['criteria'][key]['pass'] is value, (label, key)
        croot = fake_reports(tmp / 'disc', lose=('code-02', 'agent-01', 'agent-02'), win=('code-05',))
        rc, rep = run_compare(tmp, croot)
        overall = rep['pairs']['overall']
        assert overall['baseline_only_pass'] == ['agent-01', 'agent-02', 'code-02'] and overall['candidate_only_pass'] == ['code-05']
        assert overall['net_loss'] == 2 and overall['mcnemar_exact_two_sided_p'] == 0.625 and rep['decision'] == 'fp16-compat'
        rc, rep = run_compare(tmp, fake_reports(tmp / 'inv', base_invalid=True))
        assert rc == 1 and rep['decision'] == 'NO_DECISION' and 'baseline_invalid' in rep['problems']
        rc, rep = run_compare(tmp, fake_reports(tmp / 'env', cand_env_extra={'QWEN36_OTHER': '1'}))
        assert rc == 1 and 'configurations_differ_beyond_compat_flag' in rep['problems']
        croot = fake_reports(tmp / 'cherry')
        earlier = json.loads((croot / 'reports/cand-eval-v1.json').read_text())
        earlier.update(name='cand0', started_utc='2000-01-01T00:00:00Z')
        (croot / 'reports/cand0-eval-v1.json').write_text(json.dumps(earlier))
        rc, rep = run_compare(tmp, croot)
        assert rc == 1 and rep['problems'] == ['candidate_not_first_valid_run'], rep['problems']
        croot = fake_reports(tmp / 'nll')
        teacher_jsonl(croot / 'reports/teacher.jsonl', 0.0)
        rc = cmp_.main(['--name', 'x', '--baseline', 'base', '--candidate', 'cand', '--fp16-teacher-run-id', 'r16',
                        '--fp32-teacher-run-id', 'nope'], root=croot, freeze_path=ev.FREEZE)
        assert rc == 1 and 'nll_run_missing' in json.loads((croot / 'reports/x.json').read_text())['problems']
        assert cmp_.main(['--name', 'x', '--baseline', 'base', '--candidate', 'cand', '--fp16-teacher-run-id', 'r16',
                          '--fp32-teacher-run-id', 'r32'], root=croot, freeze_path=ev.FREEZE) == 2  # O_EXCL
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def frozen_freeze_verifies_if_present():
    real = HERE / 'eval-v1' / 'freeze.json'
    if not real.exists():
        return 'skipped: no freeze yet'
    ev.FREEZE = real
    suite, freeze, problems = ev.verify_fixture(real)
    assert problems == [] and freeze['case_count'] == 60
    assert freeze['decision_rule'] == cmp_.RULE
    assert freeze['runtime_expectations']['source_files'] == {r: ev.sha256_file(ev.runtime_file(r)) for r in ev.RUNTIME_FILES}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out')
    a = p.parse_args()
    results, ok = [], True
    for fn in TESTS:
        saved = ev.FREEZE
        try:
            note = fn()
            results.append({'test': fn.__name__, 'pass': True, **({'note': note} if note else {})})
        except Exception as error:  # noqa: BLE001
            ok = False
            results.append({'test': fn.__name__, 'pass': False, 'error': f'{type(error).__name__}: {str(error)[:300]}',
                            'trace': traceback.format_exc()[-1500:]})
        finally:
            ev.FREEZE = saved
        print(json.dumps(results[-1], ensure_ascii=False)[:2000], flush=True)
    summary = {'passed': sum(r['pass'] for r in results), 'total': len(results),
               'network_isolated_sandbox': sandbox.network_isolation_available(),
               'sources': ev.source_hashes(), 'results': results}
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('passed', 'total', 'network_isolated_sandbox')}))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
