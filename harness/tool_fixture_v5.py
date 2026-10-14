"""Versioned natural-EOS auto tool-cycle fixture V5: fixture_add(17,25) -> 42.

Patterned after the Russian quality.py tool cycle (user-only Russian request, the
model must obtain the value from the tool and must not invent it). It is a NEW
fixture: it inherits no PASS from quality.py, network-acceptance.py or V4.

Contract (not relaxed):
  step 1  tool_choice=auto must return exactly one call: type=function, valid
          string id, name fixture_add, JSON-string arguments that decode to an
          object of exactly two integers a=17, b=25, finish_reason=tool_calls.
          Absence of a call is FAIL (no_call), never a PASS.
  step 2  the genuine local result {"result":42} is returned with the same id;
          the final answer must contain 42, without further calls, and stop.
Only tool_choice=auto is used: required/named are rejected (400) by the guard.

The English V4 fixture is retired with its baseline FAIL preserved below.
"""
import copy, hashlib, json, re

FIXTURE_ID = 'tool-cycle-ru-fixture-add-v5'
FIXTURE_VERSION = 5
TOOL_CHOICE = 'auto'
SAMPLING = {'seed': 17, 'temperature': 0, 'top_p': 1}
STEP_LIMITS = (128, 96)  # maximum tokens: tool-call step, final-answer step
TOOLS = [{'type': 'function', 'function': {
    'name': 'fixture_add',
    'description': 'Return the synthetic fixture sum of two integers.',
    'parameters': {'type': 'object',
                   'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                   'required': ['a', 'b'], 'additionalProperties': False}}}]
FIRST_MESSAGES = [{'role': 'user', 'content':
    'Вызови fixture_add с a=17 и b=25, затем сообщи результат из ответа функции. '
    'Не вычисляй и не придумывай результат сам.'}]
EXPECTED_NAME = 'fixture_add'
EXPECTED_ARGS = {'a': 17, 'b': 25}
EXPECTED_RESULT = 42
CALL_ID_PATTERN = re.compile(r'[A-Za-z0-9_-]{1,128}')
FINAL_PATTERN = re.compile(r'(?<![0-9])42(?![0-9])')

# Static failure codes, in canonical (most fundamental first) order. Reports carry
# these codes, never raw text/arguments; failure_code is the first one present.
FIRST_STEP_CODES = ('transport_error', 'http_status', 'bad_response_schema', 'no_call', 'wrong_count',
                    'bad_type', 'bad_id', 'bad_name', 'bad_json', 'bad_args', 'bad_finish',
                    'reasoning_present')
CYCLE_CODES = ('id_mismatch',)
FINAL_STEP_CODES = ('final_transport_error', 'final_http_status', 'final_bad_response_schema',
                    'final_reasoning_present', 'final_unexpected_tool_call', 'final_bad_finish',
                    'final_missing_42')
REPORT_CODES = ('receipt_invalid', 'serving_generation_changed')
FAILURE_CODES = FIRST_STEP_CODES + CYCLE_CODES + FINAL_STEP_CODES + REPORT_CODES
# A tool result may be sent only when these did not occur: the call identity and
# arguments are then fully validated. finish/reasoning defects still fail the row.
_BLOCKING_FIRST = frozenset(FIRST_STEP_CODES) - {'reasoning_present', 'bad_finish'}

RETIRED_FIXTURES = {
    'tool-cycle-en-v4': {
        'status': 'FAIL',
        'meaning': 'Real baseline failure on this English fixture; not a parser/ID/JSON bug. Not renamed, not relabelled PASS.',
        'observed_on': 'bf128 final3-v4 repeat 0 and bounded standalone diagnostic',
        'observed': {'http_status': 200, 'first_finish': 'stop', 'completion_tokens': 3,
                     'tool_calls': 0, 'reasoning_chars': 0},
        'evidence': ['outputs/qwen36-api/optimization/tool-task-v4-diagnostic.json',
                     '/home/USER/qwen36-api/optimization/reports/final3-v4-0-bf128-tool-task.json',
                     '/home/USER/qwen36-api/optimization/reports/final3-v4-matrix.json'],
        'tool_task_source_sha256': '7a4e27d1d2fa212b1a3a4a6a7903ddd1fe0fd818255300728a0d3409c67fb34c',
        'task_fixture_sha256_by_repeat': [
            'eee3aa24b74a2753835bf8532a9d33b8f0a8713273c3b896435cac3c4998cdc0',
            'ccb6e3eafe87023cc500f770d5fd90a84a840486948d48cf5927a1bf5890e84c',
            '87cd791efa2112efcfcf4519d8f905ad5c414f669348ac02de299edd9888c72e',
            'c80af7fab375b31e724b0c687a7702f006d9171201c9507ac7a6576a0caf3ea3',
            'e2160387e93bebc4290a9f04fb80613406e2660f50330dfe19cd18172e3eef5d'],
        'replaced_by': FIXTURE_ID,
        'inherits_pass': False}}


def ordered(codes):
    """Deduplicate and sort codes in canonical order; unknown codes are a harness bug."""
    unknown = set(codes) - set(FAILURE_CODES)
    if unknown:
        raise RuntimeError('unregistered tool failure code')
    return sorted(set(codes), key=FAILURE_CODES.index)


def _canonical(value):
    # Same digest representation as bench.canonical_hash; no heavy imports here.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def first_messages():
    """Repeat-independent input: the pilot qualifies exactly what the matrix sends."""
    return copy.deepcopy(FIRST_MESSAGES)


def task_fixture_sha256():
    return _canonical({'messages': FIRST_MESSAGES, 'tools': TOOLS})


def fixture_sha256():
    """Identity of the complete fixture contract, frozen into the V5 manifest."""
    return _canonical({'fixture_id': FIXTURE_ID, 'fixture_version': FIXTURE_VERSION,
                       'messages': FIRST_MESSAGES, 'tools': TOOLS, 'tool_choice': TOOL_CHOICE,
                       'sampling': SAMPLING, 'step_limits': list(STEP_LIMITS),
                       'expected_name': EXPECTED_NAME, 'expected_args': EXPECTED_ARGS,
                       'expected_result': EXPECTED_RESULT, 'call_id_pattern': CALL_ID_PATTERN.pattern,
                       'final_pattern': FINAL_PATTERN.pattern, 'failure_codes': list(FAILURE_CODES)})


def describe():
    return {'id': FIXTURE_ID, 'version': FIXTURE_VERSION, 'sha256': fixture_sha256(),
            'task_fixture_sha256': task_fixture_sha256(), 'tool_choice': TOOL_CHOICE}


def _strict_object(pairs):
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate key')
    return dict(pairs)


def _reject_constant(_name):
    raise ValueError('non-finite constant')


def validate_first(message, finish_reason, reasoning_chars):
    """Return (codes, call, value). call/value are set only for a fully valid call."""
    if not isinstance(message, dict) or message.get('role') != 'assistant':
        return ['bad_response_schema'], None, None
    codes = []
    if reasoning_chars:
        codes.append('reasoning_present')
    if finish_reason != 'tool_calls':
        codes.append('bad_finish')
    calls = message.get('tool_calls')
    if calls is None or calls == []:
        return codes + ['no_call'], None, None
    if not isinstance(calls, list):
        return codes + ['bad_response_schema'], None, None
    if len(calls) != 1:
        return codes + ['wrong_count'], None, None
    call = calls[0]
    if not isinstance(call, dict):
        return codes + ['bad_type'], None, None
    if call.get('type') != 'function':
        codes.append('bad_type')
    call_id = call.get('id')
    if not isinstance(call_id, str) or CALL_ID_PATTERN.fullmatch(call_id) is None:
        codes.append('bad_id')
    function = call.get('function')
    if not isinstance(function, dict):
        return codes + ['bad_name', 'bad_json'], None, None
    if function.get('name') != EXPECTED_NAME:
        codes.append('bad_name')
    arguments = function.get('arguments')
    args = None
    if not isinstance(arguments, str):
        codes.append('bad_json')
    else:
        try:
            args = json.loads(arguments, object_pairs_hook=_strict_object, parse_constant=_reject_constant)
        except (ValueError, TypeError, RecursionError):
            codes.append('bad_json')
    if 'bad_json' not in codes:
        if (not isinstance(args, dict) or set(args) != set(EXPECTED_ARGS)
                or any(type(args[key]) is not int for key in args) or args != EXPECTED_ARGS):
            codes.append('bad_args')
    if _BLOCKING_FIRST.intersection(codes):
        return codes, None, None
    # Genuine local tool execution: integer addition of validated arguments only.
    return codes, copy.deepcopy(call), args['a'] + args['b']


def validate_final(message, finish_reason, reasoning_chars):
    if not isinstance(message, dict) or message.get('role') != 'assistant':
        return ['final_bad_response_schema']
    codes = []
    if reasoning_chars:
        codes.append('final_reasoning_present')
    if message.get('tool_calls'):
        codes.append('final_unexpected_tool_call')
    if finish_reason != 'stop':
        codes.append('final_bad_finish')
    content = message.get('content')
    if not isinstance(content, str) or FINAL_PATTERN.search(content) is None:
        codes.append('final_missing_42')
    return codes
