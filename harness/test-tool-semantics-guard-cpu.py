"""ASGI guard checks for parallel_tool_calls, function strict and legacy functions.

Companion to test-tool-choice-guard-cpu.py (41 cases, left unchanged). Runs the
selected guard with a stub application: no model, CUDA, network or API secret.
Accepted payloads are also validated by the pinned FreeToken request schema, so
a guard PASS cannot hide a downstream 422. Report holds categories/counters only.
--gate-script (opt-in, default run unchanged) also requires that CPU gate script
to run both guard suites: test-guard.py alone passes a guard without v2 checks.
"""
import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

PRIVATE = 'SYNTHETIC_PRIVATE_SENTINEL_NOT_TO_BE_LOGGED'
TOOL_CHOICE = 'tool_choice supports only auto or none; forced calls are unavailable'
PARALLEL_TYPE = 'parallel_tool_calls must be a boolean'
PARALLEL_FALSE = 'parallel_tool_calls=false is unsupported; a single tool call is not enforced'
STRICT = 'strict function schemas are unsupported; omit strict or set it to false'
FUNCTIONS = 'Legacy functions are unsupported; use tools with tool_choice auto or none'
DEFAULT = object()
ABSENT = object()


def tool(name, **function_extra):
    return {'type':'function', 'function':{'name':name, 'parameters':{'type':'object'}, **function_extra}}


def build(*, tools=DEFAULT, history=False, **fields):
    """tools=DEFAULT: two declared tools; ABSENT: no tools key; otherwise literal value."""
    payload = {'model':'qwen36-35b', 'messages':[{'role':'user','content':PRIVATE}], 'max_tokens':8}
    if tools is DEFAULT:
        payload['tools'] = [tool('fixture'), tool('second_fixture')]
    elif tools is not ABSENT:
        payload['tools'] = tools
    payload.update(fields)
    if history:
        payload['messages'].extend([
            {'role':'assistant','tool_calls':[{'id':'fixture-id','type':'function',
                'function':{'name':'fixture','arguments':'{"value":1}'}}]},
            {'role':'tool','tool_call_id':'fixture-id','content':'{"value":2}'},
        ])
    return payload


async def exercise(module, payload, *, stream, auth=True):
    payload = dict(copy.deepcopy(payload), stream=stream)
    events = []
    counters = {'app_calls':0, 'metric_configure_calls':0, 'payload_unchanged':False}
    async def app(scope, receive, send):
        counters['app_calls'] += 1
        message = await receive()
        counters['payload_unchanged'] = json.loads(message['body']) == payload
        await send({'type':'http.response.start','status':200,'headers':[]})
        await send({'type':'http.response.body','body':b'{}'})
    def configured(_):
        counters['metric_configure_calls'] += 1
    # Fixed dummy key belongs only to this in-memory test and is never printed.
    with patch.dict(os.environ, {'QWEN36_API_KEY':'x'*43}):
        guard = module.APIGuard(app)
    first = True
    async def receive():
        nonlocal first
        if first:
            first = False
            return {'type':'http.request','body':json.dumps(payload).encode(),'more_body':False}
        await asyncio.Future()
    async def send(event):
        events.append(event)
    headers = [(b'authorization', b'Bearer '+b'x'*43)] if auth else []
    scope = {'type':'http','method':'POST','path':'/v1/chat/completions','headers':headers,
             'qwen36.metrics':SimpleNamespace(configure=configured)}
    await guard(scope, receive, send)
    starts = [e for e in events if e['type'] == 'http.response.start']
    assert len(starts) == 1
    body = b''.join(e.get('body',b'') for e in events if e['type'] == 'http.response.body')
    content_type = dict(starts[0]['headers']).get(b'content-type', b'').decode()
    try:
        message = json.loads(body).get('error', {}).get('message')
        json_body = True
    except ValueError:
        message, json_body = None, False
    return {'status':starts[0]['status'], **counters, 'slot_busy_after':guard.busy,
            'cleanup_failed':guard.cleanup_failed,
            'first_event_is_start':events[0]['type'] == 'http.response.start',
            'content_type':content_type, 'json_body':json_body,
            'sse_bytes':b'data:' in body or 'event-stream' in content_type,
            'private_marker_leaked':PRIVATE.encode() in body, 'error_message':message}


def schema_accepts(payload):
    from freetoken.server.api_models import ChatCompletionRequest
    from pydantic import ValidationError
    try:
        ChatCompletionRequest(**payload)
        return True
    except ValidationError:
        return False


VALID = [
    ('parallel_omitted_auto_default', build()),
    ('parallel_null', build(parallel_tool_calls=None)),
    ('parallel_true', build(parallel_tool_calls=True)),
    ('parallel_true_tool_choice_auto', build(parallel_tool_calls=True, tool_choice='auto')),
    ('parallel_true_without_tools', build(tools=ABSENT, parallel_tool_calls=True)),
    ('parallel_false_tools_absent', build(tools=ABSENT, parallel_tool_calls=False)),
    ('parallel_false_tools_null', build(tools=None, parallel_tool_calls=False)),
    ('parallel_false_tools_empty', build(tools=[], parallel_tool_calls=False)),
    ('parallel_false_tool_choice_none', build(parallel_tool_calls=False, tool_choice='none')),
    ('strict_false', build(tools=[tool('fixture', strict=False), tool('second_fixture', strict=False)])),
    ('strict_null', build(tools=[tool('fixture', strict=None)])),
    ('functions_null', build(functions=None)),
    ('sdk_style_auto_history_parallel_true_strict_false',
        build(tools=[tool('fixture', strict=False)], tool_choice='auto', parallel_tool_calls=True, history=True)),
]
INVALID = [
    ('parallel_false_tool_choice_omitted', build(parallel_tool_calls=False), PARALLEL_FALSE),
    ('parallel_false_tool_choice_null', build(parallel_tool_calls=False, tool_choice=None), PARALLEL_FALSE),
    ('parallel_false_tool_choice_auto', build(parallel_tool_calls=False, tool_choice='auto'), PARALLEL_FALSE),
    ('parallel_false_single_tool', build(tools=[tool('fixture')], parallel_tool_calls=False), PARALLEL_FALSE),
    ('parallel_false_auto_history', build(parallel_tool_calls=False, history=True), PARALLEL_FALSE),
    ('parallel_string_false', build(parallel_tool_calls='false'), PARALLEL_TYPE),
    ('parallel_string_true', build(parallel_tool_calls='true'), PARALLEL_TYPE),
    ('parallel_integer_zero', build(parallel_tool_calls=0), PARALLEL_TYPE),
    ('parallel_integer_one', build(parallel_tool_calls=1), PARALLEL_TYPE),
    ('parallel_list', build(parallel_tool_calls=[]), PARALLEL_TYPE),
    ('parallel_object', build(parallel_tool_calls={}), PARALLEL_TYPE),
    ('strict_true', build(tools=[tool('fixture', strict=True)]), STRICT),
    ('strict_true_second_tool_only', build(tools=[tool('fixture'), tool(PRIVATE, strict=True)]), STRICT),
    ('strict_true_tool_choice_none', build(tools=[tool('fixture', strict=True)], tool_choice='none'), STRICT),
    ('strict_true_parallel_true', build(tools=[tool('fixture', strict=True)], parallel_tool_calls=True), STRICT),
    ('strict_string_true', build(tools=[tool('fixture', strict='true')]), STRICT),
    ('strict_string_false', build(tools=[tool('fixture', strict='false')]), STRICT),
    ('strict_integer_one', build(tools=[tool('fixture', strict=1)]), STRICT),
    ('strict_integer_zero', build(tools=[tool('fixture', strict=0)]), STRICT),
    ('strict_tool_level', build(tools=[dict(tool('fixture'), strict=True)]), STRICT),
    ('legacy_functions', build(tools=None, functions=[{'name':PRIVATE,'parameters':{'type':'object'}}]), FUNCTIONS),
    ('legacy_functions_empty', build(tools=None, functions=[]), FUNCTIONS),
    ('tool_choice_error_precedes_parallel', build(tool_choice='required', parallel_tool_calls=False), TOOL_CHOICE),
]
GATE_SUITES = ('test-tool-choice-guard-cpu.py', 'test-tool-semantics-guard-cpu.py')


def gate_wiring(script):
    """Static gate contract: errexit, and each suite on one line with --guard and no ||."""
    if script.suffix == '.py':
        spec = importlib.util.spec_from_file_location('cpu_gate_wiring', script)
        runner = importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        wired = {name: '--guard' in runner.command_for_suite(script.with_name(name)) for name in GATE_SUITES}
        return {'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
                'suites_wired':wired,'pass':all(wired.values())}
    lines = [line.split('#', 1)[0] for line in script.read_text().splitlines()]
    errexit = any(re.match(r'\s*set\s+-\w*e', line) for line in lines)
    wired = {name: any(name in line and '--guard' in line and '||' not in line for line in lines)
             for name in GATE_SUITES}
    return {'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(), 'errexit':errexit,
            'suites_wired':wired, 'pass':errexit and all(wired.values())}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument('--guard', type=Path, required=True)
    p.add_argument('--report', type=Path)
    p.add_argument('--gate-script', type=Path)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('tool_semantics_guard_under_test', args.guard)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for stream in (False, True):
        for kind, payload in VALID:
            r = await exercise(module, payload, stream=stream)
            schema = schema_accepts(dict(payload, stream=stream))
            passed = (r['status'] == 200 and r['app_calls'] == 1 and r['metric_configure_calls'] == 1
                      and r['payload_unchanged'] and not r['slot_busy_after'] and schema)
            rows.append({'case':kind, 'expect':'accept', 'stream':stream, **r,
                         'freetoken_schema_accepts':schema, 'pass':passed})
        for kind, payload, error in INVALID:
            r = await exercise(module, payload, stream=stream)
            r['error_exact'] = r['error_message'] == error
            passed = (r['status'] == 400 and r['app_calls'] == 0 and r['metric_configure_calls'] == 0
                      and r['error_exact'] and r['first_event_is_start'] and r['json_body']
                      and r['content_type'] == 'application/json' and not r['sse_bytes']
                      and not r['private_marker_leaked'] and not r['slot_busy_after'] and not r['cleanup_failed'])
            rows.append({'case':kind, 'expect':'reject_400', 'stream':stream, **r, 'pass':passed})
    for stream in (False, True):
        r = await exercise(module, build(parallel_tool_calls=False), stream=stream, auth=False)
        rows.append({'case':'authentication_still_first', 'expect':'reject_401', 'stream':stream, **r,
                     'pass':r['status'] == 401 and r['app_calls'] == 0 and r['metric_configure_calls'] == 0})
    for r in rows:
        r.pop('error_message')  # only fixed server strings; exactness is kept as error_exact
    failed = [r['case']+(':stream' if r['stream'] else ':nonstream') for r in rows if not r['pass']]
    # The FreeToken package init imports torch; that alone must not initialize CUDA.
    torch = sys.modules.get('torch')
    cuda_initialized = bool(torch is not None and torch.cuda.is_initialized())
    if cuda_initialized:
        failed.append('cuda_initialized')
    gate = gate_wiring(args.gate_script) if args.gate_script else None
    if gate is not None and not gate['pass']:
        failed.append('gate_wiring')
    report = {'guard_sha256':hashlib.sha256(args.guard.read_bytes()).hexdigest(),
              'passed':not failed, 'case_count':len(rows), 'failed_count':len(failed), 'failed_cases':failed,
              'accept_cases':sum(r['expect'] == 'accept' for r in rows),
              'reject_400_cases':sum(r['expect'] == 'reject_400' for r in rows),
              'rows':rows, 'runtime_changed':False, 'model_tested':False,
              'torch_imported':'torch' in sys.modules, 'cuda_initialized':cuda_initialized,
              'scope':'ASGI guard: parallel_tool_calls/strict/functions rejection before app, metrics, '
                      'admission and SSE; accepted payloads also pass the FreeToken request schema'}
    summary = {'passed':report['passed'], 'case_count':len(rows), 'failed_cases':len(failed),
               'torch_imported':report['torch_imported'], 'cuda_initialized':cuda_initialized, 'model_tested':False}
    if gate is not None:
        report['gate_wiring'] = gate
        summary['gate_wiring_pass'] = gate['pass']
    if args.report:
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(summary))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
