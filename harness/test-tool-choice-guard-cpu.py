"""Exercise the selected guard as ASGI, without a model, API secret or CUDA.

--guard may point to the isolated candidate or integrated guard. Report contains
case categories/counters only, never request bodies, headers or tool arguments.
"""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ERROR = 'tool_choice supports only auto or none; forced calls are unavailable'
OMIT = object()
PRIVATE = 'SYNTHETIC_PRIVATE_SENTINEL_NOT_TO_BE_LOGGED'


async def exercise(module, choice, *, stream=False, auth=True, history=False):
    payload = {'model':'qwen36-35b', 'messages':[{'role':'user','content':PRIVATE}],
               'tools':[{'type':'function','function':{'name':'fixture','parameters':{'type':'object'}}}],
               'stream':stream, 'max_tokens':8}
    if choice is not OMIT:
        payload['tool_choice'] = choice
    if history:
        payload['messages'].extend([
            {'role':'assistant','tool_calls':[{'id':'fixture-id','type':'function',
                'function':{'name':'fixture','arguments':'{"value":1}'}}]},
            {'role':'tool','tool_call_id':'fixture-id','content':'{"value":2}'},
        ])
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
            return {'type':'http.request','body':json.dumps(payload).encode(), 'more_body':False}
        await asyncio.Future()
    async def send(event):
        events.append(event)
    headers = [(b'authorization', b'Bearer '+b'x'*43)] if auth else []
    scope = {'type':'http','method':'POST','path':'/v1/chat/completions','headers':headers,
             'qwen36.metrics':SimpleNamespace(configure=configured)}
    await guard(scope, receive, send)
    starts = [e for e in events if e['type'] == 'http.response.start']
    assert len(starts) == 1
    body = b''.join(e.get('body',b'') for e in events if e['type']=='http.response.body')
    result = {'status':starts[0]['status'], **counters, 'slot_busy_after':guard.busy,
              'private_marker_leaked':PRIVATE.encode() in body,
              'error_exact':json.loads(body).get('error',{}).get('message') == ERROR}
    assert not guard.busy and not guard.cleanup_failed
    return result


async def main():
    p = argparse.ArgumentParser()
    p.add_argument('--guard',type=Path,required=True)
    p.add_argument('--report',type=Path)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('tool_choice_guard_under_test',args.guard)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    valid = [('omitted',OMIT),('null',None),('auto','auto'),('none','none')]
    invalid = [('required','required'),('named',{'type':'function','function':{'name':'fixture'}}),
               ('unknown_named',{'type':'function','function':{'name':PRIVATE}}),
               ('empty_object',{}),('malformed_function',{'type':'function','function':None}),
               ('string_invalid','invalid'),('string_case','AUTO'),('string_empty',''),
               ('string_whitespace',' auto '),('integer',0),('boolean_false',False),
               ('boolean_true',True),('float',0.0),('list_empty',[]),('list_value',['auto'])]
    for stream in (False,True):
        for kind,choice in valid:
            result = await exercise(module,choice,stream=stream)
            passed = result['status']==200 and result['app_calls']==1 and result['metric_configure_calls']==1 and result['payload_unchanged']
            rows.append({'case':kind,'stream':stream,**result,'pass':passed})
        for kind,choice in invalid:
            result = await exercise(module,choice,stream=stream)
            passed = result['status']==400 and result['app_calls']==0 and result['metric_configure_calls']==0 and result['error_exact'] and not result['private_marker_leaked']
            rows.append({'case':kind,'stream':stream,**result,'pass':passed})
        result = await exercise(module,'auto',stream=stream,history=True)
        rows.append({'case':'auto_tool_result_history','stream':stream,**result,
                     'pass':result['status']==200 and result['app_calls']==1 and result['payload_unchanged']})
    result = await exercise(module,'required',auth=False)
    rows.append({'case':'authentication_still_first',**result,
                 'pass':result['status']==401 and result['app_calls']==0 and result['metric_configure_calls']==0})
    report = {'guard_sha256':hashlib.sha256(args.guard.read_bytes()).hexdigest(),
              'passed':all(r['pass'] for r in rows),'case_count':len(rows),'rows':rows,
              'runtime_changed':False,'model_tested':False,'cuda_imported':False,
              'scope':'ASGI guard rejection before application/metrics/admission; auto history preservation'}
    if args.report:
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'case_count':len(rows),
                      'failed_cases':sum(not r['pass'] for r in rows),'model_tested':False}))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    sys.exit(asyncio.run(main()))
