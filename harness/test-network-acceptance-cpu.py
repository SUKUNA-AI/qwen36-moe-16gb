"""Synthetic HTTP response tests. No network, SSH, service or credentials."""
import importlib.util
import io
import json
import pathlib
import unittest
import urllib.error
from unittest.mock import patch

path=pathlib.Path(__file__).with_name('network-acceptance.py')
spec=importlib.util.spec_from_file_location('network_fixture_module',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def guard_rejects(body):
    # Mirrors guard.py validate_payload v2 refusals for the synthetic server.
    choice,tools,parallel=body.get('tool_choice'),body.get('tools'),body.get('parallel_tool_calls')
    if choice is not None and choice not in ('auto','none'):return True
    if body.get('response_format')=={'type':'json_object'}:return True
    if parallel is not None and type(parallel) is not bool:return True
    if parallel is False and tools and choice!='none':return True
    for tool in tools or ():
        for owner in (tool,tool.get('function')):
            if isinstance(owner,dict) and owner.get('strict') not in (None,False):return True
    return body.get('functions') is not None


def guard_v2_kind(body):
    tools=body.get('tools') or []
    if 'functions' in body:return 'legacy-functions'
    if 'parallel_tool_calls' in body:return 'parallel-false' if body['parallel_tool_calls'] is False else 'parallel-nonbool'
    if tools and 'strict' in tools[0]:return 'strict-tool'
    if tools and 'strict' in tools[0].get('function',{}):return 'strict-function'
    return None


GUARD_V2=('parallel-false','parallel-nonbool','strict-function','strict-tool','legacy-functions')


class Reply(io.BytesIO):
    def __init__(self,status,data=b'',content_type='application/json'):
        super().__init__(data);self.status=status;self.headers={'Content-Type':content_type}


class NetworkFixtures(unittest.TestCase):
    def setUp(self):
        self.remote={};exec(compile(m.REMOTE,'synthetic-remote','exec'),self.remote)
        self.calls=[];self.failure=None;self.payloads=[];self.instances=['generation-a','generation-a']
        self.remote['call']=self.call
        sleeper=patch.object(self.remote['time'],'sleep');self.sleep=sleeper.start();self.addCleanup(sleeper.stop)

    def call(self,key,base,path,body=None,auth=True,method=None,timeout=15):
        self.calls.append((base,path,auth,method))
        if body is not None:self.payloads.append((base,body))
        if self.failure:
            special=self.failure(base,path,body,auth,method)
            if special is not None:return special
        if base=='https://forge.example.com':
            self.assertIs(auth,False)
            return Reply(200,b'synthetic site')
        if path=='/health' and base=='http://127.0.0.1:1919' and auth is True and method is None:
            # Caddy keeps /health closed; only the edge loopback tunnel serves it.
            return Reply(200,json.dumps({'status':'ok','model':'qwen36-35b','instance_id':self.pending.pop(0)}).encode())
        if path not in ('/v1/models','/v1/chat/completions') or method in ('HEAD','OPTIONS'):
            return Reply(404)
        if auth is not True:return Reply(401)
        if path=='/v1/models':return Reply(200,json.dumps({'data':[{'id':'qwen36-35b'}]}).encode())
        if guard_rejects(body):
            return Reply(400,b'{"error":{"type":"invalid_request_error","message":"PRIVATE_GUARD_MESSAGE"}}')
        if body.get('tool_choice')=='auto':return self.tool_first()
        if body.get('tool_choice')=='none':
            # The followup must depend on the preceding model-produced call id,
            # exact validated argument string and tool result, not another prompt.
            self.assertEqual(base,'https://llm.example.com')
            self.assertEqual(body['messages'][-2]['tool_calls'][0]['id'],'call_fixture_17_25')
            self.assertEqual(body['messages'][-1]['tool_call_id'],'call_fixture_17_25')
            self.assertEqual(json.loads(body['messages'][-1]['content']),{'result':42})
            self.assertEqual(json.loads(body['messages'][-2]['tool_calls'][0]['function']['arguments']),{'a':17,'b':25})
        if body.get('stream'):
            return Reply(200,b'data: {"model":"qwen36-35b","choices":[{"delta":{"content":"42"}}]}\n\ndata: [DONE]\n\n','text/event-stream; charset=utf-8')
        return Reply(200,json.dumps({'model':'qwen36-35b','choices':[{'message':{'content':'42'}}],
          'usage':{'prompt_tokens':17,'completion_tokens':2,'private_extra':'must not escape'}}).encode())

    def tool_first(self,**changes):
        tool={'id':'call_fixture_17_25','type':'function','function':{'name':'fixture_add','arguments':'{"a":17,"b":25}'}}
        tool.update(changes)
        return Reply(200,json.dumps({'model':'qwen36-35b','choices':[{'finish_reason':'tool_calls',
            'message':{'role':'assistant','content':None,'tool_calls':[tool]}}]}).encode())

    def collect(self):
        self.pending=list(self.instances)
        return self.remote['collect']('SYNTHETIC-NOT-A-CREDENTIAL')

    def test_success_exact_alias_and_closed_routes(self):
        report=self.collect()
        self.assertTrue(report['pass']);self.assertTrue(report['tls_verified'])
        self.assertEqual(len(report['rows']),60)
        self.assertEqual(report['external_e2e'],'NOT_RUN')
        self.assertTrue(report['same_instance']);self.assertEqual(report['instance_id'],'generation-a')
        self.assertNotIn('private_extra',json.dumps(report))
        self.assertNotIn('PRIVATE_GUARD_MESSAGE',json.dumps(report))
        self.assertEqual(len([row for row in report['rows'] if row['case'].startswith('auth')]),8)
        unsupported=[row for row in report['rows'] if row['case'].startswith('unsupported-')]
        self.assertEqual(len(unsupported),32)
        self.assertTrue(all(row['status']==400 and row['json_error_response'] and not row['sse_started'] for row in unsupported))
        cycle=next(row for row in report['rows'] if row['case']=='auto-tool-cycle')
        self.assertTrue(cycle['pass'] and cycle['arguments_valid'] and cycle['call_id_valid'] and cycle['tool_result_id_matched'])
        tools=[(base,body) for base,body in self.payloads if body.get('tool_choice') in ('auto','none')]
        self.assertEqual(len(tools),2)
        self.assertTrue(all(base=='https://llm.example.com' for base,body in tools))
        self.assertNotIn('call_fixture_17_25',json.dumps(report))
        self.assertNotIn('"a": 17',json.dumps(report))

    def test_forced_choices_and_json_must_be_400_before_sse(self):
        for code,ctype in ((200,'text/event-stream'),(400,'text/event-stream'),(401,'application/json'),(429,'application/json')):
            self.failure=lambda base,path,body,auth,method:Reply(code,b'{"error":{}}',ctype) if body and (
                body.get('tool_choice')=='required' or isinstance(body.get('tool_choice'),dict) or body.get('response_format')) else None
            report=self.collect();self.assertFalse(report['pass'])
            rejected=[row for row in report['rows'] if row['case'].startswith(('unsupported-forced-','unsupported-json-mode-'))]
            self.assertEqual(len(rejected),12)
            self.assertTrue(all(not row['pass'] for row in rejected))

    def test_guard_v2_rows_cover_both_bases_and_stream_modes(self):
        report=self.collect()
        names={(row['base'],row['case']) for row in report['rows']}
        for base in ('http://127.0.0.1:1919','https://llm.example.com'):
            for kind in GUARD_V2:
                for suffix in ('stream','nonstream'):
                    self.assertIn((base,'unsupported-'+kind+'-'+suffix),names)
        # Paced: every rejected request waits before it is sent.
        self.assertGreaterEqual(len([c for c in self.sleep.call_args_list if c.args==(0.12,)]),32)

    def test_guard_v2_bodies_differ_only_by_the_unsupported_field(self):
        self.collect()
        sent={}
        for base,body in self.payloads:
            kind=guard_v2_kind(body)
            if kind:sent.setdefault(kind,[]).append((base,body))
        self.assertEqual(sorted(sent),sorted(GUARD_V2))
        for kind,items in sent.items():
            self.assertEqual(len(items),4)
            self.assertEqual({(base,body['stream']) for base,body in items},
                {(b,s) for b in ('http://127.0.0.1:1919','https://llm.example.com') for s in (False,True)})
            for base,body in items:
                self.assertNotIn('tool_choice',body);self.assertNotIn('response_format',body)
                self.assertEqual(body['model'],'qwen36-35b')
                if kind=='legacy-functions':
                    self.assertNotIn('tools',body);self.assertEqual(body['functions'][0]['name'],'fixture_add')
                else:
                    self.assertEqual(body['tools'][0]['function']['name'],'fixture_add')
                if kind=='parallel-false':self.assertIs(body['parallel_tool_calls'],False)
                if kind=='parallel-nonbool':self.assertEqual(body['parallel_tool_calls'],'false')
                if kind=='strict-function':
                    self.assertIs(body['tools'][0]['function']['strict'],True);self.assertNotIn('strict',body['tools'][0])
                if kind=='strict-tool':
                    self.assertIs(body['tools'][0]['strict'],True);self.assertNotIn('strict',body['tools'][0]['function'])
        # The shared fixture tool definition is never mutated by strict variants.
        self.assertTrue(all('strict' not in body['tools'][0]['function'] for _,body in self.payloads
                            if body.get('tool_choice')=='auto'))

    def test_guard_v2_cases_must_be_400_json_before_sse(self):
        for code,ctype,data in ((200,'application/json',b'{"choices":[]}'),(200,'text/event-stream',b'data: [DONE]\n'),
                                (400,'text/event-stream',b'{"error":{}}'),(400,'application/json',b'{"detail":"x"}'),
                                (429,'application/json',b'{"error":{}}')):
            self.failure=lambda base,path,body,auth,method:Reply(code,data,ctype) if body and guard_v2_kind(body) else None
            report=self.collect();self.assertFalse(report['pass'])
            rejected=[row for row in report['rows'] if any(row['case'].startswith('unsupported-'+k+'-') for k in GUARD_V2)]
            self.assertEqual(len(rejected),20)
            self.assertTrue(all(not row['pass'] for row in rejected))
            others=[row for row in report['rows'] if row not in rejected]
            self.assertTrue(all(row['pass'] for row in others))

    def test_report_carries_no_key_prompt_or_tool_text(self):
        text=json.dumps(self.collect(),ensure_ascii=False)
        for marker in ('SYNTHETIC-NOT-A-CREDENTIAL','Bearer','\u0421\u043a\u043e\u043b\u044c\u043a\u043e','fixture_add','Safely add','"a": 17'):
            self.assertNotIn(marker,text)

    def test_instance_change_or_invalid_identity_fails(self):
        for instances in (['generation-a','generation-b'],['bad id with spaces','bad id with spaces'],['x'*129,'x'*129]):
            self.instances=list(instances)
            report=self.collect()
            self.assertFalse(report['pass']);self.assertFalse(report['same_instance']);self.assertIsNone(report['instance_id'])
            self.assertNotIn('bad id',json.dumps(report));self.assertNotIn('x'*129,json.dumps(report))
            self.assertFalse(next(row for row in report['rows'] if row['case']=='same-instance')['pass'])

    def test_400_without_json_error_object_is_not_supported_rejection(self):
        self.failure=lambda base,path,body,auth,method:Reply(400,b'{"detail":"PRIVATE_DETAIL"}') if body and body.get('tool_choice')=='required' else None
        report=self.collect();self.assertFalse(report['pass'])
        self.assertNotIn('PRIVATE_DETAIL',json.dumps(report))

    def test_invalid_tool_id_never_sends_result(self):
        for call_id in (None,'','line\nbreak','x'*129):
            self.payloads=[]
            self.failure=lambda base,path,body,auth,method:self.tool_first(id=call_id) if body and body.get('tool_choice')=='auto' else None
            report=self.collect();cycle=next(row for row in report['rows'] if row['case']=='auto-tool-cycle')
            self.assertFalse(cycle['pass']);self.assertFalse(cycle['call_id_valid'])
            self.assertFalse(any(body.get('tool_choice')=='none' for _,body in self.payloads))

    def test_bad_tool_arguments_and_unknown_function_never_execute(self):
        functions=[{'name':'shell','arguments':'{"a":17,"b":25}'},
                   {'name':'fixture_add','arguments':'PRIVATE_NOT_JSON'},
                   {'name':'fixture_add','arguments':{'a':17,'b':25}},
                   {'name':'fixture_add','arguments':'{"a":true,"b":25}'},
                   {'name':'fixture_add','arguments':'{"a":17,"b":24}'},
                   {'name':'fixture_add','arguments':'{"a":17,"b":25,"command":"PRIVATE_EXEC_MARKER"}'}]
        for function in functions:
            self.payloads=[]
            self.failure=lambda base,path,body,auth,method:self.tool_first(function=function) if body and body.get('tool_choice')=='auto' else None
            report=self.collect();cycle=next(row for row in report['rows'] if row['case']=='auto-tool-cycle')
            self.assertFalse(cycle['pass'])
            self.assertFalse(any(body.get('tool_choice')=='none' for _,body in self.payloads))
            self.assertNotIn('PRIVATE_',json.dumps(report))

    def test_final_tool_result_wrong_answer_or_model_fails(self):
        for model,text in (('wrong-model','42'),('qwen36-35b','41')):
            self.failure=lambda base,path,body,auth,method:Reply(200,json.dumps({'model':model,'choices':[{'message':{'content':text}}]}).encode()) if body and body.get('tool_choice')=='none' else None
            report=self.collect();cycle=next(row for row in report['rows'] if row['case']=='auto-tool-cycle')
            self.assertFalse(cycle['pass']);self.assertTrue(cycle['tool_result_sent'])

    def test_no_tool_or_multiple_tools_or_wrong_finish_reason_fails(self):
        tool={'id':'call_fixture_17_25','type':'function','function':{'name':'fixture_add','arguments':'{"a":17,"b":25}'}}
        for calls,finish in (([],'stop'),([tool,tool],'tool_calls'),([tool],'stop')):
            self.payloads=[]
            response={'model':'qwen36-35b','choices':[{'finish_reason':finish,'message':{'role':'assistant','tool_calls':calls}}]}
            self.failure=lambda base,path,body,auth,method:Reply(200,json.dumps(response).encode()) if body and body.get('tool_choice')=='auto' else None
            report=self.collect();cycle=next(row for row in report['rows'] if row['case']=='auto-tool-cycle')
            self.assertFalse(cycle['pass']);self.assertFalse(cycle['tool_result_sent'])
            self.assertFalse(any(body.get('tool_choice')=='none' for _,body in self.payloads))

    def test_tunnel_loss_is_not_auth_rejection(self):
        def fail(base,*unused):
            if base.startswith('http:'):raise urllib.error.URLError('PRIVATE_TRANSPORT_DETAIL')
        self.failure=fail
        report=self.collect();self.assertFalse(report['pass'])
        self.assertNotIn('PRIVATE_TRANSPORT_DETAIL',json.dumps(report))
        local=[row for row in report['rows'] if row['base'].startswith('http:')]
        self.assertTrue(all(not row['pass'] for row in local))
        self.assertTrue(any(row.get('transport_or_schema_failure') for row in local))
        self.assertTrue(any(row['case']=='stream' and row['pass'] for row in report['rows']))

    def test_wrong_auth_403_429_and_proxy_502_are_failure(self):
        for code in (403,429,502):
            self.failure=lambda base,path,body,auth,method:Reply(code) if auth is not True else None
            report=self.collect()
            self.assertFalse(report['pass'])
            self.assertTrue(all(not row['pass'] for row in report['rows'] if row['case'].startswith('auth')))

    def test_stream_wrong_model_fails(self):
        self.failure=lambda base,path,body,auth,method:Reply(200,b'data: {"model":"other","choices":[{"delta":{"content":"42"}}]}\n\ndata: [DONE]\n','text/event-stream') if body and body.get('stream') else None
        report=self.collect();self.assertFalse(report['pass'])
        self.assertTrue(all(not row['model_alias_valid'] for row in report['rows'] if row['case']=='stream'))

    def test_stream_missing_done_fails(self):
        self.failure=lambda base,path,body,auth,method:Reply(200,b'data: {"model":"qwen36-35b","choices":[{"delta":{"content":"42"}}]}\n','text/event-stream') if body and body.get('stream') else None
        report=self.collect();self.assertFalse(report['pass'])

    def test_invalid_json_is_sanitized_failure(self):
        self.failure=lambda base,path,body,auth,method:Reply(200,b'PRIVATE_INVALID_RESPONSE') if path=='/v1/models' else None
        report=self.collect();self.assertFalse(report['pass']);self.assertFalse(report['tls_verified'])
        self.assertNotIn('PRIVATE_INVALID_RESPONSE',json.dumps(report))

    def test_redirect_is_not_followed(self):
        redirect=self.remote['NoRedirect']()
        self.assertIsNone(redirect.redirect_request(None,None,302,'',{},'https://other.invalid/'))

    def test_stream_oversized_line_is_bounded(self):
        self.failure=lambda base,path,body,auth,method:Reply(200,b'x'*65537,'text/event-stream') if body and body.get('stream') else None
        report=self.collect();self.assertFalse(report['pass'])
        self.assertTrue(all(row.get('error_type')=='ValueError' for row in report['rows'] if row['case']=='stream'))


if __name__=='__main__':unittest.main(verbosity=2)
