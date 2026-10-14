"""Caddy-host authenticated checks. Credential travels only through SSH stdin.

Live when invoked. Importing is safe for the isolated HTTP-response mock suite.
Reports distinguish a backend auth rejection from transport loss; neither TLS
nor a Caddy-LAN request is treated as external end-to-end proof.
"""
import argparse,datetime,json,os,pathlib,re,shlex,subprocess,uuid

ROOT=pathlib.Path('/home/USER/qwen36-api')
REMOTE=r'''
import hashlib,json,re,ssl,sys,time,urllib.request,urllib.error
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,newurl):return None
context=ssl.create_default_context()
# Ignore inherited proxy variables: the credential is for these direct endpoints.
# Never follow redirects with a Bearer header, even if a route is misconfigured.
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect(),urllib.request.HTTPSHandler(context=context))
def call(key,base,path,body=None,auth=True,method=None,timeout=15):
    headers={'Authorization':'Bearer '+key} if auth is True else ({'Authorization':'Bearer invalid-test'} if auth=='wrong' else {})
    if body is not None:headers['Content-Type']='application/json'
    request=urllib.request.Request(base+path,data=json.dumps(body).encode() if body is not None else None,headers=headers,method=method)
    try:return opener.open(request,timeout=timeout)
    except urllib.error.HTTPError as error:return error
def read_json(response):
    raw=response.read(1048577)
    if len(raw)>1048576:raise ValueError('response_limit')
    result=json.loads(raw)
    if not isinstance(result,dict):raise ValueError('response_schema')
    return result
def numeric_usage(value):
    return {k:v for k,v in (value or {}).items() if k in ('prompt_tokens','completion_tokens','total_tokens') and type(v) is int}
def collect(key):
    rows=[]
    def checked(base,case,fn,**meta):
        row={'base':base,'case':case,'pass':False,**meta}
        try:row.update(fn())
        except Exception as error:row.update(error_type=type(error).__name__,transport_or_schema_failure=True)
        rows.append(row)
        return row
    def status(base,path,expected,**kw):
        with call(key,base,path,**kw) as response:
            return {'status':response.status,'expected_status':expected,'pass':response.status==expected}
    def models(base):
        with call(key,base,'/v1/models') as response:
            data=read_json(response)
            return {'status':response.status,'pass':response.status==200 and any(x.get('id')=='qwen36-35b' for x in data.get('data',[]))}
    payload={'model':'qwen36-35b','messages':[{'role':'user','content':'Сколько будет 17 плюс 25? Ответь одним числом.'}],
             'max_tokens':64,'temperature':0,'seed':673,'chat_template_kwargs':{'enable_thinking':False}}
    fixture_tools=[{'type':'function','function':{'name':'fixture_add',
        'description':'Safely add two integers in this synthetic acceptance test. No external effects.',
        'parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},
                      'required':['a','b'],'additionalProperties':False}}}]
    def unsupported(base,delta,tools=True):
        # Pace rejected requests (ingress bucket 20, +10/s; guard 60, +1/s) so a
        # rate 429 can never stand in for, or hide, the expected 400.
        time.sleep(0.12)
        body={**payload,**({'tools':fixture_tools} if tools else {}),**delta}
        with call(key,base,'/v1/chat/completions',body) as response:
            content_type=response.headers.get('Content-Type','').split(';',1)[0].strip().lower()
            facts={'status':response.status,'expected_status':400,'pass':False,
                   'json_content_type':content_type=='application/json','sse_started':content_type=='text/event-stream'}
            if response.status==400 and content_type=='application/json':
                data=read_json(response)
                facts['json_error_response']=isinstance(data.get('error'),dict)
                facts['pass']=facts['json_error_response']
            return facts
    def instance(base):
        # Binds the report to one served model instance (identifier only; F-17).
        with call(key,base,'/health') as response:
            data=read_json(response);value=data.get('instance_id')
            valid=isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}',value) is not None
            return {'status':response.status,'instance_id':value if valid else None,
                    'pass':response.status==200 and data.get('model')=='qwen36-35b' and valid}
    def generate(base):
        started=time.monotonic()
        with call(key,base,'/v1/chat/completions',payload,timeout=120) as response:
            data=read_json(response);text=data.get('choices',[{}])[0].get('message',{}).get('content','')
            return {'status':response.status,'pass':response.status==200 and text.strip()=='42' and data.get('model')=='qwen36-35b',
                    'elapsed_s':time.monotonic()-started,'usage':numeric_usage(data.get('usage')),
                    'content_sha256':hashlib.sha256(text.encode()).hexdigest()}
    def stream(base):
        parts=[];usage={};first=None;done=False;models_seen=set();bytes_seen=0
        started=time.monotonic()
        with call(key,base,'/v1/chat/completions',{**payload,'stream':True,'stream_options':{'include_usage':True}},timeout=120) as response:
            content_type=response.headers.get('Content-Type','').split(';',1)[0].lower()
            if response.status!=200 or content_type!='text/event-stream':
                return {'status':response.status,'pass':False,'sse_content_type':content_type=='text/event-stream'}
            while True:
                line=response.readline(65537);bytes_seen+=len(line)
                if len(line)>65536 or bytes_seen>1048576 or time.monotonic()-started>180:raise ValueError('stream_limit')
                if not line:break
                if line.strip()==b'data: [DONE]':done=True;break
                if not line.startswith(b'data: '):continue
                event=json.loads(line[6:]);usage=numeric_usage(event.get('usage')) or usage
                # Every generation chunk must retain the served model alias.
                if event.get('choices'):models_seen.add(event.get('model'))
                for choice in event.get('choices',[]):
                    text=choice.get('delta',{}).get('content')
                    if text:
                        if first is None:first=time.monotonic()-started
                        parts.append(text)
            return {'status':response.status,'pass':done and ''.join(parts).strip()=='42' and models_seen=={'qwen36-35b'},
                    'ttft_content_s':first,'usage':usage,'done':done,'model_alias_valid':models_seen=={'qwen36-35b'}}
    def tool_cycle(base):
        # The only executable fixture is integer addition below. Model-provided
        # names/arguments never select code, a shell, a URL or another tool.
        messages=[{'role':'user','content':'Вызови функцию fixture_add с a=17 и b=25. После получения результата функции ответь только итоговым числом.'}]
        first_payload={**payload,'messages':messages,'tools':fixture_tools,'tool_choice':'auto','max_tokens':256}
        started=time.monotonic()
        with call(key,base,'/v1/chat/completions',first_payload,timeout=120) as response:
            first=read_json(response)
            facts={'pass':False,'first_status':response.status,'tool_result_sent':False,
                   'first_model_alias_valid':first.get('model')=='qwen36-35b'}
            if response.status!=200 or not facts['first_model_alias_valid']:return facts
        choices=first.get('choices',[])
        if not isinstance(choices,list) or len(choices)!=1:raise ValueError('tool_choice_count')
        choice=choices[0];message=choice.get('message',{})
        if not isinstance(message,dict) or message.get('role')!='assistant':raise ValueError('tool_message_schema')
        calls=message.get('tool_calls',[])
        facts['one_function_call']=isinstance(calls,list) and len(calls)==1
        facts['tool_finish_reason']=choice.get('finish_reason')=='tool_calls'
        if not facts['one_function_call'] or not facts['tool_finish_reason']:return facts
        tool=calls[0]
        if not isinstance(tool,dict):raise ValueError('tool_call_schema')
        function=tool.get('function',{})
        facts['tool_name_valid']=tool.get('type')=='function' and isinstance(function,dict) and function.get('name')=='fixture_add'
        call_id=tool.get('id')
        facts['call_id_valid']=isinstance(call_id,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,128}',call_id) is not None
        if not facts['tool_name_valid'] or not facts['call_id_valid']:return facts
        arguments=function.get('arguments')
        if not isinstance(arguments,str):raise ValueError('tool_arguments_must_be_json_string')
        args=json.loads(arguments)
        facts['arguments_valid']=(isinstance(args,dict) and set(args)=={'a','b'} and
            type(args['a']) is int and type(args['b']) is int and args['a']==17 and args['b']==25)
        if not facts['arguments_valid']:return facts
        result=args['a']+args['b']
        # Send the exact returned call id and validated JSON string back. Keep
        # arbitrary response fields and any reasoning out of the request/report.
        assistant={'role':'assistant','content':message.get('content'),
                   'tool_calls':[{'id':call_id,'type':'function','function':{'name':'fixture_add','arguments':arguments}}]}
        messages=messages+[assistant,{'role':'tool','tool_call_id':call_id,'content':json.dumps({'result':result})}]
        followup={**first_payload,'messages':messages,'tool_choice':'none','max_tokens':64}
        facts.update(tool_result_sent=True,tool_result_id_matched=messages[-1]['tool_call_id']==assistant['tool_calls'][0]['id'])
        with call(key,base,'/v1/chat/completions',followup,timeout=120) as response:
            final=read_json(response)
            final_choices=final.get('choices',[])
            if not isinstance(final_choices,list) or len(final_choices)!=1:raise ValueError('tool_final_choice_count')
            final_message=final_choices[0].get('message',{})
            text=final_message.get('content','')
            facts.update(final_status=response.status,final_model_alias_valid=final.get('model')=='qwen36-35b',
                final_answer_correct=isinstance(text,str) and text.strip()=='42',
                no_further_tool_calls=not final_message.get('tool_calls'),
                elapsed_s=time.monotonic()-started,first_usage=numeric_usage(first.get('usage')),
                final_usage=numeric_usage(final.get('usage')))
            facts['pass']=(response.status==200 and facts['final_model_alias_valid'] and
                facts['final_answer_correct'] and facts['no_further_tool_calls'] and facts['tool_result_id_matched'])
            return facts
    def strict_tools(level):
        function=dict(fixture_tools[0]['function'])
        if level=='function':return [{'type':'function','function':{**function,'strict':True}}]
        return [{'type':'function','strict':True,'function':function}]
    # Guard v2 refusals: each body differs from an accepted auto-tool request
    # only by the unsupported field (legacy functions are sent without tools).
    guard_v2=(('parallel-false',{'parallel_tool_calls':False},True),
              ('parallel-nonbool',{'parallel_tool_calls':'false'},True),
              ('strict-function',{'tools':strict_tools('function')},True),
              ('strict-tool',{'tools':strict_tools('tool')},True),
              ('legacy-functions',{'functions':[dict(fixture_tools[0]['function'])]},False))
    before=checked('http://127.0.0.1:1919','instance-before',lambda:instance('http://127.0.0.1:1919'))
    for base in ('http://127.0.0.1:1919','https://llm.example.com'):
        # An unreachable tunnel is a transport failure, never an auth PASS.
        reachable=checked(base,'models',lambda:models(base))
        if reachable.get('transport_or_schema_failure'):
            rows.append({'base':base,'case':'dependent_checks','pass':False,'status':'NOT_RUN_UNREACHABLE_OR_INVALID'})
            continue
        for auth in (False,'wrong'):
            checked(base,'auth-'+str(auth),lambda auth=auth:status(base,'/v1/models',401,auth=auth))
            checked(base,'auth-chat-'+str(auth),lambda auth=auth:status(base,'/v1/chat/completions',401,body=payload,auth=auth))
        # Incorrect/missing credentials must reach pinned ingress and return 401.
        # 403, 429 or a proxy 502 are failures, not equivalent authentication proof.
        checked(base,'nonstream',lambda:generate(base))
        checked(base,'stream',lambda:stream(base))
        for streaming in (False,True):
            suffix='stream' if streaming else 'nonstream'
            for kind,value in (('required','required'),('named',{'type':'function','function':{'name':'fixture_add'}})):
                delta={'tool_choice':value,'stream':streaming}
                checked(base,'unsupported-forced-'+kind+'-'+suffix,lambda delta=delta:unsupported(base,delta))
            checked(base,'unsupported-json-mode-'+suffix,lambda streaming=streaming:
                unsupported(base,{'response_format':{'type':'json_object'},'stream':streaming}))
            for kind,delta,tools in guard_v2:
                checked(base,'unsupported-'+kind+'-'+suffix,lambda delta=delta,tools=tools,streaming=streaming:
                    unsupported(base,{**delta,'stream':streaming},tools))
    base='https://llm.example.com'
    checked(base,'auto-tool-cycle',lambda:tool_cycle(base))
    for method,path in [('GET','/health'),('GET','/v1/stats'),('GET','/v1/cache/status'),('GET','/metrics'),('GET','/docs'),('POST','/v1/responses'),('POST','/v1/completions'),('HEAD','/v1/models'),('OPTIONS','/v1/chat/completions')]:
        checked(base,'closed-route',lambda method=method,path=path:status(base,path,404,method=method),method=method,path=path)
    checked('https://forge.example.com','existing-forge-site',lambda:status('https://forge.example.com','/',200,auth=False))
    after=checked('http://127.0.0.1:1919','instance-after',lambda:instance('http://127.0.0.1:1919'))
    same_instance=bool(before.get('pass') and after.get('pass') and before.get('instance_id')==after.get('instance_id'))
    rows.append({'base':'http://127.0.0.1:1919','case':'same-instance','pass':same_instance})
    tls_verified=any(row.get('status')==200 and row['base']=='https://llm.example.com' and row['case']=='models' for row in rows)
    return {'source':'Caddy edge on home LAN; NOT an external end-to-end test','external_e2e':'NOT_RUN',
            'instance_id':before.get('instance_id') if same_instance else None,'same_instance':same_instance,
            'tls_verified':tls_verified,'redirects_followed':False,'rows':rows,'pass':all(row['pass'] for row in rows)}
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='network-release-'+uuid.uuid4().hex[:8]);a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,80}',a.name):raise SystemExit('Invalid report name')
    target=ROOT/'optimization/reports'/(a.name+'.json')
    if target.exists():raise SystemExit('Report already exists; select a new name')
    if subprocess.run(['systemctl','--user','is-active','--quiet','qwen36-inference'],timeout=10).returncode:
        raise SystemExit('Production must be explicitly running before this acceptance test')
    from sys import path
    path.insert(0,str(ROOT))
    from readiness import load_key
    try:
        result=subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
            '-o','ConnectTimeout=5','EDGE_SSH_HOST','python3 -c '+shlex.quote(REMOTE+'\nprint(json.dumps(collect(sys.stdin.read().strip())))')],
            input=load_key(),capture_output=True,text=True,timeout=600)
        if result.returncode:raise RuntimeError('Caddy probe process failed')
        report=json.loads(result.stdout)
        if not isinstance(report,dict) or type(report.get('pass')) is not bool:raise ValueError('Invalid report schema')
    except Exception as error:
        report={'pass':False,'error_type':type(error).__name__,'external_e2e':'NOT_RUN','tls_verified':False,
                'scope':'SSH_OR_REMOTE_CHECK_FAILURE','rows':[]}
    report['recorded_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    out=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(out,'w') as handle:json.dump(report,handle,indent=2);handle.write('\n')
    print(json.dumps(report))
    if not report['pass']:raise SystemExit(1)

if __name__=='__main__':main()
