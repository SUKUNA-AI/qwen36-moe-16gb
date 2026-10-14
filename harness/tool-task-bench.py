"""Natural-EOS auto tool cycle (fixture V5) on an already-loaded profile; numeric/hash reports only.

Modes
  matrix:  --name STEM --repeat R      one cycle per matrix arm -> reports/STEM-tool-task.json
  pilot:   --pilot --name NAME --repeats 3 --expect-profile bf128
           N cycles on the loaded profile, no matrix -> reports/NAME-tool-pilot.json

Never starts/stops a service and never reads a raw HTTP error body. Every outcome,
including failure before a tool call, produces a report row with static failure codes.
"""
import argparse,collections,copy,hashlib,json,os,pathlib,re,sys,time
import requests
import bench
from bench import ROOT,headers,attach_metrics,canonical_hash,run_receipt,validate_receipt
import tool_fixture_v5 as fixture

SCHEMA_VERSION=2
NAME_PATTERN=re.compile(r'[a-z0-9][a-z0-9.-]{0,90}')
TIMING_SCOPE='sum of two HTTP request wall times; local integer addition and post-task metrics joins excluded'


def file_sha256(name):
    return hashlib.sha256((pathlib.Path(__file__).parent/name).read_bytes()).hexdigest()


def sources():
    return {name:file_sha256(name) for name in ('tool-task-bench.py','tool_fixture_v5.py','bench.py')}


def numeric_usage(value):
    return {key:item for key,item in value.items() if type(item) is int} if isinstance(value,dict) else {}


def normalized_history(messages):
    # Tool call IDs are transport identities, not task content. Keep all other
    # actual model output (including argument-string whitespace) in this hash.
    messages=copy.deepcopy(messages)
    for message in messages:
        for call in message.get('tool_calls') or []:
            if isinstance(call,dict):call['id']='fixture_call'
        if message.get('role')=='tool':message['tool_call_id']='fixture_call'
    return messages


def request(messages,maximum):
    """One non-stream request. Returns (step, message|None); step['error_code'] marks a failed step."""
    sampling={**fixture.SAMPLING,'max_tokens':maximum,'thinking':False,'ignore_eos':False}
    payload={'model':'qwen36-35b','messages':messages,'tools':fixture.TOOLS,'tool_choice':fixture.TOOL_CHOICE,
             'seed':fixture.SAMPLING['seed'],'temperature':fixture.SAMPLING['temperature'],'top_p':fixture.SAMPLING['top_p'],
             'max_tokens':maximum,'stream':False,'chat_template_kwargs':{'enable_thinking':False}}
    step={'sampling':sampling,'request_fixture_sha256':canonical_hash({'messages':normalized_history(messages),
          'tools':fixture.TOOLS,'tool_choice':fixture.TOOL_CHOICE})}
    started=time.perf_counter()
    try:response=requests.post('http://127.0.0.1:1919/v1/chat/completions',headers=headers(),json=payload,timeout=(5,120))
    except requests.RequestException:
        step.update(status=None,error_code='transport_error',elapsed_s=time.perf_counter()-started)
        return step,None
    step.update(status=response.status_code,elapsed_s=time.perf_counter()-started,
                request_id=response.headers.get('x-qwen36-request-id'))
    if response.status_code!=200:
        step['error_code']='http_status'  # the body is deliberately not read or stored
        return step,None
    try:data=response.json()
    except ValueError:data=None
    choices=data.get('choices') if isinstance(data,dict) else None
    if (not isinstance(choices,list) or len(choices)!=1 or not isinstance(choices[0],dict)
            or not isinstance(choices[0].get('message'),dict)):
        step['error_code']='bad_response_schema'
        return step,None
    message=choices[0]['message']
    reasoning=message.get('reasoning_content')
    step.update(usage=numeric_usage(data.get('usage')),finish_reason=choices[0].get('finish_reason'),
                reasoning_chars=len(reasoning) if isinstance(reasoning,str) else 0,
                message_sha256=canonical_hash(message),
                message_normalized_sha256=canonical_hash(normalized_history([message])[0]))
    return step,message


def execute(repeat,mode='matrix'):
    """Run one cycle. Always returns a row; never raises for model/HTTP behaviour."""
    initial=fixture.first_messages()
    row={'case':'tool_cycle','repeat':repeat,'mode':mode,'fixture_id':fixture.FIXTURE_ID,
         'fixture_version':fixture.FIXTURE_VERSION,'fixture_sha256':fixture.fixture_sha256(),
         'task_fixture_sha256':canonical_hash({'messages':initial,'tools':fixture.TOOLS}),
         'correct':False,'failure_code':None,'failure_codes':[],'steps':[],
         'returned_call_id_sha256':None,'tool_result_matches_call_id':None,'safe_tool_result':None,
         'second_history_normalized_sha256':None,'task_elapsed_s':None,'timing_scope':TIMING_SCOPE}
    codes=[]
    one,message=request(initial,fixture.STEP_LIMITS[0]);row['steps'].append(one)
    if one.get('error_code'):codes.append(one['error_code'])
    else:
        first_codes,call,value=fixture.validate_first(message,one.get('finish_reason'),one.get('reasoning_chars'))
        codes+=first_codes
        if call is not None:
            row['returned_call_id_sha256']=hashlib.sha256(call['id'].encode()).hexdigest()
            row['safe_tool_result']=value
            # Preserve the returned assistant tool-call message exactly; send the
            # genuine local result with the returned ID.
            history=initial+[message,{'role':'tool','tool_call_id':call['id'],
                                      'content':json.dumps({'result':value},separators=(',',':'))}]
            row['tool_result_matches_call_id']=history[-1]['tool_call_id']==message['tool_calls'][0].get('id')==call['id']
            if not row['tool_result_matches_call_id']:codes.append('id_mismatch')
            row['second_history_normalized_sha256']=canonical_hash(normalized_history(history))
            two,answer=request(history,fixture.STEP_LIMITS[1]);row['steps'].append(two)
            if two.get('error_code'):codes.append('final_'+two['error_code'])
            else:codes+=fixture.validate_final(answer,two.get('finish_reason'),two.get('reasoning_chars'))
    if len(row['steps'])==2:row['task_elapsed_s']=sum(step['elapsed_s'] for step in row['steps'])
    # Metrics-file joins are deliberately after both model requests; they are not task work.
    for step in row['steps']:attach_metrics(step)
    codes=fixture.ordered(codes)
    row['failure_codes']=codes;row['failure_code']=codes[0] if codes else None
    row['correct']=not codes and len(row['steps'])==2 and row['safe_tool_result']==fixture.EXPECTED_RESULT
    return row


def write_new(path,document):
    """Receipts are never overwritten (O_EXCL)."""
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as target:target.write(json.dumps(document,indent=2,allow_nan=False)+'\n')


def receipt_or_code(expected_profile=None):
    receipt=run_receipt()
    try:
        validate_receipt(receipt,receipt.get('profile') if expected_profile is None else expected_profile)
        return receipt,None
    except (ValueError,KeyError,TypeError,AttributeError):return receipt,'receipt_invalid'


def matrix_task(root,name,repeat):
    path=root/'reports'/f'{name}-tool-task.json'
    if path.exists():raise FileExistsError('Tool-task report exists; receipts are never overwritten')
    receipt,code=receipt_or_code()
    report={'name':name,'mode':'matrix','run':receipt,'schema_version':SCHEMA_VERSION,'repeat':repeat,
            'fixture':fixture.describe(),'source_sha256':file_sha256('tool-task-bench.py'),'sources':sources()}
    if code:
        report.update(status='failed',failure_code=code,failure_codes=[code],row=None)
    else:
        row=execute(repeat,'matrix')
        if run_receipt().get('run_id')!=receipt['run_id']:
            row['failure_codes']=fixture.ordered(row['failure_codes']+['serving_generation_changed'])
            row['failure_code']=row['failure_codes'][0];row['correct']=False
        report.update(row=row,status='complete' if row['correct'] else 'failed',
                      failure_code=row['failure_code'],failure_codes=row['failure_codes'])
    write_new(path,report)
    print(json.dumps({'status':report['status'],'failure_codes':report['failure_codes'],'report':path.name}))
    return 0 if report['status']=='complete' else 1


def pilot(root,name,repeats,expect_profile):
    path=root/'reports'/f'{name}-tool-pilot.json'
    if path.exists():raise FileExistsError('Pilot report exists; use a new --name')
    receipt,code=receipt_or_code(expect_profile)
    report={'name':name,'mode':'pilot','schema_version':SCHEMA_VERSION,'profile':expect_profile,'repeats':repeats,
            'run':receipt,'fixture':fixture.describe(),'source_sha256':file_sha256('tool-task-bench.py'),'sources':sources(),
            'retired_fixtures':{key:{'status':value['status'],'inherits_pass':value['inherits_pass']}
                                for key,value in fixture.RETIRED_FIXTURES.items()},
            'scope':'Repeat 0 matches the matrix condition (no prior identical prompt in the process); later repeats '
                    'may observe prefix-cache hits on the identical input, recorded per step. All repeats must pass.'}
    rows=[]
    if code:report.update(status='FAIL',failure_codes=[code],rows=rows,passes=0)
    else:
        for repeat in range(repeats):
            rows.append(execute(repeat,'pilot'));print(json.dumps({'repeat':repeat,'correct':rows[-1]['correct'],
                                                            'failure_codes':rows[-1]['failure_codes']}),flush=True)
        generation_ok=run_receipt().get('run_id')==receipt['run_id']
        counts=collections.Counter(code for row in rows for code in row['failure_codes'])
        if not generation_ok:counts['serving_generation_changed']+=1
        first=[row['steps'][0].get('message_normalized_sha256') for row in rows]
        report.update(rows=rows,passes=sum(row['correct'] for row in rows),failure_code_counts=dict(counts),
                      failure_codes=sorted(counts),
                      first_step_identical_across_repeats=len(set(first))==1 and None not in first,
                      cached_prompt_tokens_by_repeat=[[step.get('cached_prompt_tokens') for step in row['steps']] for row in rows],
                      status='PASS' if generation_ok and len(rows)==repeats and all(row['correct'] for row in rows) else 'FAIL')
    write_new(path,report)
    print(json.dumps({'status':report['status'],'passes':report['passes'],'repeats':repeats,
                      'failure_codes':report['failure_codes'],'report':path.name}))
    return 0 if report['status']=='PASS' else 1


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True)
    parser.add_argument('--repeat',type=int);parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--expect-profile')
    args=parser.parse_args(argv)
    if not NAME_PATTERN.fullmatch(args.name):parser.error('invalid --name')
    if args.pilot:
        if args.repeat is not None or not args.expect_profile or not 1<=args.repeats<=10:
            parser.error('--pilot needs --expect-profile and 1..10 --repeats, and no --repeat')
        return pilot(bench.ROOT,args.name,args.repeats,args.expect_profile)
    if args.repeat is None or args.repeat<0:parser.error('matrix mode needs --repeat >= 0')
    return matrix_task(bench.ROOT,args.name,args.repeat)


if __name__=='__main__':raise SystemExit(main())
