"""Owner-local quality qualification; saves verdicts/losses/hashes, never responses."""
import argparse, ast, os, hashlib, json, re, time
import requests
from bench import ROOT, generate, headers, document, run_receipt
from fixtures import prepare
from freetoken.quality_evidence import teacher_records, file_digest, digest

# Harness v7 teacher clean-cache step. At prefill commit the hybrid radix cache dedups a prompt's shared
# chat header and re-points its page_table rows to KV written by EARLIER requests (rounding-different GEMM
# shapes). The 128K teacher-bf128 reference computed long-120000 on an emptied tree: at context 131072 its
# admission evicts the whole ~87K-token tree. A larger KV pool (262144) keeps the tree and dedups. Each case
# listed here is therefore the first request of a freshly restarted server of the SAME profile, so every
# context compares like with like. Nothing else changes: same requests, same order, same engine.
CLEAN_CACHE_CASES=('long-120000',)
RESTART_MAY_CHANGE=('run_id','started_utc')


def teacher_rows(profile):
    """The engine's reports/teacher.jsonl rows of one profile across rotations, keyed by run/case."""
    return [row for key,row in teacher_records(ROOT/'reports/teacher.jsonl').items() if key[0] == profile]


def restart_clean(previous):
    """Restart the experiment unit with the same profile (as qualify.py starts it): empty prefix tree."""
    from matrix import run,PY
    run(['systemctl','--user','stop','qwen36-experiment'],60)
    run(['bash',str(ROOT/'start-experiment.sh'),previous['profile']],60)
    run([PY,str(ROOT/'check-experiment.py'),'300'],310)
    receipt=run_receipt()
    same=lambda r:{k:v for k,v in r.items() if k not in RESTART_MAY_CHANGE}
    if receipt.get('run_id') in (None,previous['run_id']) or same(receipt)!=same(previous):
        raise RuntimeError('clean-cache restart did not give a fresh run with an identical receipt: '+previous['profile'])
    return receipt


def teacher(tokenizer,cases):
    """Teacher-forced pass over the fixed cases. Returns (rows, receipt of the first run, clean-cache receipts)."""
    first=run_receipt()
    if not first.get('run_id') or not first.get('profile'):raise RuntimeError('teacher run needs active-run.json')
    current=first;clean=[];rows=[];served={}
    for case in cases:
        if case['case'] in CLEAN_CACHE_CASES:
            current=restart_clean(current);clean.append(current)
        result,text=generate([{'role':'user','content':case['prompt']}],len(case['ids']),case['seed'],0,True)
        row={'case':case['case'],'status':result['status'],'expected_tokens':len(case['ids']),'usage':result.get('usage'),
             'forced_text_matches':text==tokenizer.decode(case['ids']),'run_id':current['run_id'],
             'clean_cache':case['case'] in CLEAN_CACHE_CASES}
        served[case['case']]=current['run_id']
        rows.append(row);print(json.dumps(row),flush=True)
    deadline=time.monotonic()+30
    while True:
        logged=teacher_rows(first['profile'])
        identities={(row['case'],row['run_id']) for row in logged}
        missing=[case for case,run in served.items() if (case,run) not in identities]
        if not missing:break
        if time.monotonic()>=deadline:raise RuntimeError(f'teacher rows missing for this run: {missing}')
        time.sleep(.5)
    for receipt in clean:
        own={row['case'] for row in logged if row['run_id']==receipt['run_id']}
        if own!={case for case,run in served.items() if run==receipt['run_id']}:
            raise RuntimeError(f"clean-cache run {receipt['run_id']} served other teacher cases: {own}")
    return rows,first,clean


def code_check(text):
    fenced=re.search(r'```(?:python)?\s*(.*?)```',text,re.S)
    code=fenced.group(1) if fenced else text
    try:
        tree=ast.parse(code)
        # This fixture needs only scalar/list arithmetic. No attributes, imports,
        # comprehensions, loops or arbitrary calls are admitted for execution.
        allowed=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.Assign,
                 ast.Name,ast.Load,ast.Store,ast.Constant,ast.If,ast.Compare,ast.Eq,
                 ast.Lt,ast.Gt,ast.LtE,ast.GtE,ast.Not,ast.UnaryOp,ast.USub,
                 ast.BinOp,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod,
                 ast.Subscript,ast.Slice,ast.Call,ast.BoolOp,ast.And,ast.Or,ast.IfExp,
                 ast.List,ast.Tuple)
        for node in ast.walk(tree):
            if not isinstance(node,allowed):return False
            if isinstance(node,ast.Call) and (not isinstance(node.func,ast.Name) or node.func.id not in ('len','list')):return False
            if isinstance(node,ast.Name) and node.id.startswith('__'):return False
        if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):return False
        namespace={'__builtins__':{'len':len,'list':list}}
        exec(compile(tree,'<restricted-fixture>','exec'),namespace)
        f=namespace['rotate']
        return all(f(a,k)==expected for a,k,expected in [([],3,[]),([1,2,3,4],1,[4,1,2,3]),([1,2,3],0,[1,2,3]),([1,2,3],5,[2,3,1]),([1,2,3],-1,[2,3,1])])
    except Exception:return False



def request(messages,**kwargs):
    payload={'model':'qwen36-35b','messages':messages,'max_tokens':512,'temperature':0,
             'seed':417,'chat_template_kwargs':{'enable_thinking':False},**kwargs}
    r=requests.post('http://127.0.0.1:1919/v1/chat/completions',headers=headers(),json=payload,timeout=900)
    r.raise_for_status();return r.json()


def functional(tokenizer,long_length):
    rows=[]
    result,text=generate([{'role':'user','content':'Ответь одним русским словом: столица России?'}],64,temperature=0)
    rows.append({'case':'russian','status':result['status'],'pass':'москва' in text.lower()})
    result,text=generate([{'role':'user','content':'Напиши только функцию Python rotate(a, k), циклически сдвигающую список вправо на k. Пустой список верни пустым. k может быть отрицательным. Используй срезы, без импортов, циклов, comprehensions, методов и комментариев. Только функцию.'}],256,temperature=0)
    rows.append({'case':'executable-code','status':result['status'],'pass':code_check(text),'sha256':hashlib.sha256(text.encode()).hexdigest()})
    tools=[{'type':'function','function':{'name':'get_fixture_weather','description':'Return synthetic weather for a city.','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city'],'additionalProperties':False}}}]
    messages=[{'role':'user','content':'Вызови get_fixture_weather для города Казань, затем сообщи температуру из результата. Не придумывай результат.'}]
    first=request(messages,tools=tools)
    msg=first['choices'][0]['message'];calls=msg.get('tool_calls',[])
    ok=len(calls)==1
    if ok:
        call=calls[0]
        try:args=json.loads(call['function']['arguments']);ok=bool(call['id']) and call['function']['name']=='get_fixture_weather' and args.get('city')=='Казань'
        except (ValueError,KeyError):ok=False
    if ok:
        second=request(messages+[msg,{'role':'tool','tool_call_id':call['id'],'content':'{"temperature_c":17,"synthetic":true}'}],tools=tools)
        ok='17' in (second['choices'][0]['message'].get('content') or '')
    rows.append({'case':'tool-cycle','status':200,'pass':ok})
    for length in [16384,long_length]:
        body=document(tokenizer,length,9721+length);a=len(body)//3;b=2*len(body)//3
        content='Код задачи АЛЬФА-917.\n'+body[:a]+'\nКоличество деталей: 23.\n'+body[a:b]+'\nОтветственный: Ирина.\n'+body[b:]+'\nЦена детали 14 рублей. Ответь только JSON с полями code, count, owner, total для этой задачи.'
        result,text=generate([{'role':'user','content':content}],256,temperature=0)
        parsed=None
        try:
            parsed=json.loads(re.search(r'\{.*\}',text,re.S).group(0))
            ok=parsed=={'code':'АЛЬФА-917','count':23,'owner':'Ирина','total':322}
        except Exception:ok=False
        rows.append({'case':f'long-facts-{length}','status':result['status'],'pass':ok,'usage':result.get('usage'),'ttft_content_s':result.get('ttft_content_s'),
            'field_equal':{k:isinstance(parsed,dict) and parsed.get(k)==v for k,v in {'code':'АЛЬФА-917','count':23,'owner':'Ирина','total':322}.items()},
            'content_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'cached_prompt_tokens':result.get('cached_prompt_tokens'),'engine':result.get('engine'),
            'engine_metrics_status':result.get('engine_metrics_status')})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--teacher',action='store_true');p.add_argument('--prepare',action='store_true');p.add_argument('--long',type=int,default=120000)
    from transformers import AutoTokenizer
    args=p.parse_args();tokenizer=AutoTokenizer.from_pretrained(os.environ.get('QWEN36_MODEL_DIR','/path/to/LLM/qwen36-api/model'),local_files_only=True)
    cases=prepare(tokenizer,create=args.prepare) if args.teacher or args.prepare else None
    if args.prepare:print('Prepared 20 fixed holdout cases');return
    extra={}
    if args.teacher:
        rows,run,clean=teacher(tokenizer,cases)
        extra['teacher_evidence']=[r for r in teacher_rows(run['profile'])
            if any(r['case']==row['case'] and r['run_id']==row['run_id'] for row in rows)]
        extra['fixture_sha256']=file_digest(ROOT/'config/teacher-fixture.json')
        extra['targets']={c['case']:{'count':len(c['ids']),'sha256':digest(c['ids'])} for c in cases}
        extra['clean_cache']={'harness':'v7','mode':'restart_same_profile','cases':list(CLEAN_CACHE_CASES),'runs':clean}
    else:
        rows=functional(tokenizer,args.long);run=run_receipt()
        extra['functional_matrix']={'version':1,'long':args.long}
    report=ROOT/'reports'/f'{args.name}.json'
    with report.open('x') as out:out.write(json.dumps({'fixture_version':1,'run':run,'rows':rows,**extra},indent=2,allow_nan=False))
    print(json.dumps({'report':str(report),'rows':rows}),flush=True)

if __name__=='__main__':main()
