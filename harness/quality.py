"""Owner-local quality qualification; saves verdicts/losses/hashes, never responses."""
import argparse, ast, hashlib, json, re, time
import requests
from transformers import AutoTokenizer
from bench import ROOT, generate, headers, document, run_receipt
from fixtures import prepare

# Harness v7 teacher clean-cache step. At prefill commit the hybrid radix cache dedups a prompt's shared
# chat header and re-points its page_table rows to KV written by EARLIER requests (rounding-different GEMM
# shapes). The 128K teacher-bf128 reference computed long-120000 on an emptied tree: at context 131072 its
# admission evicts the whole ~87K-token tree. A larger KV pool (262144) keeps the tree and dedups. Each case
# listed here is therefore the first request of a freshly restarted server of the SAME profile, so every
# context compares like with like. Nothing else changes: same requests, same order, same engine.
CLEAN_CACHE_CASES=('long-120000',)
RESTART_MAY_CHANGE=('run_id','started_utc')


def teacher_rows(profile):
    """The engine's reports/teacher.jsonl rows of one profile, in file order (what compare-quality.py reads)."""
    path=ROOT/'reports/teacher.jsonl'
    rows=[]
    for line in (path.read_text().splitlines() if path.exists() else []):
        try:row=json.loads(line)
        except ValueError:continue
        if isinstance(row,dict) and row.get('profile')==profile:rows.append(row)
    return rows


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
    # Bind: compare-quality.py takes the LAST teacher.jsonl row per (profile, case); it must be this run's row,
    # and a clean-cache run must hold only its own case(s), i.e. they were its first requests.
    deadline=time.monotonic()+30
    while True:
        logged=teacher_rows(first['profile'])
        latest={row.get('case'):row.get('run_id') for row in logged}
        unbound={case:latest.get(case) for case,run_id in served.items() if latest.get(case)!=run_id}
        if not unbound:break
        if time.monotonic()>=deadline:raise RuntimeError(f'teacher.jsonl rows not bound to this run: {unbound}')
        time.sleep(.5)
    for receipt in clean:
        own=[row.get('case') for row in logged if row.get('run_id')==receipt['run_id']]
        if own!=[case for case,run_id in served.items() if run_id==receipt['run_id']]:
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
    _,text=generate([{'role':'user','content':'Ответь одним русским словом: столица России?'}],64,temperature=0)
    rows.append({'case':'russian','pass':'москва' in text.lower()})
    _,text=generate([{'role':'user','content':'Напиши только функцию Python rotate(a, k), циклически сдвигающую список вправо на k. Пустой список верни пустым. k может быть отрицательным. Используй срезы, без импортов, циклов, comprehensions, методов и комментариев. Только функцию.'}],256,temperature=0)
    rows.append({'case':'executable-code','pass':code_check(text),'sha256':hashlib.sha256(text.encode()).hexdigest()})
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
    rows.append({'case':'tool-cycle','pass':ok})
    for length in [16384,long_length]:
        body=document(tokenizer,length,9721+length);a=len(body)//3;b=2*len(body)//3
        content='Код задачи АЛЬФА-917.\n'+body[:a]+'\nКоличество деталей: 23.\n'+body[a:b]+'\nОтветственный: Ирина.\n'+body[b:]+'\nЦена детали 14 рублей. Ответь только JSON с полями code, count, owner, total для этой задачи.'
        result,text=generate([{'role':'user','content':content}],256,temperature=0)
        parsed=None
        try:
            parsed=json.loads(re.search(r'\{.*\}',text,re.S).group(0))
            ok=parsed=={'code':'АЛЬФА-917','count':23,'owner':'Ирина','total':322}
        except Exception:ok=False
        rows.append({'case':f'long-facts-{length}','pass':ok,'usage':result.get('usage'),'ttft_content_s':result.get('ttft_content_s'),
            'field_equal':{k:isinstance(parsed,dict) and parsed.get(k)==v for k,v in {'code':'АЛЬФА-917','count':23,'owner':'Ирина','total':322}.items()},
            'content_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'cached_prompt_tokens':result.get('cached_prompt_tokens'),'engine':result.get('engine'),
            'engine_metrics_status':result.get('engine_metrics_status')})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--teacher',action='store_true');p.add_argument('--prepare',action='store_true');p.add_argument('--long',type=int,default=120000)
    args=p.parse_args();tokenizer=AutoTokenizer.from_pretrained('/path/to/LLM/qwen36-api/model',local_files_only=True)
    cases=prepare(tokenizer)
    if args.prepare:print('Prepared 20 fixed holdout cases');return
    extra={}
    if args.teacher:
        rows,run,clean=teacher(tokenizer,cases)
        extra['clean_cache']={'harness':'v7','mode':'restart_same_profile','cases':list(CLEAN_CACHE_CASES),'runs':clean}
    else:
        rows=functional(tokenizer,args.long);run=run_receipt()
    report=ROOT/'reports'/f'{args.name}.json';report.write_text(json.dumps({'fixture_version':1,'run':run,'rows':rows,**extra},indent=2))
    print(json.dumps({'report':str(report),'rows':rows}),flush=True)

if __name__=='__main__':main()
