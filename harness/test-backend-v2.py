import argparse, asyncio, importlib, json, pathlib, sys, time
import requests
from bench import headers, generate, ROOT
from guard import validate_json_depth, validate_payload

url='http://127.0.0.1:1919'
parser=argparse.ArgumentParser();parser.add_argument('--context',type=int,default=65536)
parser.add_argument('--name',default='backend-v2');args=parser.parse_args()
base={'model':'qwen36-35b','messages':[{'role':'user','content':'Сколько будет 12 + 30? Ответь одним числом.'}],
      'max_tokens':32,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
checks={}
for choice in ('required',{'type':'function','function':{'name':'fixture_add'}}):
    for stream in (False,True):
        response=requests.post(url+'/v1/chat/completions',json=dict(base,tool_choice=choice,stream=stream),headers=headers(),timeout=10)
        name=('forced_required' if isinstance(choice,str) else 'forced_named')+('_stream' if stream else '_nonstream')
        checks[name]=response.status_code==400 and response.headers.get('content-type','').startswith('application/json')
        time.sleep(.12)
for name,delta in [('penalty',{'presence_penalty':1}),('seed_type',{'seed':True}),
                   ('conflict',{'max_completion_tokens':31}),('nan',{'top_p':-1}),
                   ('media',{'messages':[{'role':'user','content':[{'type':'image_url','image_url':'http://127.0.0.1/'}]}]})]:
    response=requests.post(url+'/v1/chat/completions',json=dict(base,**delta),headers=headers(),timeout=10)
    checks[name]=response.status_code==400
    time.sleep(.12)
deep=b'{"model":"qwen36-35b","a":'+b'['*70+b'0'+b']'*70+b'}'
checks['deep_json']=requests.post(url+'/v1/chat/completions',data=deep,headers=headers(),timeout=10).status_code==400
response=requests.post(url+'/v1/chat/completions',json=dict(base,messages=[{'role':'user','content':'word '*(args.context+500)}],max_tokens=1),headers=headers(),timeout=60)
checks['strict_context']=response.status_code==400 and response.json().get('error',{}).get('code')=='context_length_exceeded'
response=requests.post(url+'/v1/chat/completions',json=base,headers=headers(),timeout=30)
checks['nonstream']=response.status_code==200 and '42' in response.json()['choices'][0]['message']['content']
messages=[{'role':'user','content':'Перечисли пять простых чисел и объясни, что означает простое число.'}]
outputs=[]
for _ in range(3):
    row,output=generate(messages,48,777,.6)
    outputs.append(output)
checks['seed_repeated_output']=len(set(outputs))==1
report={'checks':checks,'all_pass':all(checks.values())}
(ROOT/'reports'/f'{args.name}.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report));sys.exit(0 if report['all_pass'] else 1)
