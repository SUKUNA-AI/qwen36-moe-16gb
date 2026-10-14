"""Verify actual emitted-token budgets through both API response modes."""
import json,time
import requests
from bench import ROOT,headers,generate

rows=[]
messages=[{'role':'user','content':'Подробно объясни устройство двоичного поиска и приведи реализацию на Python.'}]
for streaming in (False,True):
    for limit in (1,2,3,127,128):
        if streaming:
            result,_=generate(messages,limit,seed=664,temperature=0,ignore_eos=True)
            status=result['status'];usage=result.get('usage',{});finish=result.get('finish_reason')
        else:
            response=requests.post('http://127.0.0.1:1919/v1/chat/completions',headers=headers(),json={
                'model':'qwen36-35b','messages':messages,'max_tokens':limit,'ignore_eos':True,
                'temperature':0,'seed':664,'chat_template_kwargs':{'enable_thinking':False}},timeout=60)
            status=response.status_code;result=response.json();usage=result.get('usage',{})
            finish=result.get('choices',[{}])[0].get('finish_reason')
        rows.append({'streaming':streaming,'requested':limit,'status':status,'usage':usage,
                     'finish_reason':finish,'pass':status==200 and usage.get('completion_tokens')==limit and finish=='length'})
        time.sleep(.15)
(ROOT/'reports/output-budget.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows));assert all(row['pass'] for row in rows)
