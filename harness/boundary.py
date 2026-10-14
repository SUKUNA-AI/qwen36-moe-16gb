"""Near-limit generation and strict prompt+output admission on the same text."""
import argparse,json
import requests
from transformers import AutoTokenizer
from bench import ROOT,document,generate,headers

def main():
    p=argparse.ArgumentParser();p.add_argument('--context',type=int,required=True);p.add_argument('--name',required=True)
    a=p.parse_args();t=AutoTokenizer.from_pretrained('/path/to/LLM/qwen36-api/model',local_files_only=True)
    body=document(t,a.context-700,781324)
    messages=[{'role':'user','content':body+'\nОтветь кратко: какие поля имеют записи?'}]
    count=len(t.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False))
    maximum=min(8192,a.context-count)
    assert 0<maximum<8192 and count>a.context*.98,(count,maximum)
    row,_=generate(messages,maximum,seed=6271,temperature=0)
    row.update(case='near-context-limit',planned_prompt_tokens=count,requested_output=maximum,context=a.context)
    assert row['status']==200 and row['usage']['prompt_tokens']==count
    payload={'model':'qwen36-35b','messages':messages,'max_tokens':maximum+1,'stream':True,
             'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
    r=requests.post('http://127.0.0.1:1919/v1/chat/completions',headers=headers(),json=payload,timeout=60)
    reject={'status':r.status_code,'code':r.json().get('error',{}).get('code')}
    passed=reject=={'status':400,'code':'context_length_exceeded'}
    (ROOT/'reports'/f'{a.name}.json').write_text(json.dumps({'generation':row,'overflow':reject,'pass':passed},indent=2))
    print(json.dumps({'generation':row,'overflow':reject,'pass':passed}));assert passed

if __name__=='__main__':main()
