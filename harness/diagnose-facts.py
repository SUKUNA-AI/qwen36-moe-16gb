"""Local frozen quality-case localization; no input/output text is retained."""
import argparse,json,re,hashlib
from transformers import AutoTokenizer
from bench import ROOT,document,generate

p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--repeats',type=int,default=2);a=p.parse_args()
t=AutoTokenizer.from_pretrained('/path/to/LLM/qwen36-api/model',local_files_only=True)
length=16384;body=document(t,length,9721+length);i=len(body)//3;j=2*len(body)//3
content='Код задачи АЛЬФА-917.\n'+body[:i]+'\nКоличество деталей: 23.\n'+body[i:j]+'\nОтветственный: Ирина.\n'+body[j:]+'\nЦена детали 14 рублей. Ответь только JSON с полями code, count, owner, total для этой задачи.'
expected={'code':'АЛЬФА-917','count':23,'owner':'Ирина','total':322};rows=[]
for repeat in range(a.repeats):
    metrics,text=generate([{'role':'user','content':content}],256,temperature=0)
    parsed=None
    try:parsed=json.loads(re.search(r'\{.*\}',text,re.S).group(0))
    except Exception:pass
    row={'repeat':repeat,'parse_pass':isinstance(parsed,dict),'sha256':hashlib.sha256(text.encode()).hexdigest(),
         'field_equal':{k:isinstance(parsed,dict) and parsed.get(k)==v for k,v in expected.items()},
         'field_type':{k:type(parsed.get(k)).__name__ if isinstance(parsed,dict) else None for k in expected},
         'extra_key_count':len(set(parsed)-set(expected)) if isinstance(parsed,dict) else None,'metrics':metrics}
    rows.append(row);print(json.dumps(row),flush=True)
(ROOT/'reports'/f'{a.name}.json').write_text(json.dumps(rows,indent=2))
