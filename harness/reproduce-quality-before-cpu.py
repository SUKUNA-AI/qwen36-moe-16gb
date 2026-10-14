"""Reproduce original false acceptance with public synthetic data, no model."""
import json,pathlib,subprocess,tempfile,sys
repo=pathlib.Path(__file__).resolve().parents[1]
source=subprocess.check_output(['git','-C',str(repo),'show','7c78a4824faf03c0eee6f8b3e6fa9cd4cf7df401:harness/compare-quality.py'],text=True)
with tempfile.TemporaryDirectory() as d:
 root=pathlib.Path(d);reports=root/'reports';reports.mkdir()
 rows=[]
 for profile in ('teacher-bf128','teacher-fp128'):
  for i in range(20):rows.append({'profile':profile,'case':f'case-{i}','run_id':'historical-unbound','nll':[1.],
     'logit_comparisons':[{'finite':True,'max_abs':.25,'relative_l2':.01}]})
 (reports/'teacher.jsonl').write_text('\n'.join(map(json.dumps,rows)))
 (reports/'teacher-fp128.json').write_text(json.dumps({'rows':[{'case':f'case-{i}','status':200,'forced_text_matches':True,'usage':{'completion_tokens':1},'expected_tokens':1} for i in range(20)]}))
 (reports/'functional-bf128.json').write_text(json.dumps({'rows':[{'case':'russian','pass':True}]}))
 (reports/'functional-fp128.json').write_text(json.dumps({'rows':[]}))
 source=source.replace("ROOT=pathlib.Path('/home/USER/qwen36-api/optimization')",f'ROOT=pathlib.Path({str(root)!r})')
 script=root/'original.py';script.write_text(source)
 result=subprocess.run([sys.executable,str(script)],capture_output=True,text=True)
 print('original exit:',result.returncode)
 report=json.loads(result.stdout)
 print(json.dumps({k:report[k] for k in ('quality_gate_pass','new_functional_failures','logit_comparisons')}))
 if result.returncode != 0 or report['quality_gate_pass'] is not True:raise SystemExit(1)
