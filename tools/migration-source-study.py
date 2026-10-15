#!/usr/bin/env python3
"""Compare public upstream snapshots; no installs, compilation or service operations."""
from pathlib import Path
import ast,hashlib,json,subprocess,tempfile
import argparse
parser=argparse.ArgumentParser(description="Read-only source/three-way merge study; never edits the input trees")
parser.add_argument('--fork',type=Path,required=True,help='this repository root')
parser.add_argument('--base',type=Path,required=True,help='pristine FreeToken base checkout')
parser.add_argument('--target',type=Path,required=True,help='pristine FreeToken target checkout')
parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args();R=a.fork.resolve();B=a.base.resolve();U=a.target.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=True)
base_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=B,text=True).strip()
target_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=U,text=True).strip()
def read(p):return p.read_bytes().replace(b'\r\n',b'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
rows=[]
with tempfile.TemporaryDirectory() as temp:
 for p in sorted((R/'freetoken').rglob('*')):
  if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.cpp','.h','.toml','.sh'):continue
  rel=p.relative_to(R/'freetoken');b=B/rel;u=U/rel;ours=read(p)
  if b.exists() and ours==read(b):continue
  row={'path':str(rel),'fork_sha256_lf':sha(ours),'base_exists':b.exists(),'upstream_exists':u.exists()}
  if b.exists():row['base_sha256_lf']=sha(read(b))
  if u.exists():row['upstream_sha256_lf']=sha(read(u))
  if b.exists() and u.exists():
   row['upstream_changed']=read(b)!=read(u)
   for key,data in [('ours',ours),('base',read(b)),('theirs',read(u))]:Path(temp,key).write_bytes(data)
   r=subprocess.run(['git','merge-file','-p',str(Path(temp,'ours')),str(Path(temp,'base')),str(Path(temp,'theirs'))],capture_output=True)
   row['textual_merge_conflicts']=r.returncode
  else:row['status']='fork-only' if not b.exists() and not u.exists() else 'manual-add-delete'
  rows.append(row)
(O/'fork-rebase.json').write_text(json.dumps({'base':base_sha,'target':target_sha,'line_endings':'LF normalized for diff only','files':rows},indent=2)+'\n')
paths=['python/freetoken/kernel/triton/nvfp4_fused_moe.py','python/freetoken/moe/fused_nvfp4.py','python/freetoken/attention/fi.py','python/freetoken/kernel/triton/fp8_pertensor_linear.py','python/freetoken/kernel/triton/nvfp4_linear.py','python/freetoken/models/qwen3_5_moe/gdn.py']
paths+=list(str(p.relative_to(B)) for p in (B/'python/freetoken/kernel/fla').glob('*.py'))
sources=[]
for rel in paths:
 b,u=read(B/rel),read(U/rel);row={'path':rel,'base_sha256_lf':sha(b),'target_sha256_lf':sha(u),'identical':b==u}
 if rel.endswith('nvfp4_fused_moe.py'):
  def fn(data):
   tree=ast.parse(data);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_prefill_nvfp4_moe_kernel');return ast.dump(node,include_attributes=False)
  row['prefill_function_ast_identical']=fn(b)==fn(u)
 sources.append(row)
(O/'kernel-source-comparison.json').write_text(json.dumps(sources,indent=2)+'\n')
print('fork changed/added',len(rows),'textual conflicts',[(r['path'],r['textual_merge_conflicts']) for r in rows if r.get('textual_merge_conflicts',0)])
print('kernel unchanged',sum(r['identical'] for r in sources),'of',len(sources))
