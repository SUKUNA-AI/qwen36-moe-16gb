#!/usr/bin/env python3
"""Read all tracked source text; emit scope/syntax/risk-index evidence, never run it."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
PATTERNS={'stream_order':r'wait_(event|stream)|synchronize\(|record_stream\(',
          'native_load':r'CDLL\(|load_library\(|load_jit\(|torch\.load\(',
          'process':r'shell=True|subprocess\.|os\.system\(|exec\(|eval\(',
          'checks':r'assert |sha256|qualified|receipt',
          'cluster_bulk':r'cp\.async\.bulk[^\n]*shared::cluster'}

def main():
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    files=[];counts=collections.Counter();errors=[]
    for name in names:
        p=ROOT/name
        if p.suffix not in ('.py','.sh','.ps1','.cu','.cuh','.cpp','.h','.hpp','.json','.md','.toml','.yaml','.yml') or not p.is_file():continue
        data=p.read_bytes();text=data.decode('utf-8-sig');counts[p.suffix]+=1
        row={'path':name,'sha256':hashlib.sha256(data).hexdigest(),'lines':len(text.splitlines())}
        row['signals']={key:[i for i,line in enumerate(text.splitlines(),1) if re.search(pattern,line)] for key,pattern in PATTERNS.items()}
        row['signals']={k:v for k,v in row['signals'].items() if v}
        try:
            if p.suffix=='.py':ast.parse(text,filename=name)
            elif p.suffix=='.sh':
                # Public archive contains CRLF; syntax-check normalized text without execution.
                r=subprocess.run(['bash','-n'],input=text.replace('\r\n','\n'),text=True,capture_output=True)
                if r.returncode:raise ValueError(r.stderr.strip())
        except (SyntaxError,ValueError) as e:errors.append({'path':name,'error':str(e)})
        files.append(row)
    result={'scope':'all tracked text sources/configs/docs, automated inventory; not a proof of correctness',
            'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'counts':dict(counts),'files':files,'syntax_errors':errors}
    json.dump(result,sys.stdout,indent=2);print()
    return bool(errors)
if __name__=='__main__':raise SystemExit(main())
