"""Bounded fixture-code worker. Resource isolation, not a security sandbox."""
import ast
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile

MAX_INPUT = 16384
MAX_AST_NODES = 2048
WALL_TIMEOUT = 3.0
MEMORY_LIMIT = 192 * 1024 * 1024

def _run_code(text, *, timeout=WALL_TIMEOUT):
    if sys.platform != 'linux' or type(text) is not str or len(text) > MAX_INPUT or len(text.encode()) > MAX_INPUT:
        return False
    # Parsing/compilation happen only after the child's resource limits are set.
    with tempfile.TemporaryDirectory(prefix='qwen36-code-') as d:
        payload = Path(d)/'input.json'
        payload.write_text(json.dumps(text))
        output = Path(d)/'result'
        with output.open('wb') as out:
            child = subprocess.Popen([sys.executable, '-I', str(Path(__file__).resolve()), '--worker', str(payload)],
                                     stdout=out, stderr=subprocess.DEVNULL, cwd=d, start_new_session=True,
                                     env={'PATH':os.defpath, 'PYTHONHASHSEED':'0'})
            try:
                rc = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait()
                return False
            finally:
                if child.poll() is None:
                    try:os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait()
        return rc == 0 and output.stat().st_size == 2 and output.read_bytes() == b'1\n'

def code_check(text, *, timeout=WALL_TIMEOUT):
    try:
        return _run_code(text, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return False


def worker(path):
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024, 1024))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    text = json.loads(Path(path).read_text())
    fenced = re.search(r'```(?:python)?\s*(.*?)```', text, re.S)
    code = fenced.group(1) if fenced else text
    try:
        tree = ast.parse(code)
        allowed = (ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.Assign,
                   ast.Name,ast.Load,ast.Store,ast.Constant,ast.If,ast.Compare,ast.Eq,
                   ast.Lt,ast.Gt,ast.LtE,ast.GtE,ast.Not,ast.UnaryOp,ast.USub,
                   ast.BinOp,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod,
                   ast.Subscript,ast.Slice,ast.Call,ast.BoolOp,ast.And,ast.Or,ast.IfExp,ast.List,ast.Tuple)
        nodes = list(ast.walk(tree))
        if len(nodes) > MAX_AST_NODES:
            return False
        for node in nodes:
            if not isinstance(node, allowed):return False
            if isinstance(node,ast.Call) and (not isinstance(node.func,ast.Name) or node.func.id not in ('len','list')):return False
            if isinstance(node,ast.Name) and node.id.startswith('__'):return False
        if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):return False
        namespace={'__builtins__':{'len':len,'list':list}}
        exec(compile(tree,'<restricted-fixture>','exec'),namespace)
        f=namespace['rotate']
        return all(f(a,k)==expected for a,k,expected in [([],3,[]),([1,2,3,4],1,[4,1,2,3]),([1,2,3],0,[1,2,3]),([1,2,3],5,[2,3,1]),([1,2,3],-1,[2,3,1])])
    except (Exception, MemoryError):
        return False

if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--worker':raise SystemExit(2)
    try:result=worker(sys.argv[2])
    except Exception:result=False
    print('1' if result else '0')
