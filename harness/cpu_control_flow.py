"""Run actual method bodies with controlled dependencies and no model imports.

AST only locates the method. Python executes its real control flow; this CPU
regression tool does not establish CUDA race freedom or graph correctness.
"""
import ast
import __future__

def methods(path, class_name, names, namespace, mutate=None):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(body) != len(names):
        raise RuntimeError('required control-flow method missing')
    module = ast.Module(body=body, type_ignores=[])
    if mutate is not None:
        mutate(module)
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), 'exec', flags=__future__.annotations.compiler_flag), namespace)
    return namespace
