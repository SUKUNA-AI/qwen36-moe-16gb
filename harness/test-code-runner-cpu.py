import time
import unittest
from unittest.mock import patch
import code_runner as m
GOOD='def rotate(a,k):\n    if not a: return []\n    k = k % len(a)\n    return a[-k:] + a[:-k] if k else list(a)\n'
class CodeRunner(unittest.TestCase):
    def test_valid_rotate_and_fence(self):
        self.assertTrue(m.code_check(GOOD));self.assertTrue(m.code_check('```python\n'+GOOD+'```'))
    def test_wrong_exception_and_forbidden_ast(self):
        for code in ('def rotate(a,k): return a', 'def rotate(a,k): return 1/0',
                     'import os\n'+GOOD,'def rotate(a,k): return a.copy()',
                     'def rotate(a,k):\n    while True: pass'):
            self.assertFalse(m.code_check(code))
    def test_memory_exhaustion_and_heavy_ast_are_child_bounded(self):
        start=time.monotonic()
        self.assertFalse(m.code_check('def rotate(a,k): return [0] * 1000000000'))
        self.assertFalse(m.code_check('def rotate(a,k): return '+('('*2000)+'1'+(')'*2000)))
        self.assertLess(time.monotonic()-start,6)
        self.assertTrue(m.code_check(GOOD))
    def test_wall_timeout_cleans_up_process(self):
        children=[];original=m.subprocess.Popen
        def spawn(*a,**kw):
            child=original(*a,**kw);children.append(child);return child
        with patch.object(m.subprocess,'Popen',side_effect=spawn):
            self.assertFalse(m.code_check(GOOD,timeout=.000001))
        self.assertIsNotNone(children[0].poll())
    def test_spawn_failure_is_controlled(self):
        with patch.object(m.subprocess,'Popen',side_effect=OSError('resource exhausted')):
            self.assertFalse(m.code_check(GOOD))
    def test_input_limit_and_unsupported_platform(self):
        self.assertFalse(m.code_check('x'*20000))
        with patch.object(m.sys,'platform','win32'):self.assertFalse(m.code_check(GOOD))
if __name__=='__main__':unittest.main(verbosity=2)
