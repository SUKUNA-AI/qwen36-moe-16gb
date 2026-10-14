import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import cpu_checks as m
class Entry(unittest.TestCase):
    def test_counts_zero_and_real_script_scenarios(self):
        self.assertEqual(m.counts('Ran 0 tests in 0s','x')[0],0)
        self.assertEqual(m.counts('', 'test-guard.py')[0],0)
        self.assertEqual(m.counts('Ran 3 tests in 0s\nOK (skipped=1)','x'),(3,0,1))
        self.assertEqual(m.counts('{"test_count":9}','x')[0],9)
    def test_resolver_other_cwd_and_spaces(self):
        with tempfile.TemporaryDirectory(prefix='checkout with spaces ') as d:
            code='import repo_paths; print(repo_paths.CHECKOUT)'
            p=subprocess.run([sys.executable,'-c',code],cwd='/',env=dict(os.environ,QWEN36_CHECKOUT=d),capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(p.stdout.strip(),d)
    def test_collection_failure_and_empty_suite_fail_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);h=root/'harness';s=root/'serving';h.mkdir();s.mkdir()
            for name in ('test-guard.py','test-metrics.py'):(s/name).write_text('pass')
            (h/'test-log-failure.py').write_text('pass')
            (h/'test-zero-cpu.py').write_text('import unittest\nunittest.main()')
            fake=subprocess.CompletedProcess([],0,'Ran 0 tests in 0s\n','')
            with patch.object(m,'HARNESS',h),patch.object(m,'SERVING',s),patch.object(m.sys,'argv',['runner','--output',str(root/'logs')]),\
                 patch.object(m.subprocess,'run',return_value=fake):
                self.assertEqual(m.main(),1)
            fake=subprocess.CompletedProcess([],1,'','ModuleNotFoundError: required CPU dependency')
            with patch.object(m,'HARNESS',h),patch.object(m,'SERVING',s),patch.object(m.sys,'argv',['runner','--output',str(root/'logs')]),\
                 patch.object(m.subprocess,'run',return_value=fake):
                self.assertEqual(m.main(),1)
if __name__=='__main__':unittest.main(verbosity=2)
