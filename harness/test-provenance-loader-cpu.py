import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from repo_paths import FREETOKEN
ROOT=FREETOKEN/'python/freetoken/kernel'
package=types.ModuleType('isolated_kernel');package.__path__=[str(ROOT)];sys.modules[package.__name__]=package

def load(name):
    spec=importlib.util.spec_from_file_location('isolated_kernel.'+name,ROOT/(name+'.py'))
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
p=load('kernel_provenance');u=load('utils')
class Provenance(unittest.TestCase):
    def test_strict_loader_refuses_before_load(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'csrc';src.mkdir();(src/'x.cu').write_text('// synthetic')
            recipe=p.make_recipe('x','jit',[src/'x.cu'],[],{},csrc=src)
            so=root/'x/x.so';so.parent.mkdir();so.write_bytes(b'synthetic, never dlopened')
            entry={'recipe':recipe,'recipe_digest':p.recipe_digest(recipe),'so_sha256':p.file_sha256(so)}
            good={'schema':p.SCHEMA,'modules':{'x':entry},'upstream':{'csrc':recipe['sources']}}
            loaded=[];ffi=types.ModuleType('tvm_ffi');ffi.load_module=lambda path:loaded.append(path) or object()
            for manifest in (None,dict(good,modules={}),dict(good,modules={'x':dict(entry,recipe_digest='bad')}),
                             dict(good,modules={'x':dict(entry,so_sha256='bad')})):
                with self.subTest(manifest=manifest),patch.dict(os.environ,{'FREETOKEN_STRICT_PROVENANCE':'1'},clear=True),\
                     patch.object(u,'_kernel_cache_dir',return_value=root),patch.object(p,'load_manifest',return_value=manifest),\
                     patch.dict(sys.modules,{'tvm_ffi':ffi}):
                    with self.assertRaises(p.ProvenanceError):u._load_prebuilt('x',recipe)
                    self.assertEqual(loaded,[])
            changed=dict(recipe,sources={**recipe['sources'],'x.cu':'changed'})
            with patch.dict(os.environ,{'FREETOKEN_STRICT_PROVENANCE':'1'},clear=True),patch.object(u,'_kernel_cache_dir',return_value=root),\
                 patch.object(p,'load_manifest',return_value=good),patch.dict(sys.modules,{'tvm_ffi':ffi}):
                with self.assertRaises(p.ProvenanceError):u._load_prebuilt('x',changed)
                self.assertEqual(loaded,[])
            with patch.dict(os.environ,{'FREETOKEN_STRICT_PROVENANCE':'1'},clear=True),\
                 patch.object(u,'_kernel_cache_dir',return_value=root),patch.object(p,'load_manifest',return_value=good),\
                 patch.dict(sys.modules,{'tvm_ffi':ffi}):
                u._load_prebuilt('x',recipe)
            self.assertEqual(loaded,[str(so)])
    def test_dev_compatibility_and_strict_no_jit_contract(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);so=root/'x.so';so.write_bytes(b'synthetic');recipe={'sources':{'x.cu':'a'},'include':{}}
            manifest={'upstream':{'csrc':{'x.cu':'a'}},'modules':{}}
            self.assertEqual(p.decide_prebuilt('x',recipe,so,manifest=manifest,cache_dir=None)['provenance'],'prebuilt:upstream-unverified')
            self.assertFalse(p.decide_prebuilt('x',recipe,so,manifest=manifest,cache_dir=None,strict=True)['use'])
            with patch.dict(os.environ,{'FREETOKEN_STRICT_PROVENANCE':'1'},clear=True),patch.object(u,'_kernel_cache_dir',return_value=None):
                with self.assertRaises(RuntimeError):u._load_prebuilt('x',recipe)
if __name__=='__main__':unittest.main(verbosity=2)
