"""Synthetic Git/JSON fixtures only. Never reads live reports, overlay or model files."""
import argparse
import collections
import importlib.util
import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPT=pathlib.Path(__file__).with_name('export-release-evidence.py')
spec=importlib.util.spec_from_file_location('export_fixture_module',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ExportFixture(unittest.TestCase):
    def setUp(self):
        workspace=SCRIPT.resolve().parents[3]
        scratch=workspace/'work';scratch.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='export-fixture-',dir=scratch)
        self.root=pathlib.Path(self.temp.name).resolve()
        assert self.root.is_relative_to(scratch.resolve())
        self.repo=self.root/'work/FreeToken-opt';self.repo.mkdir(parents=True)
        self.public=self.root/'outputs/qwen36-api/optimization';self.public.mkdir(parents=True)
        self.service=self.public.parent
        self.run_git('init','--quiet')
        self.file('python/freetoken/engine/engine.py',b'VALUE = 1\n')
        self.file('python/freetoken/core.py',b'OLD = True\n')
        self.run_git('add','.')
        self.run_git('-c','user.name=Synthetic Fixture','-c','user.email=fixture@example.invalid','commit','--quiet','-m','Synthetic base')
        self.base=self.run_git('rev-parse','HEAD').decode().strip()
        self.file('python/freetoken/engine/engine.py',b'VALUE = 2\n')
        self.file('python/freetoken/engine/telemetry.py',b'ENABLED = False\n')
        for name in m.OVERLAY:(self.service/name).write_bytes(b'{}\n' if name.endswith('.json') else b'# synthetic source\n')
        (self.public/'qwen36.cmd').write_bytes(b'@rem synthetic launcher\n')
        self.overrides=patch.multiple(m,WORKSPACE=self.root,PUBLIC=self.public,SERVICE=self.service,BASE=self.base)
        self.overrides.start()

    def tearDown(self):
        self.overrides.stop()
        # The one generated fixture tree is the only recursively removed path.
        self.assertTrue(self.root.is_relative_to(SCRIPT.resolve().parents[3]/'work'))
        self.temp.cleanup()

    def run_git(self,*args):
        return subprocess.run(['git','-C',str(self.repo),'-c','core.autocrlf=false',*args],capture_output=True,check=True,timeout=20).stdout

    def file(self,name,data):
        path=self.repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)

    def export(self,selectors=(),name='fixture'):
        return m.export(argparse.Namespace(name=name,no_default_results=True,result=list(selectors),native_root=str(self.root/'native-does-not-exist')))

    def evidence(self,value):
        (self.public/'synthetic.json').write_text(json.dumps(value),encoding='utf-8')

    def test_patch_roundtrip_tracked_new_and_deleted(self):
        (self.repo/'python/freetoken/core.py').unlink()
        self.file('python/freetoken/engine/telemetry.py',b'ENABLED = False')
        self.file('python/freetoken/engine/speculative.py',b'')
        self.file('python/freetoken/engine/mtp.py',b'CRLF = True\r\n')
        before=self.run_git('status','--porcelain')
        receipt=self.export()
        target=self.root/receipt['export']
        manifest=json.loads((target/'manifest.json').read_text())
        self.assertFalse(manifest['qualified_flags_changed'])
        self.assertEqual(manifest['qualification'],'not_evaluated_or_modified_by_exporter')
        self.assertEqual(before,self.run_git('status','--porcelain'))
        clean=self.root/'clean'
        subprocess.run(['git','-c','core.autocrlf=false','clone','--quiet',str(self.repo),str(clean)],check=True,capture_output=True,timeout=20)
        patchpath=target/'runtime/integration.patch'
        subprocess.run(['git','-C',str(clean),'-c','core.autocrlf=false','apply','--check',str(patchpath)],check=True,capture_output=True,timeout=20)
        subprocess.run(['git','-C',str(clean),'-c','core.autocrlf=false','apply',str(patchpath)],check=True,capture_output=True,timeout=20)
        for row in manifest['runtime_files']:
            if row['sha256'] is None:self.assertFalse((clean/row['path']).exists())
            else:self.assertEqual((clean/row['path']).read_bytes(),(self.repo/row['path']).read_bytes())

    def test_lossy_json_preserves_numbers_flags_and_hash(self):
        self.evidence({'qualified':False,'pass':True,'ttft_s':1.23,'token_count':'123',
          'profile':'main128','nvcc':'Cuda compilation tools, release 13.4, V13.4.59',
          'sha256':'a'*64,'Authorization':'Bearer NOT_REAL_FIXTURE','messages':[{'content':'PRIVATE'}],
          'prompt':'PRIVATE','stdout':'PRIVATE','run_id':'PRIVATE','token_ids':[1,2,3],
          'input_tokens':[1,2,3],'hostname':'PRIVATE','path':'/home/PRIVATE/data','api_secret':'PRIVATE',
          'key':'f'*64,'untyped_fixture_value':'e'*64,'env':{'COUNT':123}})
        receipt=self.export(['public:synthetic.json']);target=self.root/receipt['export']
        result=json.loads((target/'evidence/public/synthetic.json').read_text())
        self.assertEqual(result,{'qualified':False,'pass':True,'ttft_s':1.23,'token_count':123,
          'profile':'main128','nvcc':'13.4','sha256':'a'*64})
        self.assertNotIn('PRIVATE',json.dumps(result))

    def test_refuses_existing_destination(self):
        self.export()
        with self.assertRaisesRegex(ValueError,'already exists'):self.export()

    def test_refuses_unknown_changed_source(self):
        self.file('python/freetoken/unknown_fixture.py',b'X = 1\n')
        with self.assertRaisesRegex(ValueError,'feature mapping'):self.export()

    def test_refuses_head_moved(self):
        self.run_git('add','.')
        self.run_git('-c','user.name=Synthetic Fixture','-c','user.email=fixture@example.invalid','commit','--quiet','-m','Changed head')
        with self.assertRaisesRegex(ValueError,'HEAD moved'):self.export()

    def test_refuses_path_escape_raw_secret_and_missing(self):
        for i,selector in enumerate(('public:../outside.json','native:secrets/key.json',
          'public:raw-profile.json','public:missing.json','service:guard.json',
          'native:reports/raw/private.json','public:Secrets/private.json')):
            with self.subTest(selector=selector),self.assertRaises(ValueError):self.export([selector],name='invalid-'+str(i))

    def test_rejects_literal_credential_in_current_source(self):
        self.file('python/freetoken/engine/telemetry.py',b'api_key = "'+b'a'*32+b'"\n')
        with self.assertRaisesRegex(ValueError,'credential'):self.export()

    def test_rejects_deleted_credential_in_patch(self):
        self.file('python/freetoken/core.py',b'api_key = "'+b'z'*32+b'"\n')
        self.run_git('add','python/freetoken/core.py')
        self.run_git('-c','user.name=Synthetic Fixture','-c','user.email=fixture@example.invalid','commit','--quiet','-m','Synthetic historical credential')
        fixture_base=self.run_git('rev-parse','HEAD').decode().strip()
        self.file('python/freetoken/core.py',b'SAFE = True\n')
        with patch.object(m,'BASE',fixture_base),self.assertRaisesRegex(ValueError,'credential'):self.export()

    def test_sanitization_collision_refused(self):
        self.evidence({'/home/one/file.so':'a'*64,'/home/two/file.so':'b'*64})
        with self.assertRaisesRegex(ValueError,'collision'):self.export(['public:synthetic.json'])

    def test_nonfinite_and_depth_are_bounded(self):
        counts=collections.Counter()
        self.assertIsNone(m.scrub(float('nan'),counts))
        value=1
        for _ in range(70):value={'nested':value}
        json.dumps(m.scrub(value,counts),allow_nan=False)
        self.assertGreater(counts['depth_limit'],0)

    def test_export_maps_bundle_files_and_switch_names(self):
        self.file('python/freetoken/kernel/decode_gather.py',b'ENV = "QWEN36_DECODE_GATHER"\n')
        self.file('python/freetoken/scheduler/layer_major.py',b'ENV = "QWEN36_LAYER_MAJOR_PREFILL"\n')
        receipt=self.export();target=self.root/receipt['export']
        rows={row['path']:row for row in json.loads((target/'manifest.json').read_text())['runtime_files']}
        self.assertEqual(rows['python/freetoken/kernel/decode_gather.py']['groups'],['C'])
        self.assertEqual(rows['python/freetoken/kernel/decode_gather.py']['switch_names_observed'],['QWEN36_DECODE_GATHER'])
        self.assertEqual(rows['python/freetoken/scheduler/layer_major.py']['groups'],['C'])
        feature=json.loads((target/'runtime/features/C.json').read_text())
        self.assertTrue({'QWEN36_DECODE_GATHER','QWEN36_LAYER_MAJOR_PREFILL','QWEN36_LAYER_MAJOR_REFILL'}<=set(feature['switches']))


PROFILE_RUNTIME=SCRIPT.resolve().parent.parent/'profile-runtime.py'


class BundleMappingTests(unittest.TestCase):
    """The exporter's feature groups cover every production switch variable and the bundle's source files."""

    def test_switch_names_cover_production_environment(self):
        spec=importlib.util.spec_from_file_location('profile_runtime_for_export_test',PROFILE_RUNTIME)
        runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
        exported={key for exports in runtime.SWITCH_ENV.values() for key in exports}
        exported|={env for env,_ in runtime.NATIVE_LIBRARIES.values()}|{'QWEN36_PREFIX_MARGIN'}
        exported|={spec[0] for spec in runtime.VALUE_FIELDS.values()}
        listed={name for group in m.GROUPS.values() for name in group['switches']}
        self.assertEqual(exported-listed,set())
        groups={key:set(group['switches']) for key,group in m.GROUPS.items()}
        self.assertTrue({'QWEN36_CUDA_GDN','QWEN36_CUDA_GDN_LIBRARY','QWEN36_CUDA_GDN_DECODE','QWEN36_FP8_GEMV',
                         'QWEN36_FP8_GEMV_LIBRARY'}<=groups['B'])
        self.assertTrue({'QWEN36_LAYER_MAJOR_PREFILL','QWEN36_LAYER_MAJOR_REFILL','QWEN36_DECODE_GATHER'}<=groups['C'])

    def test_bundle_sources_have_feature_groups(self):
        expected={'python/freetoken/kernel/cuda_gdn.py':'B','python/freetoken/kernel/cuda_fp8_gemv.py':'B',
                  'python/freetoken/kernel/triton/fp8_pertensor_linear.py':'B',
                  'python/freetoken/layers/quantization/linear/fp8_tensor.py':'B',
                  'python/freetoken/models/qwen3_5_moe/gdn.py':'B','tests/kernels/test_cuda_fp8_gemv.py':'B',
                  'python/freetoken/engine/layer_major.py':'C','python/freetoken/moe/layer_major.py':'C',
                  'python/freetoken/scheduler/layer_major.py':'C','python/freetoken/layers/moe.py':'C',
                  'python/freetoken/kernel/decode_gather.py':'C','python/freetoken/kernel/csrc/jit/decode_gather.cuh':'C',
                  'python/freetoken/kernel/aot.py':'C','tests/moe/test_decode_gather.py':'C',
                  'python/freetoken/moe/slot_refill.py':'C'}
        for name,group in expected.items():
            with self.subTest(name):
                self.assertTrue(m.allowed_source(name))
                self.assertIn(group,m.feature_groups(name))


if __name__=='__main__':unittest.main(verbosity=2)
