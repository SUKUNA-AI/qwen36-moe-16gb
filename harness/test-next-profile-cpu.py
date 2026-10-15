import copy
import importlib.util
import json
from pathlib import Path
import unittest
from repo_paths import SERVING,HARNESS

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=load('next_profile',SERVING/'profile-runtime.py');q=load('next_qualification',HARNESS/'make-qualification.py')
class NextProfile(unittest.TestCase):
    def profile(self):
        p=json.loads((SERVING/'profiles.json').read_text())['profiles']['main128'];p['qualified']=False
        # the released f1 fields: the main128 release draft also carries the T1/T2/T3 settings under test here
        for key in ('layer_major_min_tokens','cuda_gdn_fla','decode_shared_overlap','host_embeddings','cuda_oproj_gemv',
                    'cuda_moe_decode','prefill_copy_priority','prefill_selected','prefill_warm_decode'):p.pop(key,None)
        p['flags']=[];return p
    def test_threshold_schema_combinations_and_hash(self):
        p=self.profile();before=r.canonical_profile_sha256(p)
        for value in (True,False,-1,'2049',2.5,None,2147483648):
            with self.subTest(value=value),self.assertRaises(ValueError):r.validate_profile_schema('main128',dict(p,layer_major_min_tokens=value))
        for value in (0,2049):r.validate_profile_schema('main128',dict(p,layer_major_min_tokens=value))
        for value in (p['chunk']+2,2147483647):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'chunk \\+ 1'):
                r.validate_profile_schema('main128',dict(p,layer_major_min_tokens=value))
        both=dict(p,flags=['--moe-prefill-hit-d2d'])
        for value in (None,0):
            with self.assertRaises(ValueError):r.validate_profile_schema('main128',dict(both,**({} if value is None else {'layer_major_min_tokens':value})))
        r.validate_profile_schema('main128',dict(both,layer_major_min_tokens=2049))
        with self.assertRaises(ValueError):r.validate_profile_schema('main128',dict(p,layer_major_prefill=False,layer_major_min_tokens=2049))
        self.assertNotEqual(before,r.canonical_profile_sha256(dict(p,layer_major_min_tokens=2049)))
        chunked=dict(p,layer_major_prefill=False,flags=['--moe-prefill-hit-d2d']);chunked.pop('layer_major_attn_batch',None)
        r.validate_profile_schema('main128',chunked)
    def test_fla_values_env_and_unchanged_profiles(self):
        p=self.profile();original=r.canonical_profile_sha256(p)
        for value in (True,False,'1','unknown','',0,None):
            with self.subTest(value=value),self.assertRaises(ValueError):r.validate_profile_schema('main128',dict(p,cuda_gdn_fla=value))
        for value in ('wu','ho'):
            profile=dict(p,cuda_gdn_fla=value);r.validate_profile_schema('main128',profile)
            env={'QWEN36_CUDA_GDN_FLA':'unknown'};r.configure_environment('main128',profile,env)
            self.assertEqual(env['QWEN36_CUDA_GDN_FLA'],value)
            self.assertNotEqual(original,r.canonical_profile_sha256(profile))
            with self.assertRaises(ValueError):r.validate_profile_schema('main128',dict(profile,cuda_gdn=False))
        env={'QWEN36_CUDA_GDN_FLA':'wu'};r.configure_environment('main128',p,env);self.assertNotIn('QWEN36_CUDA_GDN_FLA',env)
        r.validate_profile_schema('main128',p)
    def test_fla_receipt_requires_separate_variant_library_and_component_binding(self):
        p=self.profile();p['cuda_gdn_fla']='wu';evidence={'library':r.GDN_LIBRARY,'library_sha256':'a'*64,
            'run_receipts':['run'],'bitexact_reports':['teacher']}
        record={'evidence':[{'path':x,'sha256':'b'*64} for x in ('run','teacher','component')]}
        manifest={}
        for switch in ('cuda_nvfp4_prefill','cuda_gdn','cuda_fp8_gemv'):
            if p.get(switch):
                lib=r.CUDA_EVIDENCE_LIBRARIES[switch];manifest[lib]='a'*64;record[switch]=dict(evidence,library=lib)
        with self.assertRaisesRegex(ValueError,'cuda_gdn_fla needs'):r._cuda_evidence_valid('main128',p,record,manifest)
        record['cuda_gdn_fla']=dict(evidence,variant='wu',component_reports=['component'])
        r._cuda_evidence_valid('main128',p,record,manifest)
        for key,value in (('variant','ho'),('library_sha256','bad'),('component_reports',[]),('component_reports',['unlisted'])):
            bad=copy.deepcopy(record);bad['cuda_gdn_fla'][key]=value
            with self.assertRaises(ValueError):r._cuda_evidence_valid('main128',p,bad,manifest)
        component={'all_pass':True,'library_sha256':'a'*64,'fla_fused_candidate':True,'fla_variant':'wu',
                   'fla_fused_calls':2,'cases':1,'results':[{'case':'synthetic','pass':True}]}
        q.check_fla_component_report(component,'wu','a'*64)
        for key,value in (('library_sha256','bad'),('fla_variant','ho'),('fla_fused_calls',0),('results',[])):
            with self.assertRaises(ValueError):q.check_fla_component_report(dict(component,**{key:value}),'wu','a'*64)
    def test_raw_engine_fla_parser(self):
        from freetoken.gdn_options import fla_variant
        for enabled in (False,True):
            for raw in (None,'','0'):self.assertIsNone(fla_variant(enabled,raw))
        for raw,want in (('1','wu'),('wu','wu'),('ho','ho')):
            self.assertEqual(fla_variant(True,raw),want)
            with self.assertRaises(ValueError):fla_variant(False,raw)
        for raw in ('unknown','on','WU',True,1):
            with self.assertRaises(ValueError):fla_variant(True,raw)

    def test_candidate_configs_are_explicit_and_unqualified(self):
        for name in ('main-release-shortroute','main-release-fla-wu','main-release-shortroute-fla-wu'):
            cfg=json.loads((HARNESS/(name+'.json')).read_text());self.assertIs(cfg['qualified'],False)
            mapped=q.run_equivalent_profile(cfg,r.FEATURE_SWITCHES,r.KERNEL_LOADER_ENV,r.VALUE_FIELDS)
            self.assertEqual(mapped.get('cuda_gdn_fla'),cfg['env'].get('QWEN36_CUDA_GDN_FLA'))
if __name__=='__main__':unittest.main(verbosity=2)
