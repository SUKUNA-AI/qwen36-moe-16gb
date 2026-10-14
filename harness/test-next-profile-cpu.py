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
        p=json.loads((SERVING/'profiles.json').read_text())['profiles']['main128'];p['qualified']=False;return p
    def test_threshold_schema_combinations_and_hash(self):
        p=self.profile();before=r.canonical_profile_sha256(p)
        for value in (True,False,-1,'2049',2.5,None,2147483648):
            with self.subTest(value=value),self.assertRaises(ValueError):r.validate_profile_schema('main128',dict(p,layer_major_min_tokens=value))
        for value in (0,2049):r.validate_profile_schema('main128',dict(p,layer_major_min_tokens=value))
        both=dict(p,flags=['--moe-prefill-hit-d2d'])
        for value in (None,0):
            with self.assertRaises(ValueError):r.validate_profile_schema('main128',dict(both,**({} if value is None else {'layer_major_min_tokens':value})))
        r.validate_profile_schema('main128',dict(both,layer_major_min_tokens=2049))
        with self.assertRaises(ValueError):r.validate_profile_schema('main128',dict(p,layer_major_prefill=False,layer_major_min_tokens=2049))
        self.assertNotEqual(before,r.canonical_profile_sha256(dict(p,layer_major_min_tokens=2049)))
        chunked=dict(p,layer_major_prefill=False,flags=['--moe-prefill-hit-d2d']);chunked.pop('layer_major_attn_batch',None)
        r.validate_profile_schema('main128',chunked)
if __name__=='__main__':unittest.main(verbosity=2)
