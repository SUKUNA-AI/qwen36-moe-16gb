import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import requests
spec=importlib.util.spec_from_file_location('profile_request',Path(__file__).with_name('profile-request.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Health(unittest.TestCase):
    def test_models_ready_health_not_ready_and_authorization(self):
        response=unittest.mock.Mock(status_code=200);response.json.return_value={'status':'loading'}
        with patch('bench.headers',return_value={'Authorization':'Bearer synthetic'}),patch('requests.get',return_value=response) as get:
            self.assertFalse(m.http_ready());self.assertTrue(get.call_args.args[0].endswith('/health'))
            self.assertEqual(get.call_args.kwargs['headers'],{'Authorization':'Bearer synthetic'})
    def test_ready_transition(self):
        now=[0.]; seen=[]
        def probe(timeout):
            seen.append(timeout);return len(seen)==2,'loading'
        m.wait_health(5,probe=probe,clock=lambda:now[0],sleep=lambda n:now.__setitem__(0,now[0]+n))
        self.assertEqual(len(seen),2)
    def test_timeout_and_bounded_request(self):
        now=[0.]; seen=[]
        def probe(timeout):seen.append(timeout);return False,'HTTP 200: loading'
        with self.assertRaisesRegex(RuntimeError,'loading'):
            m.wait_health(2,probe=probe,clock=lambda:now[0],sleep=lambda n:now.__setitem__(0,now[0]+n))
        self.assertEqual(seen,[2,1])
    def test_network_error(self):
        with patch('bench.headers',return_value={}),patch('requests.get',side_effect=requests.ConnectionError('synthetic')):
            ready,reason=m.health_probe();self.assertFalse(ready);self.assertIn('ConnectionError',reason)
if __name__=='__main__':unittest.main(verbosity=2)
