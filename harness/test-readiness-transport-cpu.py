"""Exercise real urllib policy with in-memory transports; no sockets or key files."""
import contextlib
import email.message
import importlib.util
import io
import json
import os
import pathlib
import socket
import ssl
import unittest
import urllib.error
import urllib.parse
import urllib.request
import urllib.response
from unittest import mock

path = __import__("repo_paths").SERVING / 'readiness.py'
spec = importlib.util.spec_from_file_location('readiness_transport_fixture', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
TOKEN = 'SYNTHETIC-NOT-A-CREDENTIAL'
GOOD = {'status':'ok', 'model':'qwen36-35b', 'maintenance':'serving', 'instance_id':'generation-a'}


class MemoryTransport(urllib.request.BaseHandler):
    # ProxyHandler runs at 100. Inspect its effect, then satisfy HTTP ourselves.
    handler_order = 150

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def http_open(self, request):
        self.calls.append({'url':request.full_url, 'host':request.host,
            'authorization':request.get_header('Authorization'), 'proxy':request.has_proxy()})
        if request.full_url not in self.responses:
            raise AssertionError('Unexpected destination; third party must never receive credential')
        code, data, location = self.responses[request.full_url]
        headers = email.message.Message()
        if location: headers['Location'] = location
        response = urllib.response.addinfourl(io.BytesIO(json.dumps(data).encode()), headers, request.full_url, code)
        response.msg = 'synthetic response'
        return response

    https_open = http_open


class ReadinessTransportTests(unittest.TestCase):
    def setUp(self):
        self.socket_patch = mock.patch.object(socket, 'create_connection', side_effect=AssertionError('No real sockets in CPU fixture'))
        self.socket_patch.start()

    def tearDown(self): self.socket_patch.stop()

    def backend(self, transport):
        opener = m.direct_opener(); opener.add_handler(transport)
        with mock.patch.object(m, 'OPENER', opener): return m.backend_health(TOKEN)

    def remote(self, transport):
        build = urllib.request.build_opener
        def mocked_build(*handlers): return build(*handlers, transport)
        output = io.StringIO()
        with mock.patch.object(urllib.request, 'build_opener', side_effect=mocked_build), \
             mock.patch('sys.stdin', io.StringIO(TOKEN)), contextlib.redirect_stdout(output):
            exec(compile(m.REMOTE, 'synthetic-remote-readiness', 'exec'), {})
        return json.loads(output.getvalue())

    def test_backend_redirect_never_sends_bearer_to_third_party(self):
        for code in (301,302,303,307,308):
            with self.subTest(code=code):
                transport = MemoryTransport({'http://127.0.0.1:1919/health':(code, {}, 'https://third-party.invalid/collect')})
                with self.assertRaises(urllib.error.HTTPError): self.backend(transport)
                self.assertEqual(len(transport.calls), 1)
                self.assertEqual(transport.calls[0]['host'], '127.0.0.1:1919')
                self.assertEqual(transport.calls[0]['authorization'], 'Bearer '+TOKEN)

    def test_remote_redirects_never_reach_third_party(self):
        responses = {url:(302, {}, 'https://third-party.invalid/collect') for url in
            ('http://127.0.0.1:1919/health', 'https://llm.example.com/v1/models')}
        transport = MemoryTransport(responses)
        result = self.remote(transport)
        self.assertFalse(result['tunnel']['ready']); self.assertFalse(result['https']['ready'])
        self.assertTrue(all(result[name]['error_type']=='HTTPError' for name in ('tunnel','https')))
        self.assertEqual({call['url'] for call in transport.calls}, set(responses))
        self.assertNotIn(TOKEN, json.dumps(result))

    def test_inherited_proxy_ignored_local_and_remote(self):
        proxies = {key:'http://third-party.invalid:8888' for key in
            ('HTTP_PROXY','http_proxy','HTTPS_PROXY','https_proxy','ALL_PROXY','all_proxy')}
        proxies.update(NO_PROXY='', no_proxy='')
        with mock.patch.dict(os.environ, proxies):
            local = MemoryTransport({'http://127.0.0.1:1919/health':(200, GOOD, None)})
            self.assertEqual(self.backend(local)['status'], 'ok')
            remote = MemoryTransport({'http://127.0.0.1:1919/health':(200, GOOD, None),
                'https://llm.example.com/v1/models':(200, {'data':[{'id':'qwen36-35b'}]}, None)})
            result = self.remote(remote)
            self.assertTrue(result['tunnel']['ready']); self.assertTrue(result['https']['ready'])
            for call in local.calls + remote.calls:
                self.assertFalse(call['proxy'])
                self.assertEqual(call['host'], urllib.parse.urlparse(call['url']).netloc)

    def test_tls_verification_and_hostname_check_preserved(self):
        opener = m.direct_opener()
        https = next(handler for handler in opener.handlers if isinstance(handler, urllib.request.HTTPSHandler))
        self.assertEqual(https._context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(https._context.check_hostname)

    def test_non_200_success_shape_is_not_readiness(self):
        local = MemoryTransport({'http://127.0.0.1:1919/health':(201, GOOD, None)})
        with self.assertRaises(ValueError): self.backend(local)
        remote = MemoryTransport({'http://127.0.0.1:1919/health':(201, GOOD, None),
            'https://llm.example.com/v1/models':(201, {'data':[{'id':'qwen36-35b'}]}, None)})
        result = self.remote(remote)
        self.assertFalse(result['tunnel']['ready']); self.assertFalse(result['https']['ready'])

    def test_loading_wrong_model_and_maintenance_are_unready(self):
        for change in ({'status':'loading'}, {'model':'other'}, {'maintenance':'paused'}):
            remote = MemoryTransport({'http://127.0.0.1:1919/health':(200, {**GOOD, **change}, None),
                'https://llm.example.com/v1/models':(200, {'data':[{'id':'qwen36-35b'}]}, None)})
            self.assertFalse(self.remote(remote)['tunnel']['ready'])

    def test_generation_mismatch_fails_complete_readiness(self):
        route = {'ready':True,'tunnel':{'ready':True,'instance_id':'other-generation'},'https':{'ready':True}}
        with mock.patch.object(m, 'load_key', return_value=TOKEN), mock.patch.object(m, 'backend_health', return_value=dict(GOOD)), \
             mock.patch.object(m, 'edge_health', return_value=route), mock.patch.object(m, 'served_profile', return_value={}):
            result = m.collect(True)
        self.assertFalse(result['ready'])
        self.assertEqual(result['route']['tunnel']['error_type'], 'generation_mismatch')

    def test_stats_request_is_direct_and_redirect_free(self):
        stats = {'instance_id':'generation-a', 'model':{'id':'qwen36-35b', 'ctx':131072}}
        proxies = {key:'http://third-party.invalid:8888' for key in ('HTTP_PROXY','http_proxy','ALL_PROXY','all_proxy')}
        proxies.update(NO_PROXY='', no_proxy='')
        with mock.patch.dict(os.environ, proxies), mock.patch.object(m.pathlib.Path, 'read_text',
                return_value=json.dumps({'profiles':{'main128':{'context':131072}}})):
            local = MemoryTransport({'http://127.0.0.1:1919/v1/stats':(200, stats, None)})
            opener = m.direct_opener(); opener.add_handler(local)
            with mock.patch.object(m, 'OPENER', opener):
                served = m.served_profile(TOKEN, 'main128')
            self.assertEqual(served, {'served_context':131072, 'profile_context':131072, 'stats_instance_id':'generation-a'})
            self.assertEqual([(c['host'], c['proxy'], c['authorization']) for c in local.calls], [('127.0.0.1:1919', False, 'Bearer '+TOKEN)])
            moved = MemoryTransport({'http://127.0.0.1:1919/v1/stats':(302, {}, 'https://third-party.invalid/collect')})
            opener = m.direct_opener(); opener.add_handler(moved)
            with mock.patch.object(m, 'OPENER', opener), self.assertRaises(urllib.error.HTTPError):
                m.served_profile(TOKEN, 'main128')
            self.assertEqual(len(moved.calls), 1)


if __name__ == '__main__': unittest.main(verbosity=2)
