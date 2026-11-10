"""Run against the patched S v0.6 app, with its installed checkpoint."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from continuum.engine import chat
from continuum.server import AppServer
from s_intelligence import browser


class NativeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = AppServer('127.0.0.1', 0, Path(self.temp.name) / 'test.sqlite3')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)
        self.store = self.server.store
        project = self.store.create_project('SYNTHETIC native verification')
        self.session = self.store.create_session(project['id'], 'Synthetic test')['id']

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def test_advertised_only_when_checkpoint_available(self):
        with patch('continuum.server.ollama_models', return_value=[]):
            with urlopen(self.url + '/api/health') as response:
                self.assertIn(browser.MODEL_ID, json.load(response)['models'])
            with patch('s_intelligence.browser.CHECKPOINT', Path(self.temp.name) / 'missing.pt'):
                with urlopen(self.url + '/api/health') as response:
                    self.assertNotIn(browser.MODEL_ID, json.load(response)['models'])

    def test_native_route_offline_persistence_and_determinism(self):
        with patch('socket.socket.connect', side_effect=AssertionError('No network')):
            first = chat(self.store, self.session, '2 + 2 =', browser.MODEL_ID)
            second = chat(self.store, self.session, '2 + 2 =', browser.MODEL_ID)
        self.assertEqual(first['content'], second['content'])
        self.assertTrue(first['content'].startswith(browser.NOTICE))
        self.assertEqual(first['model'], browser.MODEL_ID)
        self.assertEqual(len(self.store.list_messages(self.session)), 4)

    def test_http_chat_and_unicode_failure(self):
        def request(text):
            body = json.dumps({'session_id': self.session, 'text': text, 'model': browser.MODEL_ID}).encode()
            return urlopen(Request(self.url + '/api/chat', data=body, headers={'Content-Type': 'application/json'}))
        with patch('continuum.engine.urlopen', side_effect=AssertionError('Ollama forbidden')):
            with request('Copy: blue') as response:
                self.assertEqual(json.load(response)['model'], browser.MODEL_ID)
            with self.assertRaises(HTTPError) as caught:
                request('বাংলা')
            self.assertEqual(caught.exception.code, 503)
            self.assertIn('ASCII', json.load(caught.exception)['error'])
