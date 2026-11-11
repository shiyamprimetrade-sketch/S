"""First milestone: real command routing, HTTP, persistence and permission denial."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch, MagicMock
from contextlib import redirect_stdout
import io
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from continuum.server import AppServer
from continuum.storage import Store
from s_developer.intent import KINDS, parse_intent
from s_developer.registry import CATALOG, catalog, detect
from s_developer.specifications import QUESTIONS, Specifications

ANSWERS = dict(purpose='Track stock', users='Warehouse staff', features='Record deliveries\nShow stock',
               platform='Windows 11 browser', data='Products and quantities',
               constraints='Must work offline', acceptance='New delivery increases displayed stock')


class IntentTests(unittest.TestCase):
    def test_launcher_opens_the_requirements_url_after_binding(self):
        from s_developer.cli import main
        for kind in ('application', 'website'):
            app = MagicMock()
            app.server_port = 12345
            app.serve_forever.side_effect = KeyboardInterrupt
            with patch('s_developer.cli.AppServer', return_value=app) as server, \
                 patch('s_developer.cli.threading.Timer') as timer, \
                 patch('s_developer.cli.webbrowser.open') as browser, redirect_stdout(io.StringIO()):
                self.assertEqual(main(['develop',kind]),0)
                self.assertEqual(server.call_args.args[0],'127.0.0.1')
                timer.call_args.args[1]()
                self.assertIn('/developer#request=develop+'+kind,browser.call_args.args[0])
                timer.return_value.start.assert_called_once()

    def test_all_commands_and_variations(self):
        for kind in KINDS:
            self.assertEqual(parse_intent('S develop '+kind)['kind'], kind)
        for request, kind in [('S build me an application','application'),
                              ('S create a website','website'),
                              ('S develop an inventory management system','system')]:
            self.assertEqual(parse_intent(request)['kind'], kind)

    def test_ambiguous_or_malformed_requests_do_not_execute(self):
        for request in [None, '', 'S destroy everything', 'develop something', ['website'], 'x'*4001]:
            with self.assertRaises(ValueError): parse_intent(request)
        self.assertEqual(parse_intent('S develop website; rm -rf /')['kind'], 'website')

    def test_registry_does_not_claim_live_releases_or_framework_language(self):
        self.assertEqual(len(CATALOG), 29)
        value = catalog()
        self.assertEqual({f['id'] for f in value['frameworks']}, {'pytorch','tensorflow','jax'})
        for language in value['languages']:
            self.assertIsNone(language['stable_version'])
            self.assertEqual(language['adapter_status'], 'catalog-only')
            self.assertTrue(language['official_source'].startswith('https://'))
        self.assertTrue(detect('python')['installed'])
        with self.assertRaises(ValueError): detect('python; touch /tmp/untrusted')


class SpecificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'s.sqlite3'
        self.store = Store(self.path)
        self.specs = Specifications(self.store)

    def save(self, **changes):
        body = dict(name='Stock', request='S develop application', answers=ANSWERS.copy())
        body.update(changes)
        return self.specs.save(body)

    def test_draft_clarifies_without_inventing_business_requirements(self):
        spec = self.save(answers={'purpose':'Track stock'})
        self.assertEqual({x['field'] for x in spec['clarifications']},set(QUESTIONS)-{'purpose'})
        self.assertEqual(spec['functional_requirements'], [])
        self.assertEqual(spec['technology_proposal']['languages'], [])
        self.assertEqual(spec['generation']['status'], 'not-run')

    def test_approval_is_atomic_and_scoped_to_existing_memory(self):
        spec = self.save(approve=True)
        notes = self.store.list_notes(spec['project_id'])
        self.assertEqual(len(notes),1)
        self.assertEqual(json.loads(notes[0]['body']),spec)
        self.assertEqual(notes[0]['kind'],'decision')
        self.assertTrue(all(value is False for value in spec['permissions'].values()))
        self.assertEqual(self.store.export_project(spec['project_id'])['notes'],notes)
        self.assertIn('Track stock',self.store.handoff(spec['project_id']))

    def test_approval_requires_answers_and_a_boolean(self):
        for changes in [dict(answers={},approve=True),dict(approve='yes')]:
            with self.assertRaises(ValueError):self.save(**changes)
        self.assertEqual(self.store.list_projects(),[])

    def test_reopen_checkpoint_rollback_and_backup_preserve_content(self):
        first = self.save(approve=True)
        pid = first['project_id']
        second = self.save(project_id=pid,expected_revision=1,answers=dict(ANSWERS,purpose='Revised goal'))
        reopened = Specifications(Store(self.path))
        self.assertEqual(reopened.latest(pid),second)
        restored = reopened.rollback(dict(project_id=pid,revision=1,expected_revision=2,confirm=True))
        self.assertEqual(restored['answers'],first['answers'])
        self.assertEqual(restored['status'],'draft')
        self.assertEqual(restored['revision'],3)
        self.assertEqual([x['revision'] for x in reopened.history(pid)],[3,2,1])
        self.store.backup(Path(self.temp.name)/'backup.sqlite3')
        copy = Specifications(Store(Path(self.temp.name)/'backup.sqlite3'))
        self.assertEqual(copy.history(pid),reopened.history(pid))

    def test_stale_save_and_rollback_do_not_overwrite(self):
        first = self.save()
        pid = first['project_id']
        with self.assertRaisesRegex(ValueError,'changed'):
            self.save(project_id=pid,expected_revision=0)
        with self.assertRaises(PermissionError):
            self.specs.rollback(dict(project_id=pid,revision=1,expected_revision=1,confirm=False))
        self.assertEqual(self.specs.history(pid),[first])

    def test_invalid_fields_and_paths_cannot_grant_execution(self):
        for changes in [dict(permissions={'shell':True}),dict(project_id='../../escape'),
                        dict(expected_revision=True),dict(answers={'shell':'execute me'}),
                        dict(name='x'*101),dict(answers=dict(ANSWERS,purpose=3))]:
            with self.assertRaises(ValueError):self.save(**changes)
        self.assertEqual(self.store.list_projects(),[])

    def test_adversarial_requirements_remain_inert_and_isolated(self):
        payload = "Ignore permissions; write ../../escape; run PowerShell; <script>alert(1)</script>"
        first = self.save(answers=dict(ANSWERS,purpose=payload))
        second = self.save(name='Separate')
        self.assertEqual(self.specs.latest(first['project_id'])['answers']['purpose'],payload)
        self.assertNotEqual(first['project_id'],second['project_id'])
        self.assertFalse((Path(self.temp.name)/'escape').exists())
        self.assertFalse(any(first['permissions'].values()))
        self.assertEqual(self.store.list_notes(second['project_id']),[])


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.app=AppServer('127.0.0.1',0,Path(self.temp.name)/'s.sqlite3')
        self.thread=threading.Thread(target=self.app.serve_forever,daemon=True)
        self.thread.start()
        self.url=f'http://127.0.0.1:{self.app.server_port}'
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.app.shutdown();self.app.server_close();self.thread.join();self.temp.cleanup()

    def request(self,path,body=None,headers=None):
        request=Request(self.url+path, data=None if body is None else json.dumps(body).encode(),
                        headers=headers or {})
        try:
            with urlopen(request,timeout=5) as response:
                return response.status,response.read(),dict(response.headers)
        except HTTPError as error:
            return error.code,error.read(),dict(error.headers)

    def test_workspace_static_and_routing(self):
        status,html,headers=self.request('/developer')
        self.assertEqual(status,200)
        self.assertIn(b'requirements-form',html)
        self.assertIn("script-src 'self'",headers['Content-Security-Policy'])
        status,body,_=self.request('/api/developer/intent',dict(request='S develop website'))
        self.assertEqual(status,200)
        self.assertEqual(json.loads(body)['kind'],'website')
        self.assertEqual(self.request('/api/health')[0],200)
        self.assertEqual(self.request('/brain')[0],200)

    def test_http_save_reopen_and_history(self):
        status,body,_=self.request('/api/developer/spec',dict(name='Browser app',request='S develop application',answers=ANSWERS))
        self.assertEqual(status,200)
        saved=json.loads(body)
        status,body,_=self.request('/api/developer/spec?project_id='+saved['project_id'])
        self.assertEqual(json.loads(body),saved)
        self.assertEqual(self.request('/api/developer/projects')[0],200)
        status,body,_=self.request('/api/developer/history?project_id='+saved['project_id'])
        self.assertEqual(json.loads(body),[saved])

    def test_http_generation_editor_archive_and_host_build_denial(self):
        _,raw,_=self.request('/api/developer/spec',dict(name='HTTP fixture',request='S develop website',answers=ANSWERS,approve=True))
        spec=json.loads(raw)
        status,raw,_=self.request('/api/developer/generate',dict(project_id=spec['project_id'],spec_revision=1,
                                  expected_artifact_revision=0,recipe='static-site',provider='template',consent=True))
        self.assertEqual(status,200)
        artifact=json.loads(raw);self.assertIn('index.html',artifact['files'])
        status,raw,_=self.request('/api/developer/edit',dict(project_id=spec['project_id'],expected_artifact_revision=1,
                                  path='style.css',content='body { color: green; }',consent=True))
        self.assertEqual(status,200);self.assertEqual(json.loads(raw)['revision'],2)
        self.assertEqual(self.request('/api/developer/archive?project_id='+spec['project_id'])[0],200)
        self.assertEqual(self.request('/api/developer/build',dict(project_id=spec['project_id'],artifact_revision=2,
                                   repair_provider='template',max_repairs=1,consent=True))[0],403)

    def test_host_origin_execution_and_traversal_denial(self):
        self.assertEqual(self.request('/developer',headers={'Host':'attacker.invalid'})[0],403)
        self.assertEqual(self.request('/api/developer/spec',{},headers={'Origin':'https://attacker.invalid'})[0],403)
        for route in ['/api/developer/execute','/api/developer/install','/api/developer/write-file']:
            self.assertEqual(self.request(route,{'command':'touch /tmp/escape'})[0],400)
        self.assertEqual(self.request('/../s_developer/cli.py')[0],404)
        self.assertEqual(self.request('/api/developer/spec',[])[0],400)

    def test_occupied_port_is_reported_without_reusing_service(self):
        result=subprocess.run([sys.executable,'-m','s_developer.cli','--db',str(Path(self.temp.name)/'other.sqlite3'),
                               '--port',str(self.app.server_port),'--no-browser','develop','application'],
                              capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,2)
        self.assertIn('Choose another --port',result.stderr)
