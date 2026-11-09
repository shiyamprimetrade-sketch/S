"""S UI + native offline learning integration tests."""
from __future__ import annotations
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from unittest.mock import patch
from continuum.storage import Store
from continuum.engine import chat,S_RESEARCH_MODEL,LocalModelError,ollama_models
from continuum.s_reasoning import respond
from continuum.server import AppServer

ROOT=Path(__file__).resolve().parent.parent

class SStorageTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory()
        self.store=Store(Path(self.t.name)/'S.db')
        self.project=self.store.create_project('S')
        self.session=self.store.create_session(self.project['id'],'S original chat')
    def tearDown(self):self.t.cleanup()
    def test_rule_research_no_external_model(self):
        with patch('continuum.engine.urlopen',side_effect=AssertionError('network must not be used')):
            a=chat(self.store,self.session['id'],'Invent a way for symbols to learn meaning',model=S_RESEARCH_MODEL)
        self.assertEqual(a['model'],'s-research:local')
        self.assertIn('Status:',a['content'])
        self.assertIn('rule-based',a['content'])
        self.assertEqual(len(self.store.list_messages(self.session['id'])),2)
    def test_cloud_model_blocked_before_data_saved(self):
        with patch('continuum.engine.urlopen', side_effect=AssertionError('network request must not happen')):
            with self.assertRaises(LocalModelError):
                chat(self.store,self.session['id'],'private research data',model='example:cloud')
        self.assertEqual(self.store.list_messages(self.session['id']),[])
    def test_cloud_models_filtered_from_local_choices(self):
        from io import BytesIO
        fake=BytesIO(json.dumps({'models':[{'name':'test:cloud'},{'name':'local:3b'},{'name':'example-cloud'}]}).encode())
        with patch('continuum.engine.urlopen',return_value=fake):
            self.assertEqual(ollama_models(),['local:3b'])
    def test_language_interpretation_without_deep_trigger(self):
        a=respond(self.store,self.project['id'],'Why do computer languages use symbols?')
        self.assertIn('Separate symbols',a)
    def test_relevant_teacher_notes_are_used(self):
        self.store.add_note(self.project['id'],'Keyboard relation','Words and alphabet symbols connect through observed relationships','lesson')
        text=respond(self.store,self.project['id'],'Explore alphabet symbols')
        self.assertIn('Keyboard relation',text)
    def test_feedback_persistence_and_isolation(self):
        m=self.store.add_message(self.session['id'],'assistant','A testable idea',model=S_RESEARCH_MODEL)
        self.store.set_feedback(m['id'],1)
        self.assertEqual(Store(self.store.path).list_feedback(self.project['id'])[0]['rating'],1)
        other=self.store.create_project('Other')
        self.assertEqual(self.store.list_feedback(other['id']),[])
        self.store.set_feedback(m['id'],-1,'Needs proof')
        self.assertEqual(self.store.list_feedback(self.project['id'])[0]['note'],'Needs proof')
    def test_feedback_cannot_attach_to_user_message(self):
        u=self.store.add_message(self.session['id'],'user','Prompt')
        with self.assertRaises(ValueError):self.store.set_feedback(u['id'],1)
    def test_delete_is_real_and_cascades_feedback(self):
        m=self.store.add_message(self.session['id'],'assistant','Draft')
        self.store.set_feedback(m['id'],1)
        result=self.store.delete_session(self.session['id'])
        self.assertTrue(result['deleted'])
        self.assertEqual(self.store.list_sessions(self.project['id']),[])
        self.assertEqual(self.store.list_feedback(self.project['id']),[])
    def test_invalid_rating_rejected(self):
        m=self.store.add_message(self.session['id'],'assistant','Draft')
        with self.assertRaises(ValueError):self.store.set_feedback(m['id'],0)
    def test_s_source_static_offline_assets(self):
        html=(ROOT/'web/index.html').read_text(encoding='utf8')
        js=(ROOT/'web/app.js').read_text(encoding='utf8')
        self.assertIn('What shall we invent?',html)
        self.assertIn('S Researcher',html)
        self.assertNotIn('cdn.tailwindcss.com',html)
        self.assertNotIn('unpkg.com',html)
        self.assertNotIn('simulateAiResponse',js)
        self.assertIn("/api/chat",js)
        self.assertIn("/api/feedback",js)
        self.assertIn("/api/import-chatgpt",js)

class SHTTPTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory()
        self.srv=AppServer('127.0.0.1',0,Path(self.t.name)/'s.sqlite3')
        self.thread=threading.Thread(target=self.srv.serve_forever,daemon=True)
        self.thread.start()
        self.base=f'http://127.0.0.1:{self.srv.server_port}'
    def tearDown(self):
        self.srv.shutdown();self.srv.server_close();self.thread.join(3);self.t.cleanup()
    def get(self,path):
        with urlopen(self.base+path,timeout=10) as r:return r.status,r.read(),dict(r.headers)
    def post(self,path,body):
        req=Request(self.base+path,data=json.dumps(body).encode(),method='POST',headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=10) as r:return json.loads(r.read())
    def test_health_reports_actual_engines(self):
        h=json.loads(self.get('/api/health')[1]);self.assertEqual(h['app'],'S')
        self.assertIn(S_RESEARCH_MODEL,h['models'])
        self.assertEqual(h['default_model'],S_RESEARCH_MODEL)
        self.assertFalse(h['cloud_api'])
    def test_real_browser_to_s_database_chat(self):
        projects=json.loads(self.get('/api/projects')[1]);p=projects[0]
        s=self.post('/api/sessions',{'project_id':p['id'],'title':'Testing'})
        output=self.post('/api/chat',{'session_id':s['id'],'model':S_RESEARCH_MODEL,'text':'Invent a language learning test'})
        self.assertIn('Proposed next operation',output['content'])
        history=json.loads(self.get('/api/messages?session_id='+s['id'])[1]);self.assertEqual(len(history),2)
        self.assertEqual(history[1]['id'],output['id'])
        self.post('/api/feedback',{'message_id':output['id'],'rating':1})
        fb=json.loads(self.get('/api/feedback?project_id='+p['id'])[1]);self.assertEqual(fb[0]['rating'],1)
        self.post('/api/delete-session',{'session_id':s['id']})
        self.assertFalse(json.loads(self.get('/api/feedback?project_id='+p['id'])[1]))
    def test_static_css_js_and_local_csp(self):
        for url in ['/','/app.js','/style.css']:
            code,data,headers=self.get(url)
            self.assertEqual(code,200)
            self.assertGreater(len(data),500)
            self.assertIn("script-src 'self'",headers['Content-Security-Policy'])
    def test_disallows_remote_origin_and_host(self):
        p=json.loads(self.get('/api/projects')[1])[0]
        req=Request(self.base+'/api/sessions',data=json.dumps({'project_id':p['id']}).encode(),method='POST',headers={'Content-Type':'application/json','Origin':'https://bad.example'})
        with self.assertRaises(HTTPError) as e:urlopen(req,timeout=10)
        self.assertEqual(e.exception.code,403)

if __name__=='__main__':unittest.main()

class SRoundTripTest(unittest.TestCase):
    def test_export_preserves_feedback_and_archive_placeholders(self):
        with tempfile.TemporaryDirectory() as t:
            s=Store(Path(t)/'db.sqlite3')
            p=s.create_project('S')
            ses=s.create_session(p['id'])
            user=s.add_message(ses['id'],'user','APPLE -> FIVE')
            assistant=s.add_message(ses['id'],'assistant','Test the pattern',model='s-research:local')
            s.set_feedback(assistant['id'],-1,'Need a second test')
            with s.connect() as db:
                db.execute('INSERT INTO archive_extras VALUES(?,?,?,?,?,?,?)',(
                    'extra1',ses['id'],'tool','[ARCHIVED MEDIA]',user['created_at'],'original-tool',user['created_at']))
            doc=s.export_project(p['id'])
            self.assertEqual(len(doc['feedback']),1)
            self.assertEqual(doc['sessions'][0]['archive_extras'][0]['content'],'[ARCHIVED MEDIA]')
            duplicate=s.import_project(doc)
            self.assertEqual(len(s.list_feedback(duplicate['id'])),1)
            recovered=s.export_project(duplicate['id'])
            self.assertEqual(recovered['sessions'][0]['archive_extras'][0]['content'],'[ARCHIVED MEDIA]')
            self.assertEqual(recovered['feedback'][0]['note'],'Need a second test')
