import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError,HTTPError
from urllib.request import Request,urlopen
from continuum.storage import Store
from continuum.relations import words,learn_words,related
from continuum.engine import chat,LocalModelError,ollama_models
from continuum.server import AppServer

class StoreTest(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory()
        self.db=Path(self.t.name)/'test.sqlite3'
        self.s=Store(self.db)
        self.a=self.s.create_project('A')
        self.b=self.s.create_project('B')
        self.sa=self.s.create_session(self.a['id'])
    def tearDown(self):self.t.cleanup()
    def test_project_isolation(self):
        self.assertEqual([p['name'] for p in self.s.list_projects()],['A','B'])
        self.assertEqual(self.s.list_sessions(self.b['id']),[])
        with self.assertRaises(ValueError):self.s.list_sessions('nonexistent')
    def test_message_persistence(self):
        self.s.add_message(self.sa['id'],'user','ABCA CABA')
        other=Store(self.db)
        self.assertEqual(other.list_messages(self.sa['id'])[0]['content'],'ABCA CABA')
    def test_titles_autogenerate(self):
        self.s.add_message(self.sa['id'],'user','Investigate our keyboard invention')
        self.assertIn('keyboard invention',self.s.list_sessions(self.a['id'])[0]['title'])
    def test_message_length(self):
        with self.assertRaises(ValueError):self.s.add_message(self.sa['id'],'user','')
        with self.assertRaises(ValueError):self.s.add_message(self.sa['id'],'system','secret')
    def test_notes_and_evidence(self):
        self.s.add_note(self.a['id'],'Idea','Try new rule','hypothesis')
        self.assertEqual(self.s.list_notes(self.a['id'])[0]['kind'],'hypothesis')
        self.assertEqual(self.s.list_notes(self.b['id']),[])
        with self.assertRaises(ValueError):self.s.add_note(self.a['id'],'Fact','Works','evidence')
        self.s.add_note(self.a['id'],'Fact','Works','evidence','Unit test #7')
    def test_export_import_isolated(self):
        self.s.add_message(self.sa['id'],'user','ABCA CABA')
        self.s.add_note(self.a['id'],'Lesson','Length reveals relation')
        doc=self.s.export_project(self.a['id'])
        imported=self.s.import_project(json.loads(json.dumps(doc)))
        self.assertEqual(len(self.s.list_sessions(imported['id'])),1)
        self.assertEqual(len(self.s.list_notes(imported['id'])),1)
        self.assertEqual(len(self.s.list_messages(self.s.list_sessions(imported['id'])[0]['id'])),1)
        self.assertEqual(len(self.s.list_notes(self.b['id'])),0)
    def test_handoff_preserves_notes_and_messages(self):
        self.s.add_note(self.a['id'],'Independent symbol principle','A finite alphabet yields unbounded sequences.','lesson')
        self.s.add_message(self.sa['id'],'user','ABCA CABA four letters')
        text=self.s.handoff(self.a['id'])
        self.assertIn('Independent symbol principle',text)
        self.assertIn('ABCA CABA',text)
        self.assertNotIn('Project: B',text)
    def test_bad_import(self):
        with self.assertRaises(ValueError):self.s.import_project({'format':'bad'})
    def test_backup_integrity(self):
        self.s.add_message(self.sa['id'],'user','Durable words')
        file=Path(self.t.name)/'backups'/'save.sqlite3'
        self.s.backup(file)
        self.assertEqual(Store(file).list_messages(self.sa['id'])[0]['content'],'Durable words')
        with self.assertRaises(ValueError):self.s.backup(self.db)
    def test_word_count_rule(self):
        self.assertEqual(words('ABCA caba FOUR'),['ABCA','CABA','FOUR'])
        self.s.add_message(self.sa['id'],'user','ABCA CABA'); rs=related(self.s,self.a['id'],'ABCA')
        self.assertEqual(rs[0]['word'],'CABA')
        self.assertTrue(any('same letter counts' in s for s in rs[0]['reasons']))
    def test_word_isolation(self):
        self.s.add_message(self.sa['id'],'user','ABCA CABA')
        self.assertEqual(related(self.s,self.b['id'],'ABCA'),[])
    def test_relation_repeated_cooccurrence(self):
        self.s.add_message(self.sa['id'],'user','APPLE ORANGE')
        self.s.add_message(self.sa['id'],'user','APPLE ORANGE')
        results=related(self.s,self.a['id'],'APPLE')
        self.assertEqual(results[0]['word'],'ORANGE')
        self.assertIn('2 message(s)',results[0]['reasons'][-1])
    def test_invalid_word(self):
        with self.assertRaises(ValueError):related(self.s,self.a['id'],'A')
    def test_seed(self):
        s=Store(Path(self.t.name)/'seed.sqlite3')
        self.assertTrue(s.seed_if_empty())
        self.assertFalse(s.seed_if_empty())
        self.assertEqual(len(s.list_projects()),1)
        self.assertTrue(len(s.list_notes(s.list_projects()[0]['id']))>=5)
    def test_no_secret_leak_in_error(self):
        with patch('continuum.engine.urlopen',side_effect=URLError('no local server')):
            with self.assertRaises(LocalModelError):chat(self.s,self.sa['id'],'Research ABCA',timeout=1)
        self.assertEqual(len(self.s.list_messages(self.sa['id'])),1)
    def test_mock_local_inference(self):
        class Response:
            def __enter__(self):return self
            def __exit__(self,*a):return False
            def read(self,*args):return json.dumps({'message':{'content':'A testable relationship.'}}).encode()
        with patch('continuum.engine.urlopen',return_value=Response()):
            m=chat(self.s,self.sa['id'],'ABCA CABA',model='my-model:1',timeout=1)
        self.assertEqual(m['role'],'assistant');self.assertEqual(m['model'],'my-model:1')
        self.assertEqual(len(self.s.list_messages(self.sa['id'])),2)
    def test_model_name_validation(self):
        with self.assertRaises(ValueError):chat(self.s,self.sa['id'],'hi',model='https://evil')
        self.assertEqual(self.s.list_messages(self.sa['id']),[])

class HTTPTest(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory()
        self.http=AppServer('127.0.0.1',0,Path(self.t.name)/'api.sqlite3')
        self.thread=threading.Thread(target=self.http.serve_forever,daemon=True);self.thread.start()
        self.base=f'http://127.0.0.1:{self.http.server_port}'
    def tearDown(self):
        self.http.shutdown();self.http.server_close();self.thread.join(2);self.t.cleanup()
    def get(self,path):
        with urlopen(self.base+path,timeout=4) as resp:return resp.status,resp.read()
    def post(self,path,body,headers=None):
        req=Request(self.base+path,data=json.dumps(body).encode(),method='POST',headers={'Content-Type':'application/json',**(headers or {})})
        with urlopen(req,timeout=4) as resp:return resp.status,json.load(resp)
    def test_health_and_static(self):
        with patch('continuum.server.ollama_models',return_value=[]):
            status,data=self.get('/api/health')
        self.assertEqual(status,200)
        self.assertIn('teacherlab:local',json.loads(data)['models'])
        self.assertIn(b'AxiomFlow',self.get('/')[1])
    def test_api_chat_project_flow(self):
        p=json.loads(self.get('/api/projects')[1])[0]
        _,s=self.post('/api/sessions',{'project_id':p['id'],'title':'Working session'})
        self.assertEqual(s['title'],'Working session')
        _,note=self.post('/api/notes',{'project_id':p['id'],'title':'Fact','body':'Test','kind':'lesson'})
        self.assertEqual(note['kind'],'lesson')
        self.assertEqual(len(json.loads(self.get('/api/notes?project_id='+p['id'])[1])),7)
    def test_cross_origin_block(self):
        with self.assertRaises(HTTPError) as ctx:self.post('/api/projects',{'name':'bad'}, {'Origin':'https://other-site.example'})
        self.assertEqual(ctx.exception.code,403)
    def test_bad_host_block(self):
        with self.assertRaises(HTTPError) as ctx:
            req=Request(self.base+'/api/projects',headers={'Host':'evil.example'})
            urlopen(req,timeout=4)
        self.assertEqual(ctx.exception.code,403)
    def test_bad_query_status(self):
        with self.assertRaises(HTTPError) as ctx:self.get('/api/relations?project_id=no&word=A')
        self.assertEqual(ctx.exception.code,400)

if __name__=='__main__':unittest.main()
