"""S v0.7: real retrieval tests, migration, persistence, privacy and HTTP."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from continuum.apprenticeship import Apprenticeship
from continuum.meaning import TeacherMemory
from continuum.server import AppServer
from continuum.storage import Store


class BenchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.db=Path(self.temp.name)/'s.db'
        self.store=Store(self.db)
        self.p=self.store.create_project('S Engineering')['id']
        self.q=self.store.create_project('Other Project')['id']
        self.memory=TeacherMemory(self.store)
        self.lab=Apprenticeship(self.store)

    def tearDown(self):self.temp.cleanup()

    def teach(self):
        lesson=self.memory.teach(self.p,'Investigate assumptions',
            'Look for evidence and compare alternative hypotheses before deciding.',
            ['Use rigorous analysis when solving inventions'])
        return lesson

    def test_aliases_improve_measured_retrieval(self):
        lesson=self.teach()
        self.lab.add_case(self.p,'Apply careful investigation to a new design',lesson['id'])
        self.lab.add_case(self.p,'Please bake cinnamon rolls',None)
        initial=self.lab.run(self.p)
        self.assertEqual(initial['metrics']['baseline_correct'],1)
        self.assertEqual(initial['metrics']['taught_meaning_correct'],1)
        self.memory.define_concept(self.p,'thinking effort',
                                   ['rigorous analysis','careful investigation'])
        later=self.lab.run(self.p)
        self.assertEqual(later['metrics']['baseline_correct'],1)
        self.assertEqual(later['metrics']['taught_meaning_correct'],2)
        self.assertEqual(later['metrics']['evaluated'],2)
        self.assertNotEqual(initial['memory_sha256'],later['memory_sha256'])
        self.assertEqual(initial['dataset_sha256'],later['dataset_sha256'])
        self.assertEqual(self.lab.cases(self.p)[0]['question'],'Apply careful investigation to a new design')
        self.assertEqual(self.memory.lessons(self.p,'approved')[0]['cues'],
                         ['Use rigorous analysis when solving inventions'])

    def test_duplicate_and_verbatim_leakage_rejected(self):
        lesson=self.teach()
        with self.assertRaisesRegex(ValueError,'blind test'):
            self.lab.add_case(self.p,'Use rigorous analysis when solving inventions',lesson['id'])
        self.lab.add_case(self.p,'Apply careful investigation to a design',lesson['id'])
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            self.lab.add_case(self.p,'apply careful investigation to a design',None)
        with self.assertRaises(ValueError):self.lab.add_case(self.p,'  ',lesson['id'])
        with self.assertRaises(ValueError):self.lab.add_case(self.p,'Does it work?',self.q)

    def test_no_test_cases_cannot_claim_score(self):
        with self.assertRaisesRegex(ValueError,'at least one'):
            self.lab.run(self.p)
        self.assertEqual(self.lab.runs(self.p),[])

    def test_stale_target_after_revision_not_counted(self):
        lesson=self.teach()
        self.lab.add_case(self.p,'Apply careful investigation to a design',lesson['id'])
        self.memory.revise(self.p,lesson['id'],'Test assumptions before deciding',
                           ['Use thoughtful inquiry for experiments'])
        result=self.lab.run(self.p)
        self.assertEqual(result['metrics']['evaluated'],0)
        self.assertEqual(result['metrics']['stale_excluded'],1)
        self.assertIsNone(result['metrics']['baseline_accuracy'])
        self.assertEqual(result['results'][0]['status'],'stale_expected_lesson')

    def test_project_separation_and_reopen(self):
        lesson=self.teach()
        case=self.lab.add_case(self.p,'Careful investigation with designs',lesson['id'])
        self.assertFalse(Apprenticeship(Store(self.db)).cases(self.q))
        self.assertEqual(Apprenticeship(Store(self.db)).cases(self.p)[0]['id'],case['id'])
        with self.assertRaisesRegex(ValueError,'not found'):
            self.lab.delete_case(self.q,case['id'])
        self.lab.delete_case(self.p,case['id'])
        self.assertFalse(self.lab.cases(self.p))

    def test_export_import_historical_evidence_and_new_run(self):
        lesson=self.teach()
        self.memory.define_concept(self.p,'thinking',['rigorous analysis','careful investigation'])
        self.lab.add_case(self.p,'Apply careful investigation to designs',lesson['id'])
        old=self.lab.run(self.p)
        export=self.store.export_project(self.p)
        imported=self.store.import_project(export)
        new_pid=imported['id']
        new_lab=Apprenticeship(self.store)
        cases=new_lab.cases(new_pid)
        self.assertEqual(len(cases),1)
        self.assertNotEqual(cases[0]['expected_lesson_id'],lesson['id'])
        self.assertEqual(len(new_lab.runs(new_pid)),1)
        self.assertEqual(new_lab.runs(new_pid)[0]['dataset_sha256'],old['dataset_sha256'])
        rerun=new_lab.run(new_pid)
        self.assertEqual(rerun['metrics']['taught_meaning_correct'],1)
        self.assertEqual(rerun['metrics']['stale_excluded'],0)
        self.assertEqual(new_lab.cases(self.p)[0]['expected_lesson_id'],lesson['id'])

    def test_backup_copies_private_evidence(self):
        lesson=self.teach()
        self.lab.add_case(self.p,'Apply careful investigation to a design',lesson['id'])
        self.lab.run(self.p)
        path=self.store.backup(Path(self.temp.name)/'backup.sqlite3')
        b=Apprenticeship(Store(path))
        self.assertEqual(len(b.cases(self.p)),1)
        self.assertEqual(len(b.runs(self.p)),1)
        handoff=self.store.handoff(self.p)
        self.assertIn('Latest run:',handoff)
        self.assertIn('retrieval only',handoff)

    def test_readiness_counts_cannot_claim_complete_private_export(self):
        before=self.lab.readiness(self.p)
        self.assertEqual(before['messages'],0)
        self.assertEqual(before['actual_private_export_imported'],'not_verifiable_from_counts')
        session=self.store.create_session(self.p,'Teacher lesson')
        self.store.add_message(session['id'],'user','I want S to improve')
        after=self.lab.readiness(self.p)
        self.assertEqual(after['messages'],1)
        self.assertEqual(after['conversations'],1)

    def test_imported_chat_correction_is_only_candidate(self):
        conversation=[{'id':'test-example','title':'Naming S', 'current_node':'third',
            'mapping':{
              'first':{'parent':None,'message':{'id':'msg1','author':{'role':'user'},'content':{'parts':['Why is S named S?']},'create_time':1700000000}},
              'second':{'parent':'first','message':{'id':'msg2','author':{'role':'assistant'},'content':{'parts':['Because it is short.']},'create_time':1700000001}},
              'third':{'parent':'second','message':{'id':'msg3','author':{'role':'user'},'content':{'parts':['No, adding S to words can transform their meaning.']},'create_time':1700000002}},
            }}]
        imported=self.store.import_chatgpt(conversation)
        pid=imported['project_id']
        self.assertEqual(imported['chat_messages'],3)
        mem=TeacherMemory(self.store)
        self.assertEqual(mem.suggest_archive_corrections(pid)['candidates_added'],1)
        self.assertFalse(mem.recall(pid,'Why is S named S?'))
        lesson=mem.lessons(pid,'candidate')[0]
        mem.review(pid,lesson['id'],'approved')
        self.assertEqual(mem.recall(pid,'Why is S named S?')[0]['id'],lesson['id'])
        self.assertFalse(mem.recall(self.p,'Why is S named S?'))
        self.assertEqual(Apprenticeship(self.store).readiness(pid)['messages'],3)


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.server=AppServer('127.0.0.1',0,Path(cls.temp.name)/'s.db')
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.host=f'http://127.0.0.1:{cls.server.server_port}'
        cls.pid=cls.server.store.list_projects()[0]['id']

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join(timeout=3);cls.temp.cleanup()

    def call(self,path,body=None):
        kw={'headers':{'Host':f'127.0.0.1:{self.server.server_port}'}}
        if body is not None:
            kw['data']=json.dumps(body).encode('utf8')
            kw['headers']['Content-Type']='application/json'
            kw['method']='POST'
        with urlopen(Request(self.host+path,**kw),timeout=5) as response:
            return json.load(response)

    def test_real_http_workflow_and_permissions(self):
        pid=self.pid
        lesson=self.call('/api/memory/teach',{'project_id':pid,'title':'Reason carefully',
                           'principle':'Check evidence before claims','cues':['Use rigorous analysis when investigating']})
        self.call('/api/memory/concept',{'project_id':pid,'label':'inquiry',
                                        'variants':['rigorous analysis','careful investigation']})
        case=self.call('/api/apprenticeship/cases',{'project_id':pid,
                          'question':'Apply careful investigation to prototypes',
                          'expected_lesson_id':lesson['id']})
        result=self.call('/api/apprenticeship/run',{'project_id':pid})
        self.assertEqual(result['metrics']['taught_meaning_correct'],1)
        self.assertEqual(result['metrics']['baseline_correct'],0)
        self.assertIn(case['id'],[x['id'] for x in self.call('/api/apprenticeship/cases?project_id='+pid)])
        self.assertEqual(len(self.call('/api/apprenticeship/runs?project_id='+pid)),1)
        self.assertEqual(self.call('/api/apprenticeship/readiness?project_id='+pid)['held_out_questions'],1)
        with self.assertRaises(HTTPError) as cm:
            self.call('/api/apprenticeship/cases',{'project_id':pid,'question':'x',
                                                    'expected_lesson_id':lesson['id']})
        self.assertEqual(cm.exception.code,400)
        self.assertTrue(self.call('/api/apprenticeship/delete',{'project_id':pid,'case_id':case['id']})['deleted'])

    def test_local_origin_still_enforced(self):
        req=Request(self.host+'/api/apprenticeship/run',
                    data=json.dumps({'project_id':self.pid}).encode(),method='POST',
                    headers={'Origin':'https://attacker.example','Content-Type':'application/json'})
        with self.assertRaises(HTTPError) as cm:urlopen(req,timeout=5)
        self.assertEqual(cm.exception.code,403)

if __name__=='__main__':unittest.main()
