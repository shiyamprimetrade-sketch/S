"""End-to-end teacher memory tests, including false positives and negative controls."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from continuum.meaning import TeacherMemory
from continuum.s_reasoning import respond
from continuum.server import AppServer
from continuum.storage import Store


class TeacherMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 's.sqlite3'
        self.store = Store(self.path)
        self.one = self.store.create_project('S Computer')
        self.two = self.store.create_project('Unrelated business')
        self.pid = self.one['id']
        self.memory = TeacherMemory(self.store)

    def tearDown(self):
        self.temp.cleanup()

    def teach(self):
        return self.memory.teach(self.pid, 'Constraints are hypotheses',
                                 'Investigate alternative implementations before accepting a design constraint.',
                                 ['How can we overcome limitations?', 'Are there technical constraints?'])

    def test_manual_teaching_is_retrievable(self):
        lesson = self.teach()
        result = self.memory.recall(self.pid, 'Are there technical constraints?')
        self.assertEqual(result[0]['id'], lesson['id'])
        self.assertEqual(result[0]['origin'], 'teacher')
        self.assertFalse(result[0]['is_verified_fact'])

    def test_teacher_defined_equivalences_generalize_to_new_query(self):
        self.teach()
        self.assertFalse(self.memory.recall(self.pid, 'How can we handle obstacles?'))
        self.memory.define_concept(self.pid, 'limitations', ['limitations', 'constraints', 'obstacles', 'barriers'])
        matches = self.memory.recall(self.pid, 'How can we handle obstacles?')
        self.assertEqual(len(matches), 1)
        self.assertIn('@limitations', matches[0]['matched_terms'])

    def test_no_unrelated_recall(self):
        self.teach()
        self.assertEqual(self.memory.recall(self.pid, 'Bake chocolate cookies'), [])

    def test_separate_projects_do_not_share_lessons_or_aliases(self):
        self.teach()
        self.memory.define_concept(self.pid, 'limitations', ['limitations', 'barriers'])
        self.assertFalse(self.memory.recall(self.two['id'], 'limitations'))
        self.assertFalse(self.memory.concepts(self.two['id']))
        with self.assertRaises(ValueError):
            self.memory.review(self.two['id'], self.memory.lessons(self.pid)[0]['id'], 'approved')

    def test_candidate_never_used_until_explicit_approval(self):
        candidate = self.memory.teach(self.pid, 'Unreviewed correction',
             'Do not mistake a renamed failure for success.', ['What is success?'], status='candidate')
        self.assertFalse(self.memory.recall(self.pid, 'What is success?'))
        self.memory.review(self.pid, candidate['id'], 'approved')
        self.assertEqual(self.memory.recall(self.pid, 'What is success?')[0]['id'], candidate['id'])

    def test_rejected_candidate_stays_out(self):
        candidate = self.memory.teach(self.pid,'Incorrect lesson','An untested claim',
                                       ['How to test a claim?'],status='candidate')
        self.memory.review(self.pid,candidate['id'],'rejected')
        self.assertFalse(self.memory.recall(self.pid,'How to test a claim?'))
        with self.assertRaises(ValueError):
            self.memory.review(self.pid,candidate['id'],'approved')

    def test_revision_replaces_old_lesson_and_preserves_version(self):
        old = self.teach()
        new = self.memory.revise(self.pid, old['id'],
               'Explore alternatives and measure outcomes before rejecting a proposal.',
               ['What are technical limitations?'])
        self.assertEqual(new['version'],2)
        self.assertEqual(new['supersedes_id'],old['id'])
        self.assertEqual(self.memory.lessons(self.pid)[1]['status'],'superseded')
        self.assertEqual(self.memory.recall(self.pid,'technical limitations')[0]['id'],new['id'])
        self.assertNotIn(old['id'],[x['id'] for x in self.memory.recall(self.pid,'technical limitations')])

    def test_persistence_across_database_reopen(self):
        self.teach()
        self.memory.define_concept(self.pid,'constraints',['constraints','obstacles'])
        other = TeacherMemory(Store(self.path))
        self.assertEqual(len(other.recall(self.pid,'How to overcome obstacles?')),1)
        self.assertEqual(len(other.concepts(self.pid)),1)

    def test_archived_correction_is_a_candidate_only(self):
        session=self.store.create_session(self.pid,'Name S teaching conversation')
        self.store.add_message(session['id'],'user','Why name it S?')
        self.store.add_message(session['id'],'assistant','Because it is short.')
        self.store.add_message(session['id'],'user',
                               'No, I chose S because a letter can change the meaning of existing words.')
        first=self.memory.suggest_archive_corrections(self.pid)
        self.assertEqual(first['candidates_added'],1)
        second=self.memory.suggest_archive_corrections(self.pid)
        self.assertEqual(second['candidates_added'],0)
        self.assertEqual(len(self.memory.lessons(self.pid,'candidate')),1)
        self.assertEqual(self.memory.recall(self.pid,'Why name S?'),[])
        cid=self.memory.lessons(self.pid,'candidate')[0]['id']
        self.memory.review(self.pid,cid,'approved')
        self.assertTrue(self.memory.recall(self.pid,'Why name S?'))

    def test_false_correction_marker_does_not_get_auto_approved(self):
        ses=self.store.create_session(self.pid,'Another session')
        self.store.add_message(ses['id'],'user','What does the software do?')
        self.store.add_message(ses['id'],'assistant','Runs tests')
        self.store.add_message(ses['id'],'user','Actually, this might need more investigation')
        self.memory.suggest_archive_corrections(self.pid)
        self.assertEqual(self.memory.lessons(self.pid,'approved'),[])

    def test_archive_source_is_validated_inside_project(self):
        ses=self.store.create_session(self.two['id'],'Private project')
        source=self.store.add_message(ses['id'],'user','Do not mix private projects')
        with self.assertRaisesRegex(ValueError,'not found'):
            self.memory.teach(self.pid,'Wrong provenance','Must not leak',
                              ['Separate projects'],origin='archive',source_id=source['id'])

    def test_negative_feedback_generates_unapproved_correction(self):
        session=self.store.create_session(self.pid,'Error correction')
        self.store.add_message(session['id'],'user','What is accuracy?')
        answer=self.store.add_message(session['id'],'assistant','Accuracy means anything we choose.')
        self.store.set_feedback(answer['id'],-1,'Accuracy needs a reference. Do not invent one.')
        lesson=self.memory.propose_feedback(self.pid,answer['id'],
                                          'Accuracy needs a reference. Do not invent one.')
        self.assertEqual(lesson['status'],'candidate')
        self.assertEqual(lesson['source_id'],answer['id'])
        self.assertFalse(self.memory.recall(self.pid,'What is accuracy?'))

    def test_s_research_reads_reviewed_teaching(self):
        self.teach()
        r=respond(self.store,self.pid,'Are there technical constraints?')
        self.assertIn('Constraints are hypotheses',r)
        self.assertIn('explicitly approved',r)
        self.assertIn('not independent proof',r)

    def test_export_import_retain_reviewed_memory_and_aliases(self):
        self.teach()
        self.memory.define_concept(self.pid,'barrier',['limitations','barrier'])
        candidate=self.memory.teach(self.pid,'Pending','Review the experiment',
                                    ['What experiments?'],status='candidate')
        self.memory.review(self.pid,candidate['id'],'rejected')
        document=self.store.export_project(self.pid)
        new=self.store.import_project(document)
        restored=TeacherMemory(self.store)
        self.assertEqual(len(restored.lessons(new['id'])),2)
        self.assertEqual(len(restored.lessons(new['id'],'approved')),1)
        self.assertEqual(restored.lessons(new['id'],'rejected')[0]['status'],'rejected')
        self.assertEqual(len(restored.concepts(new['id'])),1)
        self.assertEqual(len(restored.recall(new['id'],'barrier')),1)
        self.assertEqual(len(self.memory.lessons(self.pid)),2)

    def test_reject_malformed_or_duplicate_teaching_inputs(self):
        with self.assertRaises(ValueError):self.memory.define_concept(self.pid,'','')
        with self.assertRaises(ValueError):self.memory.define_concept(self.pid,'x',['same','same'])
        with self.assertRaises(ValueError):self.memory.teach(self.pid,'X','Y',[])
        with self.assertRaises(ValueError):self.memory.teach(self.pid,'X','Y',['A'],status='verified')
        with self.assertRaises(ValueError):self.memory.recall(self.pid,'')


class TeacherMemoryHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.server=AppServer('127.0.0.1',0,Path(cls.temp.name)/'database.sqlite3')
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.host='http://127.0.0.1:'+str(cls.server.server_port)
        cls.project=cls.server.store.list_projects()[0]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join(timeout=2);cls.temp.cleanup()

    def request(self,path,body=None):
        opts={'headers':{'Host':'127.0.0.1:'+str(self.server.server_port)}}
        if body is not None:
            opts.update(method='POST',data=json.dumps(body).encode('utf-8'))
            opts['headers']['Content-Type']='application/json'
        with urlopen(Request(self.host+path,**opts),timeout=5) as r:
            return json.load(r)

    def test_full_local_http_teach_recall_and_chat(self):
        pid=self.project['id']
        lesson=self.request('/api/memory/teach',{
            'project_id':pid,'title':'Words may transform meaning',
            'principle':'Adding a symbol changes meaning depending on context; test the result.',
            'cues':['Why do we add letters to words?','Why is S meaningful?']})
        self.assertEqual(lesson['status'],'approved')
        hits=self.request('/api/memory/recall?project_id='+pid+'&q=Why+is+S+meaningful%3F')
        self.assertEqual(hits[0]['id'],lesson['id'])
        session=self.request('/api/sessions',{'project_id':pid,'title':'Test meaning'})
        msg=self.request('/api/chat',{'session_id':session['id'],'model':'s-research:local',
                     'text':'Why is S meaningful?'})
        self.assertIn('Words may transform meaning',msg['content'])
        self.assertIn('Teacher memory',msg['content'])

    def test_api_keys_still_function(self):
        pid=self.project['id']
        key=self.request('/api/keys',{'name':'Test v06','scopes':['read']})
        req=Request(self.host+'/v1/models',headers={'Authorization':'Bearer '+key['key']})
        with urlopen(req,timeout=5) as r:self.assertIn('data',json.load(r))
        with self.assertRaises(HTTPError):urlopen(Request(self.host+'/v1/models'),timeout=5)

    def test_http_negative_feedback_is_not_approved(self):
        pid=self.project['id']
        session=self.server.store.create_session(pid,'Need correction')
        self.server.store.add_message(session['id'],'user','What is an observation?')
        answer=self.server.store.add_message(session['id'],'assistant','Everything is true')
        r=self.request('/api/feedback',{'message_id':answer['id'],'rating':-1,
                   'note':'Observations require recorded evidence'})
        self.assertTrue(r['memory_candidate_id'])
        candidates=self.request('/api/memory/lessons?project_id='+pid)
        matching=[x for x in candidates if x['id']==r['memory_candidate_id']]
        self.assertEqual(matching[0]['status'],'candidate')


if __name__=='__main__':unittest.main()

# Phrase-level generalization: different words can become a shared concept
# only after explicit human teaching, never a hardcoded keyword trigger.
class PhraseMeaningTests(unittest.TestCase):
    def test_phrase_equivalence_not_magic_trigger(self):
        with tempfile.TemporaryDirectory() as temp:
            store=Store(Path(temp)/'s.sqlite3');p=store.create_project('Teacher')
            memory=TeacherMemory(store)
            memory.teach(p['id'],'Investigative reasoning',
                         'Examine assumptions, compare possible explanations, and test evidence.',
                         ['Use rigorous analysis before deciding'])
            self.assertFalse(memory.recall(p['id'],'Apply careful investigation'))
            memory.define_concept(p['id'],'reasoning effort',
                                  ['rigorous analysis','careful investigation','deep thinking'])
            matches=memory.recall(p['id'],'Apply careful investigation')
            self.assertEqual(matches[0]['title'],'Investigative reasoning')
            self.assertIn('@reasoning_effort',matches[0]['matched_terms'])

    def test_competing_alias_sets_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            store=Store(Path(temp)/'s.sqlite3');p=store.create_project('Teacher')
            memory=TeacherMemory(store)
            memory.define_concept(p['id'],'effort',['deep thinking','careful analysis'])
            with self.assertRaisesRegex(ValueError,'competing'):
                memory.define_concept(p['id'],'another',['deep thinking','intensive inquiry'])

class ExplicitTeacherTransferTests(unittest.TestCase):
    def test_unreviewed_archive_does_not_cross_projects(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/'db.sqlite3')
            archive=store.create_project('Archive')
            invention=store.create_project('S')
            mem=TeacherMemory(store)
            lesson=mem.teach(archive['id'],'Teacher says S is transformative',
                             'A symbol changes the meaning of a word in context.',
                             ['Why is S meaningful?'],status='candidate')
            with self.assertRaisesRegex(ValueError,'approved'):
                mem.transfer(archive['id'],invention['id'],lesson['id'])
            self.assertEqual(mem.recall(invention['id'],'Why is S meaningful?'),[])
            mem.review(archive['id'],lesson['id'],'approved')
            mem.define_concept(archive['id'],'meaning',['meaning','significance'])
            result=mem.transfer(archive['id'],invention['id'],lesson['id'],include_concepts=True)
            self.assertEqual(result['lesson']['origin'],'import')
            self.assertEqual(result['concepts_copied'],1)
            self.assertIn(archive['id'],result['lesson']['source_id'])
            self.assertTrue(mem.recall(invention['id'],'What is its significance?'))
            with self.assertRaisesRegex(ValueError,'already been transferred'):
                mem.transfer(archive['id'],invention['id'],lesson['id'])

class ExportProvenanceTests(unittest.TestCase):
    def test_message_provenance_and_version_survive_import(self):
        with tempfile.TemporaryDirectory() as temp:
            store=Store(Path(temp)/'s.sqlite3');p=store.create_project('Teacher Archive')
            sess=store.create_session(p['id'],'We teach S')
            source=store.add_message(sess['id'],'user','Actually, S transforms meaning.')
            mem=TeacherMemory(store)
            lesson=mem.teach(p['id'],'How S transforms','The added symbol changes meaning.',
                             ['Why S?'],status='approved',origin='archive',source_id=source['id'])
            revised=mem.revise(p['id'],lesson['id'],'A symbol may change meaning depending on context.',
                                ['Why S?'])
            exported=store.export_project(p['id'])
            imported=store.import_project(exported)
            restored=TeacherMemory(store).lessons(imported['id'])
            self.assertEqual(len(restored),2)
            current=next(x for x in restored if x['status']=='approved')
            old=next(x for x in restored if x['status']=='superseded')
            self.assertEqual(current['version'],2)
            self.assertEqual(current['supersedes_id'],old['id'])
            self.assertEqual(old['origin'],'archive')
            with store.connect() as db:
                found=db.execute('SELECT m.role FROM messages m JOIN sessions s ON s.id=m.session_id '
                                 'WHERE m.id=? AND s.project_id=?',
                                 (old['source_id'],imported['id'])).fetchone()
            self.assertEqual(found['role'],'user')
