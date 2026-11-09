"""Real execution, falsification, correction, persistence, and API tests for S Invention Lab."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from continuum.invention import InventionLab, execute
from continuum.server import AppServer
from continuum.storage import Store


class InventionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name)/'s.sqlite3')
        self.a = self.store.create_project('S')
        self.b = self.store.create_project('Other')
        self.lab = InventionLab(self.store)
        self.goal = self.lab.create(self.a['id'], 'Investigate the meaning of S',
                                    'Generalize adding S to a new input')

    def tearDown(self): self.temp.cleanup()

    def teach(self, examples=(('CAT','CATS'), ('DOG','DOGS'))):
        for input_text, expected in examples:
            self.lab.add_example(self.a['id'], self.goal['id'], input_text, expected)

    def test_new_goal_requires_observable_success(self):
        with self.assertRaises(ValueError): self.lab.create(self.a['id'], 'A', '')

    def test_one_example_not_claimed_learning(self):
        self.lab.add_example(self.a['id'], self.goal['id'], 'CAT','CATS')
        with self.assertRaisesRegex(ValueError,'at least two'):
            self.lab.synthesize(self.a['id'], self.goal['id'])

    def test_s_rule_from_teaching_extrapolates_to_unseen(self):
        self.teach()
        hypothesis = self.lab.synthesize(self.a['id'], self.goal['id'])
        self.assertEqual(hypothesis['status'], 'hypothesis')
        self.assertEqual(hypothesis['candidate'], ['append_s'])
        unseen = self.lab.predict(self.a['id'], self.goal['id'], 'SUN')
        self.assertEqual(unseen['output'], 'SUNS')
        self.assertEqual(unseen['status'], 'hypothesis')
        self.lab.add_example(self.a['id'], self.goal['id'], 'SUN','SUNS','test')
        # Adding independent evidence invalidates stale verification, requires re-synthesis.
        with self.assertRaisesRegex(ValueError,'No candidate'):
            self.lab.evaluate(self.a['id'], self.goal['id'])
        self.lab.synthesize(self.a['id'], self.goal['id'])
        result = self.lab.evaluate(self.a['id'], self.goal['id'])
        self.assertTrue(result['passed'])
        self.assertEqual(result['status'],'verified_on_heldout_cases')
        self.assertEqual(len(result['evidence_sha256']),64)
        self.assertTrue(self.lab.detail(self.a['id'],self.goal['id'])['runs'][0]['passed'])

    def test_composed_program_learned(self):
        self.teach((('AB','BABA'),('CD','DCDC')))
        rule=self.lab.synthesize(self.a['id'],self.goal['id'])['candidate']
        self.assertEqual(len(rule),2)
        self.assertEqual(execute(rule,'EF'),'FEFE')

    def test_unseen_failure_does_not_get_hidden(self):
        self.teach()
        self.lab.add_example(self.a['id'],self.goal['id'],'SUN','OTHER','test')
        self.lab.synthesize(self.a['id'],self.goal['id'])
        outcome=self.lab.evaluate(self.a['id'],self.goal['id'])
        self.assertFalse(outcome['passed'])
        self.assertEqual(outcome['results'][0]['actual'],'SUNS')
        self.assertEqual(self.lab.detail(self.a['id'],self.goal['id'])['status'],'falsified_by_heldout_cases')

    def test_teacher_correction_resynthesizes_and_keeps_old_run(self):
        self.teach()
        heldout=self.lab.add_example(self.a['id'],self.goal['id'],'SUN','WRONG','test')
        self.lab.synthesize(self.a['id'],self.goal['id'])
        old=self.lab.evaluate(self.a['id'],self.goal['id'])
        self.assertFalse(old['passed'])
        correction=self.lab.correct(self.a['id'],self.goal['id'],heldout['id'],
                                    'SUNS','The intended rule is S as suffix')
        self.assertEqual(correction['candidate'],['append_s'])
        new=self.lab.evaluate(self.a['id'],self.goal['id'])
        self.assertTrue(new['passed'])
        runs=self.lab.detail(self.a['id'],self.goal['id'])['runs']
        self.assertEqual(len(runs),2)
        self.assertFalse(runs[1]['passed'])
        self.assertTrue(runs[0]['passed'])
        self.assertGreater(new['run_id'] != old['run_id'],False)
        self.assertEqual(self.lab.detail(self.a['id'],self.goal['id'])['examples'][-1]['revision'],1)

    def test_incompatible_teacher_correction_keeps_unresolved(self):
        self.teach()
        first=self.lab.detail(self.a['id'],self.goal['id'])['examples'][0]
        correction=self.lab.correct(self.a['id'],self.goal['id'], first['id'],'MAGIC','Different output')
        self.assertEqual(correction['status'],'unresolved')
        with self.assertRaisesRegex(ValueError,'No current candidate'):
            self.lab.predict(self.a['id'],self.goal['id'],'SUN')

    def test_persistence_in_sqlite(self):
        self.teach()
        self.lab.synthesize(self.a['id'],self.goal['id'])
        again=InventionLab(Store(self.store.path))
        self.assertEqual(again.predict(self.a['id'],self.goal['id'],'TREE')['output'],'TREES')
        self.assertEqual(len(again.list(self.a['id'])),1)

    def test_cross_project_isolation(self):
        self.teach()
        for fn in [lambda:self.lab.detail(self.b['id'],self.goal['id']),
                   lambda:self.lab.synthesize(self.b['id'],self.goal['id']),
                   lambda:self.lab.add_example(self.b['id'],self.goal['id'],'AA','AAS')]:
            with self.assertRaisesRegex(ValueError,'Unknown invention'):
                fn()
        self.assertEqual(self.lab.list(self.b['id']),[])

    def test_rejects_arbitrary_code_and_overlong(self):
        with self.assertRaises(ValueError): execute(['os.system'], 'echo test')
        with self.assertRaises(ValueError): execute(['duplicate']*3,'hi')
        with self.assertRaises(ValueError): execute(['identity'],'x'*501)
        with self.assertRaises(ValueError): self.lab.add_example(self.a['id'],self.goal['id'],'','')

    def test_duplicate_input_cannot_pollute_holdout(self):
        self.teach()
        with self.assertRaisesRegex(ValueError,'Duplicate input'):
            self.lab.add_example(self.a['id'],self.goal['id'],'CAT','CATS','test')

    def test_no_verification_without_holdout(self):
        self.teach()
        self.lab.synthesize(self.a['id'],self.goal['id'])
        with self.assertRaisesRegex(ValueError,'held-out'):
            self.lab.evaluate(self.a['id'],self.goal['id'])


class InventionHttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.server=AppServer('127.0.0.1',0,Path(self.tmp.name)/'local.sqlite3')
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}'
        self.project=self.server.store.list_projects()[0]['id']

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=3)
        self.tmp.cleanup()

    def request(self,path,body=None):
        data=json.dumps(body).encode() if body is not None else None
        req=Request(self.url+path,data=data,headers={'Content-Type':'application/json'},
                    method='POST' if data is not None else 'GET')
        try:
            with urlopen(req,timeout=5) as result:return result.status,json.load(result)
        except HTTPError as err:return err.code,json.loads(err.read())

    def test_actual_http_teach_test_evaluate(self):
        code,goal=self.request('/api/inventions',{'project_id':self.project,
             'objective':'Investigate adding S','success_criteria':'Generalize to a new word'})
        self.assertEqual(code,200)
        base={'project_id':self.project,'goal_id':goal['id']}
        for a,b,r in [('CAT','CATS','teach'),('DOG','DOGS','teach'),('SUN','SUNS','test')]:
            self.assertEqual(self.request('/api/invention/example',dict(base,input=a,expected=b,role=r))[0],200)
        code,rule=self.request('/api/invention/synthesize',base)
        self.assertEqual(rule['candidate'],['append_s'])
        code,proof=self.request('/api/invention/evaluate',base)
        self.assertEqual(code,200);self.assertTrue(proof['passed'])
        self.assertEqual(self.request('/api/invention/predict',dict(base,input='TREE'))[1]['output'],'TREES')
        self.assertEqual(self.request('/api/invention?project_id='+self.project+'&goal_id='+goal['id'])[1]['runs'][0]['evidence_sha256'],proof['evidence_sha256'])

    def test_http_unknown_project_error(self):
        code,_=self.request('/api/inventions',{'project_id':'fake','objective':'test new goal','success_criteria':'prove result'})
        self.assertEqual(code,400)

    def test_ui_exposes_invention_lab(self):
        root=Path(__file__).resolve().parent.parent
        html=(root/'web/index.html').read_text(encoding='utf8')
        js=(root/'web/app.js').read_text(encoding='utf8')
        self.assertIn('id="invention-panel"',html)
        self.assertIn('id="invention-open"',html)
        self.assertIn('refreshInventions',js)
        self.assertIn("'/api/invention/correct'",(root/'continuum/server.py').read_text())

if __name__=='__main__':unittest.main()

class InventionContinuityTests(unittest.TestCase):
    def test_project_export_import_preserves_lessons_and_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            store=Store(Path(d)/'s.sqlite3')
            p=store.create_project('Teacher')
            lab=InventionLab(store)
            goal=lab.create(p['id'],'Explore S as suffix','Prove on unseen input')
            for a,b,r in [('CAT','CATS','teach'),('DOG','DOGS','teach'),('SUN','SUNS','test')]:
                lab.add_example(p['id'],goal['id'],a,b,r)
            lab.synthesize(p['id'],goal['id'])
            proof=lab.evaluate(p['id'],goal['id'])
            archive=store.export_project(p['id'])
            self.assertEqual(len(archive['inventions']),1)
            self.assertIn('Invention experiments',store.handoff(p['id']))
            # Import into the same database as a second, isolated project.
            restored=store.import_project(archive)
            self.assertNotEqual(restored['id'],p['id'])
            goal2=InventionLab(store).list(restored['id'])[0]
            self.assertEqual(goal2['status'],'verified_on_heldout_cases')
            detail=InventionLab(store).detail(restored['id'],goal2['id'])
            self.assertEqual(detail['candidate'],['append_s'])
            self.assertEqual(detail['runs'][0]['evidence_sha256'],proof['evidence_sha256'])
            self.assertEqual(len(detail['examples']),3)
