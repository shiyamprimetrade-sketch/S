"""Integrated host permissions, scoped memory and actual neural execution."""
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from continuum.brain_console import BrainConsole
from continuum.storage import Store
from s_brain.model import Config, SBrain
from s_brain.checkpoint import save_checkpoint


class ConsoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.shared = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.shared.name) / 'checkpoint'
        cfg = Config(dim=16, layers=1, heads=2, kv_heads=1, context=1024)
        torch.manual_seed(42)
        model = SBrain(cfg)
        before = model.embed.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        ids = torch.tensor([[65,66,67,68]])
        _, loss = model(ids, ids)
        loss.backward(); optimizer.step()
        assert not torch.equal(before, model.embed.weight)
        save_checkpoint(cls.checkpoint, model, cfg, step=1, train_tokens=4,
                        train_loss=float(loss.detach()), validation_loss=None,
                        seed=42, dataset_fingerprint='synthetic-unit-fixture')

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / 'app.sqlite3')
        self.project = self.store.create_project('Synthetic test')['id']
        self.other = self.store.create_project('Other project')['id']
        self.session = self.store.create_session(self.project)['id']
        self.console = BrainConsole(self.store, configuration={'enabled':True,
            'memory_enabled':True,'tools_enabled':True,'checkpoint':str(self.checkpoint)})

    def tearDown(self):
        if self.console.memory:
            self.console.memory.close()
        self.temp.cleanup()

    def request(self, route, **body):
        return self.console.request('POST','/api/brain/'+route,{},dict(project_id=self.project,**body))

    def test_explicit_permission_and_scope(self):
        with self.assertRaises(PermissionError):
            self.request('remember',content='A fact',source='fixture')
        record=self.request('remember',content='Synthetic sky is teal.',source='fixture',consent=True)
        self.assertEqual(self.console.request('GET','/api/brain/memory',{'project_id':[self.other]},None),[])
        self.assertFalse(self.console.request('POST','/api/brain/forget',{},
            {'project_id':self.other,'id':record['id'],'consent':True})['deleted'])
        revised=self.request('update',id=record['id'],content='Synthetic sky is red.',consent=True)
        self.assertNotEqual(record['content_sha256'],revised['content_sha256'])
        self.assertEqual(revised['revision'],2)
        self.assertTrue(self.request('forget',id=record['id'],consent=True)['deleted'])

    def test_neural_generation_without_teacher_and_no_memory_capture(self):
        with patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
            result=self.console.chat({'session_id':self.session,'text':'Hello'})
        self.assertEqual(result['model'],'s-brain:experimental')
        self.assertTrue(result['brain']['independent_inference'])
        self.assertIsNone(self.console.memory)
        self.assertEqual(len(self.store.list_messages(self.session)),2)

    def test_tools_require_user_and_host_permission(self):
        body={'session_id':self.session,'text':'Multiply 6 by 7', 'tool_intents':[
            {'id':'one','name':'calculator','arguments':{'operation':'multiply','left':6,'right':7}}]}
        with self.assertRaises(PermissionError):self.console.chat(body)
        body['approve_tools']=True
        result=self.console.chat(body)
        self.assertEqual(result['brain']['audit']['tools_executed'],1)
        self.console.config['tools_enabled']=False
        with self.assertRaises(PermissionError):self.console.chat(body)

    def test_disabled_and_invalid_checkpoint(self):
        self.assertTrue(self.console.status()['available'])
        self.console.config['checkpoint']=str(Path(self.temp.name)/'missing')
        self.assertFalse(self.console.status()['available'])
        self.console.config['enabled']=False
        with self.assertRaises(PermissionError):self.console.chat({'session_id':self.session,'text':'Hi'})
        self.console.config['memory_enabled']=False
        with self.assertRaises(PermissionError):
            self.request('remember',content='A fact',source='fixture',consent=True)

    def test_session_scope_cannot_be_overridden(self):
        with self.assertRaises(PermissionError):
            self.console.chat({'session_id':self.session,'project_id':self.other,'text':'Hi'})

    def test_benchmark_uses_declared_gain_threshold(self):
        from s_brain.memory_store import MemoryStore
        from s_brain.rag_evaluation import AnswerCase
        from s_brain.real_rag_benchmark import real_model_ablation, RealRagPlan
        path=Path(self.temp.name)/'benchmark.sqlite3'
        with MemoryStore(path):
            pass
        cases=[AnswerCase(identifier=str(i),question=q,expected_answer=a,
            scope='S',category='synthetic') for i,(q,a) in enumerate([
                ('What color is the capsule?','violet'),('What is the code word?','cedar')])]
        result=real_model_ablation(self.checkpoint,path,cases,
            plan=RealRagPlan(max_new_tokens=2,minimum_gain_to_claim=0.5))
        self.assertEqual(result['summary']['passes_declared_exact_gain'],
                         result['summary']['mean_exact_gain']>=0.5)
