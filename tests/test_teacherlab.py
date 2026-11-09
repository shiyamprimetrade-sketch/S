"""Prove generated records are labelled and the offline model is reproducible."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from continuum import teacherlab as lab
from continuum.engine import chat, TEACHER_MODEL
from continuum.storage import Store

class GeneratorTests(unittest.TestCase):
    def test_seed_reproducible(self):
        first=list(lab.generate(31,seed=8))
        self.assertEqual(first,list(lab.generate(31,seed=8)))
        self.assertNotEqual(first,list(lab.generate(31,seed=9)))
        self.assertEqual(len({x['id'] for x in first}),31)
    def test_exact_count_and_provenance(self):
        rows=list(lab.read_jsonl(lab.SIMULATIONS))
        self.assertEqual(len(rows),4000)
        self.assertEqual(len({r['id'] for r in rows}),4000)
        self.assertTrue(all(r['source']=='synthetic-template-simulation' and not r['human_verified'] for r in rows))
        curated=list(lab.read_jsonl(lab.LESSONS))
        self.assertTrue(all(r['source']=='human-derived-paraphrase' and r['original_verbatim'] is False for r in curated))
    def test_invalid_counts_and_split(self):
        with self.assertRaises(ValueError):list(lab.generate(0))
        with self.assertRaises(ValueError):list(lab.generate(1,split='private'))
    def test_heldout_not_train_templates(self):
        train=set(x['teacher'] for x in lab.generate(4000))
        holdout={x['teacher'] for x in lab.generate(350,20261110,'heldout')}
        self.assertFalse(train & holdout)
    def test_measured_score_and_reproducibility(self):
        m=lab.load_model()
        x=lab.evaluate(m,lab.generate(350,20261110,'heldout'))
        report=json.loads(lab.REPORT.read_text())
        self.assertEqual(x,report['heldout'])
        self.assertGreaterEqual(x['accuracy'],0.65)
        self.assertLessEqual(x['accuracy'],1.0)
    def test_unknown_input_is_not_falsely_understood(self):
        m=lab.load_model()
        self.assertEqual(lab.predict(m,'zxczxbcv qqqqqqq ppzixczz')['label'],'unknown')
        self.assertEqual(lab.predict(m,'')['label'],'unknown')
    def test_offline_response_no_llm_claim(self):
        response=lab.respond_offline('The keyboard letters are symbols')
        self.assertIn('Experimental TeacherLab classification',response)
        self.assertIn('not proof of understanding',response)
    def test_offline_chat_no_network_and_persistent(self):
        with tempfile.TemporaryDirectory() as d:
            s=Store(Path(d)/'m.sqlite3');p=s.create_project('Course');se=s.create_session(p['id'])
            with patch('continuum.engine.urlopen',side_effect=AssertionError('network forbidden')):
                result=chat(s,se['id'],'How can words interact using letters?',model=TEACHER_MODEL)
            self.assertEqual(result['model'],TEACHER_MODEL)
            reopened=Store(Path(d)/'m.sqlite3')
            self.assertEqual([x['role'] for x in reopened.list_messages(se['id'])],['user','assistant'])
    def test_source_lesson_excludes_secrets(self):
        source=lab.LESSONS.read_text().lower()
        for secret_word in ('eyjhi', 'password=', 'secret_key=', 'token=', 'api_key='):
            self.assertNotIn(secret_word, source)
    def test_backend_classifier_classification_is_not_a_chatgpt_clone(self):
        self.assertEqual(lab.load_model()['algorithm'],'multinomial-naive-bayes')
        self.assertIn('not a trained conversational llm',lab.load_model()['note'].lower())

if __name__=='__main__': unittest.main()
