import json
from pathlib import Path
import tempfile
import unittest
from human_computer.core import HumanComputer

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.pc=HumanComputer(self.tmp.name)
    def tearDown(self):
        self.pc.close();self.tmp.cleanup()

    def test_learning_synthesis_and_unseen_validation(self):
        self.pc.teach('mirror','CAT','CATTAC')
        self.pc.teach('mirror','DOG','DOGGOD')
        finding=self.pc.invent('mirror')
        self.assertEqual(finding.status,'CANDIDATE_FOUND')
        self.assertEqual(self.pc.perform('mirror','SUN').text,'SUNNUS')
        self.assertTrue(self.pc.verify('mirror','SUN','SUNNUS').verified)

    def test_correction_does_not_fake_success(self):
        self.pc.teach('mirror','CAT','CATTAC')
        r=self.pc.teach('mirror','CAT','CAT')
        self.assertEqual(r.status,'CONFLICT')
        self.assertFalse(r.verified)

    def test_goal_default_depth_without_trigger_word(self):
        x=self.pc.respond('Invent a new computing system')
        self.assertEqual(x.status,'PLANNED')
        self.assertTrue(x.evidence['automatic_deep_analysis'])
        self.assertFalse(x.verified)

    def test_memory_survives_restart(self):
        self.pc.remember('principle','meaning before labels')
        self.pc.close()
        self.pc=HumanComputer(self.tmp.name)
        self.assertEqual(self.pc.recall('meaning')[0]['name'],'principle')

    def test_evidence_records_actual_proof(self):
        self.pc.respond('goal explore a novel OS')
        self.assertEqual(self.pc.events()[0]['status'],'PLANNED')

    def test_file_perception_and_workspace_only(self):
        self.pc.write_file('test.txt','ABC',confirmed=True)
        obs=self.pc.observe_file('test.txt')
        self.assertEqual(obs.evidence['characters'],3)
        self.assertEqual(obs.evidence['bytes'],3)
        with self.assertRaises(ValueError): self.pc.observe_file('../outside')

    def test_modification_requires_consent(self):
        r=self.pc.write_file('test.txt','ABC',confirmed=False)
        self.assertEqual(r.status,'APPROVAL_REQUIRED')
        self.assertFalse((self.pc.workspace/'test.txt').exists())

    def test_memory_commands(self):
        self.assertEqual(self.pc.respond('remember philosophy seek meaning').status,'STORED')
        self.assertIn('philosophy', self.pc.respond('recall philosophy').text)

    def test_system_status_truthful(self):
        s=self.pc.status()
        self.assertFalse(s['kernel_replaced'])
        self.assertFalse(s['human_level_intelligence'])

    def test_invent_without_evidence(self):
        r=self.pc.respond('invent imaginary')
        self.assertEqual(r.status,'ACTION_NOT_COMPLETED')
        self.assertFalse(r.verified)

    def test_read_cannot_escape_via_symlink(self):
        outside=Path(self.tmp.name)/'outside.txt';outside.write_text('secret')
        link=self.pc.workspace/'linked'
        try:link.symlink_to(outside)
        except OSError:self.skipTest('Symlink not supported')
        with self.assertRaises(ValueError):self.pc.observe_file('linked')

    def test_failed_validation_not_claimed(self):
        self.pc.teach('mirror','CAT','CATTAC')
        self.pc.teach('mirror','DOG','DOGGOD')
        self.pc.invent('mirror')
        x=self.pc.verify('mirror','SUN','BAD')
        self.assertEqual(x.status,'TEST_FAIL')
        self.assertFalse(x.verified)

if __name__=='__main__':unittest.main()