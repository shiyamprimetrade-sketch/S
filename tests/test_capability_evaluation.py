"""Host-judged output, consent, leakage and bounded repair diagnostics."""
import json
import unittest
from unittest.mock import patch
from s_developer.capability import CapabilityEvaluator,judge,TASKS,PROTOCOL
from s_developer.providers import NativeProvider

FILES={'app.py':'def solve(p): return p\n','test_app.py':'# Model tests are not the acceptance oracle.\n'}


class Candidate:
    id='test-fixture'
    def __init__(self):self.calls=[]
    def provenance(self):return {'provider':'test-fixture','independently_generated_neural_code':False}
    def generate(self,spec,recipe):self.calls.append(('generate',spec));return FILES.copy()
    def repair(self,spec,recipe,files,diagnostic):
        self.calls.append(('repair',diagnostic));return dict(files,**{'app.py':'def solve(p): return {"fixed": True}\n'})


class CapabilityTests(unittest.TestCase):
    def test_exit_zero_and_fake_test_success_do_not_pass_oracle(self):
        for text in ('','Ran 999 tests: OK','{"total":5}\nextra','{"total":true}','{"total":5,"extra":1}','NaN'):
            self.assertFalse(judge({'status':'toolchain_passed','output':text},{'total':5})['passed'])
        self.assertTrue(judge({'status':'toolchain_passed','output':'{"total":5}'},{'total':5})['passed'])
        self.assertFalse(judge({'status':'toolchain_passed','output':'{"palindrome":1}'},{'palindrome':True})['passed'])
    def test_format_only_reports_execution_unassessed_and_no_promotion(self):
        evaluator=CapabilityEvaluator({})
        with patch('s_developer.capability.provider',return_value=Candidate()),patch('s_developer.capability.DockerRunner.run') as runner:
            report=evaluator.evaluate('fixture',True,False)
        runner.assert_not_called();self.assertEqual(report['format_passed'],3)
        self.assertEqual(report['final_tasks_correct'],0);self.assertFalse(report['suite_passed']);self.assertFalse(report['promotion_eligible'])
        self.assertTrue(all(t['status']=='execution_not_assessed' for t in report['tasks']))
    def test_consents_and_budgets_cannot_enable_host_execution(self):
        with self.assertRaises(PermissionError):CapabilityEvaluator({}).evaluate('template')
        with self.assertRaises(PermissionError):CapabilityEvaluator({}).evaluate('template',True,True)
        for budget in (-1,2,True):
            with self.assertRaises(ValueError):CapabilityEvaluator({}).evaluate('template',True,False,budget)
    def test_expected_answers_never_reach_generated_container(self):
        candidate=Candidate();sent=[]
        def run(files,recipe,**kwargs):
            self.assertEqual(set(kwargs),{'timeout','evaluation_input'})
            sent.append(kwargs['evaluation_input']);return {'status':'toolchain_passed','returncode':0,'output':'{}'}
        with patch('s_developer.capability.provider',return_value=candidate),patch('s_developer.capability.DockerRunner.run',side_effect=run):
            report=CapabilityEvaluator({'execution_enabled':True}).evaluate('fixture',True,True)
        self.assertEqual(len(sent),9);self.assertEqual(report['final_tasks_correct'],0)
        for _,spec in candidate.calls:
            self.assertNotIn('cases',spec);self.assertNotIn('expected',spec)
            self.assertNotIn('Never odd or even',json.dumps(spec))
    def test_withheld_feedback_is_not_leaked_to_repair(self):
        candidate=Candidate()
        with patch('s_developer.capability.provider',return_value=candidate),patch('s_developer.capability.DockerRunner.run',return_value={'status':'toolchain_passed','output':'{}'}):
            report=CapabilityEvaluator({'execution_enabled':True}).evaluate('fixture',True,True,1)
        repairs=[str(v) for k,v in candidate.calls if k=='repair']
        self.assertEqual(len(repairs),3)
        for value in repairs:
            self.assertNotIn('Never odd or even',value);self.assertNotIn('Zebra',value);self.assertNotIn('-9',value)
        self.assertTrue(all(len(r['attempts'])==2 for r in report['tasks']))
    def test_generation_and_runtime_blocks_are_not_success(self):
        candidate=Candidate()
        with patch('s_developer.capability.provider',return_value=candidate),patch('s_developer.capability.DockerRunner.run',side_effect=RuntimeError('No Docker')):
            report=CapabilityEvaluator({'execution_enabled':True}).evaluate('fixture',True,True)
        self.assertTrue(all(r['status']=='runtime_blocked' for r in report['tasks']));self.assertFalse(report['suite_passed'])
        with patch('s_developer.capability.provider',return_value=candidate),patch.object(candidate,'generate',side_effect=ValueError('invalid JSON')):
            report=CapabilityEvaluator({}).evaluate('fixture',True)
        self.assertEqual(report['format_passed'],0)
    def test_compact_contracts_fit_torch_demo_context_without_answers(self):
        for task in TASKS:self.assertLessEqual(len((PROTOCOL+task['contract']).encode()),256)
    def test_portable_provider_rejects_context_loss_before_inference(self):
        with self.assertRaisesRegex(ValueError,'no requirements were discarded'):
            NativeProvider('unused','stdlib')._generate('x'*65,'python-cli')
