import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from project_s.secure import SecureEngine
from project_s.repair import Repair

class RepairTests(unittest.TestCase):
    def test_repair_limits_and_preservation(self):
        with tempfile.TemporaryDirectory() as root:
            e=SecureEngine(root)
            task,_=e.create('Create calculator')
            e.transition(task,'AWAITING_AUTHORIZATION','FAILED',{'error':'injected defect'})
            d=Path(root)/task
            d.mkdir()
            original=d/'calculator.py'
            original.write_text('broken original')
            repair=Repair(e)
            proposal=repair.propose(task)
            with self.assertRaises(ValueError):repair.authorize(proposal,False)
            token=repair.authorize(proposal,True)
            tampered=dict(token,mac='0'*64)
            with self.assertRaises(ValueError):repair.execute(tampered)
            with patch('project_s.orchestrator.verify',side_effect=RuntimeError('injected verifier failure')):
                with self.assertRaises(RuntimeError):repair.execute(token)
            self.assertEqual(original.read_text(),'broken original')
            with self.assertRaises(ValueError):repair.execute(token)
            result=repair.execute(repair.authorize(repair.propose(task),True))
            self.assertEqual(result['status'],'VERIFIED')
            self.assertTrue(e.artifact(result['child']).is_file())
            repair.execute(repair.authorize(repair.propose(task),True))
            with self.assertRaises(ValueError):repair.propose(task)
            self.assertEqual(original.read_text(),'broken original')
            self.assertEqual(e.get(task)['state'],'FAILED')
