import tempfile, unittest
from pathlib import Path
from project_s.secure import SecureEngine

class RecoveryTests(unittest.TestCase):
    def test_interrupted_generation(self):
        with tempfile.TemporaryDirectory() as root:
            e=SecureEngine(root)
            task,_=e.create('Create calculator')
            e.transition(task,'AWAITING_AUTHORIZATION','AUTHORIZED')
            e.transition(task,'AUTHORIZED','GENERATING')
            d=Path(root)/task
            d.mkdir()
            f=d/'calculator.py'
            f.write_text('partial')
            e=SecureEngine(root)
            self.assertEqual(e.recover_interrupted(),[task])
            self.assertEqual(e.get(task)['state'],'FAILED')
            self.assertEqual(f.read_text(),'partial')
            self.assertEqual(e.recover_interrupted(),[])
            with self.assertRaises(ValueError):e.issue(task)
