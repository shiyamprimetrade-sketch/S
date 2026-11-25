import unittest,tempfile
from pathlib import Path
from project_s.generation import plan,generate
from project_s.verification import verify
from project_s.calculator import calculate
class ConsumerTests(unittest.TestCase):
    def test_generators(self):
        with tempfile.TemporaryDirectory() as t:
            for k in ('calculator','website','inventory'):
                p=plan('Create a '+k+' for River Studio'); f=generate(p,Path(t)/k); self.assertTrue(verify(f,k)['passed'])
    def test_invalid(self):
        for x in ('','build a spaceship','calculator and website','delete inventory'):
            with self.assertRaises(ValueError):plan(x)
    def test_calc_errors(self):
        for x in ('1/0','__import__("os")','2**3'):
            with self.assertRaises(Exception):calculate(x)
    def test_authorization_and_persistence(self):
        from project_s.orchestrator import Orchestrator
        with tempfile.TemporaryDirectory() as t:
            o=Orchestrator(t); task,p=o.create('Create calculator'); o.execute(task,False); self.assertFalse((Path(t)/task).exists())
            with self.assertRaises(ValueError):o.execute(task,True)
            task,p=o.create('Create inventory'); self.assertTrue(Orchestrator(t).execute(task,True)['passed'])
            with self.assertRaises(ValueError):o.execute(task,True)
