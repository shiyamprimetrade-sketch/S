import tempfile
import unittest
from pathlib import Path
from unity.project_builder import build_project
from unity.project_repair import repair_project

class RepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.original=Path(self.tmp.name)/"original"
        self.destination=Path(self.tmp.name)/"repaired"
        build_project(self.original,"calculator",approved=True)
    def accept(self,root):
        return "def calculate(a, op, b):" in (root/"src/calculator.py").read_text()
    def test_intact(self):
        self.assertEqual(repair_project(self.original,self.destination,"calculator",accept=self.accept)["status"],"no_repair_needed")
        self.assertFalse(self.destination.exists())
    def test_fault(self):
        path=self.original/"src/calculator.py"
        path.write_text("broken")
        result=repair_project(self.original,self.destination,"calculator",accept=self.accept,approved=True)
        self.assertEqual(result["status"],"repaired")
        self.assertEqual(path.read_text(),"broken")
    def test_denial(self):
        (self.original/"src/calculator.py").write_text("broken")
        self.assertEqual(repair_project(self.original,self.destination,"calculator",accept=self.accept)["status"],"awaiting_authorization")
        self.assertFalse(self.destination.exists())
    def test_rejected(self):
        result=repair_project(self.original,self.destination,"calculator",accept=lambda root:False,approved=True)
        self.assertEqual(result["status"],"repair_failed")
        self.assertEqual(result["attempts"],1)
