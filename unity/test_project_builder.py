import tempfile
import unittest
from pathlib import Path
from unity.project_builder import build_project,verify_project

class ProjectTests(unittest.TestCase):
    def test_all_kinds(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind,options in [("calculator",{}),("inventory",{}),("website",{"name":"Lab","heading":"Science","description":"Research","sections":[("Physics","Motion")]})]:
                root=Path(tmp)/kind
                r=build_project(root,kind,options=options,approved=True)
                self.assertTrue(verify_project(root))
                self.assertTrue((root/r["manifest"]["entry"]).is_file())
    def test_denial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"new"
            self.assertEqual(build_project(root,"calculator")["status"],"awaiting_authorization")
            self.assertFalse(root.exists())
    def test_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"new"
            build_project(root,"calculator",approved=True)
            (root/"src/calculator.py").write_text("changed")
            self.assertFalse(verify_project(root))
    def test_extra(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"new"
            build_project(root,"inventory",approved=True)
            (root/"extra.txt").write_text("extra")
            self.assertFalse(verify_project(root))
