import tempfile
import unittest
from pathlib import Path
from unity.evolution_requirements import extract
from unity.evolution import dispatch

class RequirementsTests(unittest.TestCase):
    phrase='Build a website called "Museum" with heading "History" and description "Open daily"'
    def test_extract(self):
        self.assertEqual(extract(self.phrase)["heading"],"History")
        self.assertFalse(extract(self.phrase)["authorized"])
    def test_clarification(self):
        self.assertEqual(extract("Build a website")["status"],"needs_clarification")
    def test_consent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"new"
            self.assertEqual(dispatch(self.phrase,root)["status"],"awaiting_authorization")
            self.assertFalse(root.exists())
    def test_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(dispatch(self.phrase,Path(tmp)/"new",approved=True)["status"],"delivered")
