import tempfile
import unittest
from pathlib import Path
from unity.evolution import dispatch

class EvolutionTests(unittest.TestCase):
    def test_status(self):
        self.assertFalse(dispatch("status","unused")["general_programming"])
    def test_consent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"new"
            self.assertEqual(dispatch('website "Lab" "Science" "Research"',root)["status"],"awaiting_authorization")
            self.assertFalse(root.exists())
    def test_website(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(dispatch('website "Lab" "Science" "Research"',Path(tmp)/"new",approved=True)["status"],"delivered")
    def test_unknown(self):
        self.assertEqual(dispatch("Build a chess app","unused",approved=True)["status"],"unsupported")
