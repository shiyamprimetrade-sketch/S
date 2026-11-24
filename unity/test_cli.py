import tempfile
import unittest
from pathlib import Path
from unity.cli import calculate

class CLITests(unittest.TestCase):
    def test_denied_without_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"new"
            self.assertEqual(calculate("8+9",root)["status"],"denied")
            self.assertFalse(root.exists())
    def test_approved(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=calculate("8+9",tmp,approved=True)
            self.assertEqual(result["status"],"delivered")
            self.assertEqual(result["result"]["data"]["value"],"17")
    def test_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                calculate("print(1)",tmp,approved=True)
