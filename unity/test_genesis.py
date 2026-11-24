import tempfile
import unittest
from pathlib import Path
from unity.genesis import interpret, request

class GenesisTests(unittest.TestCase):
    def test_subtraction_order(self):
        self.assertEqual(interpret("Subtract 7 from 20")["expression"],"20 - 7")
    def test_unknown(self):
        self.assertIsNone(interpret("Build a chess app"))
    def test_consent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"new"
            self.assertEqual(request("Add 2 and 3",root)["status"],"awaiting_authorization")
            self.assertFalse(root.exists())
    def test_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=request("Multiply 13 by 17",tmp,approved=True)
            self.assertEqual(result["status"],"delivered")
            self.assertEqual(result["result"]["data"]["value"],"221")
