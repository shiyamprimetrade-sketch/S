import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unity.generation import generate_website
from unity.website_repair import repair_website

class RepairTests(unittest.TestCase):
    def test_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"a",Path(tmp)/"b"
            generate_website(a,"Shop","Wrong","Daily",approved=True)
            old=(a/"index.html").read_bytes()
            result=repair_website(a,b,"Shop","Correct","Daily",approved=True)
            self.assertEqual(result["status"],"repaired")
            self.assertEqual((a/"index.html").read_bytes(),old)
    def test_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"a",Path(tmp)/"b"
            self.assertEqual(repair_website(a,b,"Shop","Correct","Daily")["status"],"awaiting_repair_authorization")
            self.assertFalse(b.exists())
    def test_valid_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"a",Path(tmp)/"b"
            generate_website(a,"Shop","Correct","Daily",approved=True)
            self.assertEqual(repair_website(a,b,"Shop","Correct","Daily",approved=True)["status"],"no_repair_needed")
            self.assertFalse(b.exists())
    def test_failed_reverification(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"a",Path(tmp)/"b"
            with patch("unity.website_repair.verify_website",return_value={"status":"rejected"}):
                result=repair_website(a,b,"Shop","Correct","Daily",approved=True)
            self.assertEqual(result["status"],"repair_failed")
            self.assertEqual(result["attempts"],1)
