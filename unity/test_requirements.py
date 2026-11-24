import unittest
from unity.requirements import specify

class RequirementsTests(unittest.TestCase):
    def test_ready(self):
        s=specify("Multiply 12 by 9")
        self.assertEqual(s["status"],"ready")
        self.assertFalse(s["authorized"])
    def test_website(self):
        s=specify("Build a website for a bakery")
        self.assertEqual(s["kind"],"website")
        self.assertEqual(s["status"],"needs_clarification")
        self.assertFalse(s["execution_supported"])
    def test_unknown(self):
        self.assertEqual(specify("Build a chess engine")["status"],"unsupported")
    def test_empty(self):
        with self.assertRaises(ValueError):
            specify(" ")
