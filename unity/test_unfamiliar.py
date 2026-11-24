import tempfile
import unittest
from pathlib import Path
from decimal import Decimal
from unity.cli import calculate

class UnfamiliarAcceptance(unittest.TestCase):
    def test_new_requirements(self):
        with tempfile.TemporaryDirectory() as tmp:
            for expr,expected in [("19*23","437"),("91-127","-36"),("-12+7","-5")]:
                with self.subTest(expr=expr):
                    result=calculate(expr,tmp,approved=True)
                    self.assertEqual(result["status"],"delivered")
                    self.assertEqual(Decimal(result["result"]["data"]["value"]),Decimal(expected))
    def test_reject_code_and_unsupported_operations(self):
        with tempfile.TemporaryDirectory() as tmp:
            for expr in ("2**8","1/0","__import__('os')"):
                with self.subTest(expr=expr),self.assertRaises(ValueError):
                    calculate(expr,tmp,approved=True)
    def test_denied_does_not_create_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"denied"
            self.assertEqual(calculate("19*23",root)["status"],"denied")
            self.assertFalse(root.exists())
