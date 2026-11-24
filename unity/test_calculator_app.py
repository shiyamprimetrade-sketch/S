import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
from unity.calculator_app import generate_calculator

class CalculatorAppTests(unittest.TestCase):
    def test_consent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"app"
            self.assertEqual(generate_calculator(root)["status"],"awaiting_authorization")
            self.assertFalse(root.exists())
    def test_arithmetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=generate_calculator(Path(tmp)/"app",approved=True)
            p=subprocess.run([sys.executable,"-I",r["artifact"]],input="19 + 23\n9 - 14\n12 * 13\n7 / 2\nquit\n",text=True,capture_output=True,timeout=10,check=True)
            for value in ("42","-5","156","3.5"):
                self.assertIn(": "+value+"\n",p.stdout)
    def test_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=generate_calculator(Path(tmp)/"app",approved=True)
            p=subprocess.run([sys.executable,"-I",r["artifact"]],input="1 / 0\nNaN + 2\n1 ^ 2\nquit\n",text=True,capture_output=True,timeout=10,check=True)
            for message in ("Division by zero","Invalid number","Invalid operation"):
                self.assertIn(message,p.stdout)
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                generate_calculator(tmp,approved=True)
