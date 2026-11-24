import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unity.inventory_app import generate_inventory

class InventoryTests(unittest.TestCase):
    def run_app(self,root,*args):
        p=subprocess.run([sys.executable,"-I",str(root/"inventory.py"),*args],capture_output=True,text=True,timeout=10)
        return p.returncode,json.loads(p.stdout)
    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"app"
            generate_inventory(root,approved=True)
            self.assertEqual(self.run_app(root,"register","A1","10"),(0,{"sku":"A1","quantity":10}))
            self.assertEqual(self.run_app(root,"adjust","A1","-3"),(0,{"sku":"A1","quantity":7}))
            self.assertEqual(self.run_app(root,"search","A1"),(0,{"sku":"A1","quantity":7}))
    def test_invalid_stock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"app"
            generate_inventory(root,approved=True)
            self.run_app(root,"register","A1","2")
            self.assertEqual(self.run_app(root,"adjust","A1","-3")[0],1)
            self.assertEqual(self.run_app(root,"register","A1","9")[0],1)
            self.assertEqual(self.run_app(root,"search","A1")[1]["quantity"],2)
    def test_consent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"app"
            self.assertEqual(generate_inventory(root)["status"],"awaiting_authorization")
            self.assertFalse(root.exists())
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                generate_inventory(tmp,approved=True)
