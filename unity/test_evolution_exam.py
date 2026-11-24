import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from html.parser import HTMLParser
from unity.project_builder import build_project, verify_project
from unity.project_repair import repair_project

class Exam(unittest.TestCase):
    def test_calculator_and_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"calculator"
            build_project(root,"calculator",approved=True)
            def accept(p):
                result=subprocess.run([sys.executable,"-I",str(p/"src/calculator.py")],input="137 * 29\n17 / 8\nquit\n",text=True,capture_output=True,timeout=10)
                return result.returncode==0 and ": 3973\n" in result.stdout and ": 2.125\n" in result.stdout
            self.assertTrue(verify_project(root) and accept(root))
            (root/"src/calculator.py").write_text("damaged")
            result=repair_project(root,Path(tmp)/"repaired","calculator",accept=accept,approved=True)
            self.assertEqual(result["status"],"repaired")
            self.assertEqual((root/"src/calculator.py").read_text(),"damaged")
    def test_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"inventory"
            build_project(root,"inventory",approved=True)
            self.assertTrue(verify_project(root))
            for args,expected in [(["register","LAB-73","31"],31),(["adjust","LAB-73","-12"],19),(["search","LAB-73"],19)]:
                p=subprocess.run([sys.executable,"-I",str(root/"src/inventory.py"),*args,"--db",str(Path(tmp)/"stock.db")],capture_output=True,text=True,timeout=10,check=True)
                self.assertEqual(json.loads(p.stdout)["quantity"],expected)
            self.assertTrue(verify_project(root))
    def test_website(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"site"
            build_project(root,"website",options={"name":"Ocean Lab","heading":"Explore oceans","description":"Marine research","sections":[("Currents","Water motion"),("Habitats","Marine life")]},approved=True)
            self.assertTrue(verify_project(root))
            class Parser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.links=[]
                    self.ids=[]
                def handle_starttag(self,tag,attrs):
                    a=dict(attrs)
                    if tag=="a": self.links.append(a.get("href"))
                    if tag=="article": self.ids.append(a.get("id"))
            parser=Parser()
            parser.feed((root/"src/index.html").read_text(encoding="utf-8"))
            self.assertEqual(parser.links,["#section-0","#section-1"])
            self.assertEqual(parser.ids,["section-0","section-1"])
    def test_denial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"denied"
            self.assertEqual(build_project(root,"inventory")["status"],"awaiting_authorization")
            self.assertFalse(root.exists())
