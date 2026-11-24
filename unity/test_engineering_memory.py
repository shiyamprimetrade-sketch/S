import tempfile
import unittest
from pathlib import Path
from unity.generation import generate_website
from unity.engineering_memory import EngineeringMemory

class MemoryTests(unittest.TestCase):
    def test_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"site"
            db=Path(tmp)/"memory.db"
            generate_website(root,"Library","Books","Daily",approved=True)
            ident=EngineeringMemory(db).remember(root,"Library","Books","Daily")
            self.assertTrue(EngineeringMemory(db).recall(ident)["currently_verified"])
    def test_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=EngineeringMemory(Path(tmp)/"memory.db")
            with self.assertRaises(ValueError):
                m.remember(Path(tmp)/"missing","Site","Title","Text")
    def test_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"site"
            generate_website(root,"Site","Title","Text",approved=True)
            m=EngineeringMemory(Path(tmp)/"memory.db")
            ident=m.remember(root,"Site","Title","Text")
            (root/"index.html").write_text("changed")
            self.assertFalse(m.recall(ident)["currently_verified"])
    def test_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(EngineeringMemory(Path(tmp)/"memory.db").recall("missing"))
