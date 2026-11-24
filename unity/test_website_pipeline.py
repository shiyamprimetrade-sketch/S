import tempfile
import unittest
from pathlib import Path
from unity.website_pipeline import build_website
from unity.engineering_memory import EngineeringMemory

class PipelineTests(unittest.TestCase):
    def test_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"project"
            r=build_website(root,"Orchard","Fresh apples","Open daily",approved=True)
            self.assertEqual(r["status"],"delivered")
            self.assertTrue(Path(r["artifact"]).is_file())
            self.assertTrue(EngineeringMemory(root/"memory.db").recall(r["memory_id"])["currently_verified"])
    def test_denial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"project"
            self.assertEqual(build_website(root,"Orchard","Apples","Daily")["status"],"awaiting_authorization")
            self.assertFalse(root.exists())
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                build_website(tmp,"Orchard","Apples","Daily",approved=True)
    def test_escaped_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=build_website(Path(tmp)/"project","Lab","<Experiment>","A & B",approved=True)
            self.assertEqual(r["status"],"delivered")
            self.assertIn("&lt;Experiment&gt;",Path(r["artifact"]).read_text())
