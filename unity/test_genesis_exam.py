import tempfile
import unittest
from pathlib import Path
from unity.website_pipeline import build_website
from unity.engineering_memory import EngineeringMemory
from unity.genesis import request

class GenesisExam(unittest.TestCase):
    def test_unseen_sites(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i,(name,heading,desc) in enumerate([
                ("Museum","Explore history","Open Tuesday & Friday"),
                ("Repair Cafe","Fix & reuse","Bring your broken items"),
                ("Science Club","Experiments < 10 minutes","Learn together")]):
                root=Path(tmp)/str(i)
                result=build_website(root,name,heading,desc,approved=True)
                self.assertEqual(result["status"],"delivered")
                record=EngineeringMemory(root/"memory.db").recall(result["memory_id"])
                self.assertTrue(record["currently_verified"])
                self.assertEqual(record["spec"]["heading"],heading)
                Path(result["artifact"]).write_text("tampered",encoding="utf-8")
                self.assertFalse(EngineeringMemory(root/"memory.db").recall(result["memory_id"])["currently_verified"])
    def test_unseen_arithmetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=request("Multiply 37 by 29",tmp,approved=True)
            self.assertEqual(result["result"]["data"]["value"],"1073")
    def test_abstention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"new"
            self.assertEqual(request("Build a multiplayer chess app",root,approved=True)["status"],"unsupported")
            self.assertFalse(root.exists())
