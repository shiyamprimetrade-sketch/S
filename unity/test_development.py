import tempfile
import unittest
from pathlib import Path
from unity.development import develop

class DevelopmentTests(unittest.TestCase):
    def invoke(self,root,authorize=lambda r,t:True,repair=None,wrong=False):
        def planner(req):
            return dict(version=3,id="first",operation="calculator",
                        args=dict(a="7",op="+",b="2" if wrong else "4"))
        return develop("Calculate seven plus four",planner=planner,
            authorize=authorize,key=b"k"*32,root=root,db=root/"alpha.db",
            journal_db=root/"omega.db",allowed=("calculator",),
            verifiers={"calculator":lambda t,r,root,db:r["data"]["value"]=="11"},
            repair=repair)
    def test_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=self.invoke(Path(tmp))
            self.assertEqual(result["status"],"delivered")
            self.assertEqual(result["result"]["data"]["value"],"11")
    def test_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            result=self.invoke(root,authorize=lambda r,t:False)
            self.assertEqual(result["status"],"denied")
            self.assertFalse((root/"omega.db").exists())
    def test_bounded_repair(self):
        def repair(req,task,error):
            task["id"]="second"
            task["args"]["b"]="4"
            return task
        with tempfile.TemporaryDirectory() as tmp:
            result=self.invoke(Path(tmp),repair=repair,wrong=True)
            self.assertEqual(result["status"],"delivered")
            self.assertEqual([a["status"] for a in result["attempts"]],["failed","verified"])
    def test_repair_requires_authorization(self):
        def repair(req,task,error):
            task["id"]="second"
            return task
        with tempfile.TemporaryDirectory() as tmp:
            result=self.invoke(Path(tmp),authorize=lambda r,t:t["id"]=="first",repair=repair,wrong=True)
            self.assertEqual(result["status"],"denied")
            self.assertEqual(len(result["attempts"]),1)
