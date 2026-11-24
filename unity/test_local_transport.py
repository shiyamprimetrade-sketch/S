import hashlib
import hmac
import unittest
from unity.local_transport import canonical, execute_local

class TransportTests(unittest.TestCase):
    def setUp(self):
        self.key = b"k"*32
        self.task = dict(version=3,id="a",operation="calculator",args={})
        self.calls = 0
    def signature(self):
        return hmac.new(self.key,b"alpha-fusion-v3\0"+canonical(self.task),hashlib.sha256).hexdigest()
    def executor(self,task,signature,key,root,db,allowed):
        self.calls += 1
        record = dict(id=task["id"],request_digest=hashlib.sha256(canonical(task)).hexdigest(),result={"status":"ok"})
        return dict(record=record,tag=hmac.new(key,b"alpha-fusion-result-v3\0"+canonical(record),hashlib.sha256).hexdigest())
    def run_task(self,signature,allowed=("calculator",),executor=None):
        return execute_local(self.task,signature,self.key,".","test.db",allowed=allowed,executor=executor or self.executor)
    def test_valid(self):
        self.assertEqual(self.run_task(self.signature())["record"]["id"],"a")
        self.assertEqual(self.calls,1)
    def test_denied(self):
        with self.assertRaises(ValueError): self.run_task(self.signature(),())
        self.assertEqual(self.calls,0)
    def test_tampered(self):
        signature=self.signature()
        self.task["args"]["x"]=1
        with self.assertRaises(ValueError): self.run_task(signature)
        self.assertEqual(self.calls,0)
    def test_bad_evidence(self):
        def bad(*args,**kwargs):
            e=self.executor(*args,**kwargs)
            e["record"]["id"]="other"
            return e
        with self.assertRaises(ValueError): self.run_task(self.signature(),executor=bad)
    def test_version(self):
        self.task["version"]=2
        with self.assertRaises(ValueError): self.run_task(self.signature())
        self.assertEqual(self.calls,0)

if __name__=="__main__":
    unittest.main()
