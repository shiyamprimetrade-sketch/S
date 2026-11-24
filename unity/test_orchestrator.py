import hashlib
import hmac
import tempfile
import unittest
from pathlib import Path
from unity.local_transport import canonical
from unity.orchestrator import run,pending

class IntegrationTests(unittest.TestCase):
    def test_real_calculator(self):
        from journal import Journal
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            key=b"k"*32
            task=dict(version=3,id="calc",operation="calculator",args=dict(a="2",op="+",b="3"))
            signature=hmac.new(key,b"alpha-fusion-v3\0"+canonical(task),hashlib.sha256).hexdigest()
            journal=root/"omega.db"
            args=(task,signature,key,root,root/"alpha.db",journal)
            with self.assertRaises(ValueError):
                run(*args,allowed=(),verifiers={})
            self.assertFalse(journal.exists())
            result,sealed=run(*args,allowed=("calculator",),verifiers={"calculator":lambda t,r,root,db:r["data"]["value"]=="5"})
            self.assertEqual(result["data"]["value"],"5")
            self.assertEqual(Journal(str(journal)).get("calc"),("completed",sealed))
            self.assertEqual(pending(journal),[])
            Journal(str(journal)).begin("interrupted")
            self.assertEqual(pending(journal),["interrupted"])

if __name__=="__main__":
    unittest.main()
