import unittest
from unity.interaction import interact

class InteractionTests(unittest.TestCase):
    def setUp(self):
        self.calls=[]
    def driver(self,cmd):
        self.calls.append(cmd)
        return {"command_id":cmd["command_id"],"status":"completed"}
    def invoke(self,authorized=True,verify=lambda b,a,c:False,driver=None):
        frames=iter([b"before",b"after"])
        return interact("hello","Notepad",authorized=authorized,
                        driver=driver or self.driver,capture=lambda:next(frames),
                        analyzer=lambda image:{"label":image.decode()},verify=verify)
    def test_denial(self):
        with self.assertRaises(PermissionError): self.invoke(False)
        self.assertEqual(self.calls,[])
    def test_ack_not_verification(self):
        self.assertFalse(self.invoke()["verified"])
    def test_verified_outcome(self):
        result=self.invoke(verify=lambda b,a,c:a["findings"]["label"]=="after")
        self.assertTrue(result["verified"])
        self.assertNotEqual(result["before"]["sha256"],result["after"]["sha256"])
    def test_wrong_receipt(self):
        with self.assertRaises(RuntimeError):
            self.invoke(driver=lambda c:{"command_id":"wrong","status":"completed"})
