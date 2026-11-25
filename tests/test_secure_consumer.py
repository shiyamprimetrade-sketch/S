import unittest,tempfile
from pathlib import Path
from project_s.secure import SecureEngine
class SecurityTests(unittest.TestCase):
 def test_tokens(self):
  with tempfile.TemporaryDirectory() as t:
   e=SecureEngine(t);i,_=e.create('Create calculator');token=e.issue(i);token['scope']['operation']='shell'
   with self.assertRaises(ValueError):e.execute_token(i,token)
   with self.assertRaises(ValueError):e.execute_token(i,e.issue(i,-1))
   self.assertFalse((Path(t)/i).exists());token=e.issue(i);self.assertTrue(e.execute_token(i,token)['passed'])
   with self.assertRaises(ValueError):e.execute_token(i,token)
   f=SecureEngine(t).artifact(i);f.write_text('tampered')
   with self.assertRaises(ValueError):e.artifact(i)
 def test_symlink(self):
  with tempfile.TemporaryDirectory() as t:
   e=SecureEngine(t);i,_=e.create('Create website');(Path(t)/i).symlink_to('/tmp')
   with self.assertRaises(ValueError):e.execute_token(i,e.issue(i))
