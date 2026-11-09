import unittest
from public_replay import replay

class TestReplay(unittest.TestCase):
 def test_all_paths_fixed_inputs(self):
  x=[{'input':'Teach me an unfamiliar symbol','core':'Examine the symbol','test':'Reproduce its defined transformation','alternative':'Learn a meaning through interaction','phase':'teaching'},
     {'input':'No, do not impose a binary choice','core':'Keep both options open','test':'Compare alternatives','alternative':'Learn feedback','intent':'correction'}]
  r=replay(x)
  self.assertEqual(len(r['chats']),11)
  self.assertTrue(all(len(c['turns'])==2 for c in r['chats']))
  self.assertEqual(len({c['turns'][0]['input'] for c in r['chats']}),1)
  self.assertEqual(len({c['turns'][0]['output'] for c in r['chats']}),11)
 def test_determinism(self):
  x=[{'input':'Test instruction'}]
  self.assertEqual(replay(x),replay(x))
 def test_invalid(self):
  with self.assertRaises(ValueError):replay([])
  with self.assertRaises(ValueError):replay([{'input':''}])
if __name__=='__main__':unittest.main()