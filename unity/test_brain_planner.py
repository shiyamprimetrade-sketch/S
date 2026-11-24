import unittest
from unity.brain_planner import plan

class BrainPlannerTests(unittest.TestCase):
    def test_disabled(self):
        self.assertEqual(plan("calculator",health=lambda:self.fail("unexpected connection"))["status"],"disabled")
    def test_unavailable(self):
        self.assertEqual(plan("calculator",enabled=True,health=lambda:{"ready":False},generate=lambda *a,**k:self.fail("unexpected generation"))["status"],"unavailable")
    def test_proposal(self):
        r=plan("calculator",enabled=True,health=lambda:{"ready":True},generate=lambda *a,**k:{"response":'{"kind":"calculator","features":["division"]}'})
        self.assertEqual(r["status"],"proposed")
        self.assertFalse(r["authorized"])
        self.assertFalse(r["verified"])
    def test_rejection(self):
        for response in ['print(1)','{"kind":"calculator","features":["shell"]}','{"kind":"inventory","features":["search"],"execute":true}']:
            self.assertEqual(plan("inventory",enabled=True,health=lambda:{"ready":True},generate=lambda *a,**k:{"response":response})["status"],"rejected")
