import sys
import types
import unittest
from unittest.mock import patch
from unity.intelligence import propose, brain_status, ml_status

class IntelligenceTests(unittest.TestCase):
    @patch("unity.intelligence.s_brain_bridge.generate")
    def test_proposal_not_authorized(self, generate):
        generate.return_value={"response":"print(1)"}
        result=propose("Build a calculator")
        self.assertFalse(result["authorized"])
        self.assertFalse(result["verified"])
        self.assertEqual(result["proposal"],"print(1)")
    def test_empty_requirement(self):
        with self.assertRaises(ValueError): propose(" ")
    @patch.dict("os.environ",{"AXIOM_ENABLE_S_BRAIN":"0"})
    def test_disabled_brain(self):
        self.assertFalse(brain_status()["ready"])
    def test_ml_abstains_without_active_model(self):
        pkg=types.ModuleType("s_ml")
        module=types.ModuleType("s_ml.promotion")
        module.active_model=lambda store,mission:None
        with patch.dict(sys.modules,{"s_ml":pkg,"s_ml.promotion":module}):
            self.assertFalse(ml_status(object(),"unity")["approved_model_available"])
    def test_ml_active_does_not_authorize_execution(self):
        pkg=types.ModuleType("s_ml")
        module=types.ModuleType("s_ml.promotion")
        module.active_model=lambda store,mission:("approved",object())
        with patch.dict(sys.modules,{"s_ml":pkg,"s_ml.promotion":module}):
            result=ml_status(object(),"unity")
            self.assertTrue(result["approved_model_available"])
            self.assertFalse(result["automatic_execution"])
