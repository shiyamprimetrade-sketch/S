import unittest
from unittest.mock import patch
from unity.nexus_runtime import probe

class RuntimeTests(unittest.TestCase):
    @patch("unity.nexus_runtime.s_brain_bridge.read_health")
    def test_denial(self,health):
        self.assertEqual(probe("test")["status"],"awaiting_authorization")
        health.assert_not_called()
    @patch("unity.nexus_runtime.s_brain_bridge.generate")
    @patch("unity.nexus_runtime.s_brain_bridge.read_health",return_value={"ready":False})
    def test_unavailable(self,health,generate):
        self.assertEqual(probe("test",approved=True)["status"],"unavailable")
        generate.assert_not_called()
    @patch("unity.nexus_runtime.s_brain_bridge.generate",return_value={"response":"example","model":"s-brain:experimental"})
    @patch("unity.nexus_runtime.s_brain_bridge.read_health",return_value={"ready":True})
    def test_response(self,health,generate):
        r=probe("test",approved=True)
        self.assertEqual(r["status"],"response_received")
        self.assertEqual(len(r["response_sha256"]),64)
        self.assertFalse(r["executed"])
        self.assertFalse(r["verified"])
