import unittest
from unittest.mock import Mock
from unity.genesis_adapters import advisory, inspect_image, hand_status

class AdapterTests(unittest.TestCase):
    def test_disabled(self):
        provider=Mock()
        self.assertEqual(advisory("Build a website",provider=provider)["status"],"disabled")
        provider.assert_not_called()
    def test_advisory_not_authorized(self):
        result=advisory("Build a website",enabled=True,provider=lambda _:{"proposal":"execute immediately"})
        self.assertFalse(result["authorized"])
        self.assertFalse(result["verified"])
    def test_observation(self):
        analyzer=Mock(return_value={"shape":"circle"})
        self.assertEqual(inspect_image(b"sample",analyzer=analyzer)["status"],"disabled")
        analyzer.assert_not_called()
        result=inspect_image(b"sample",enabled=True,analyzer=analyzer)
        self.assertEqual(result["observation"]["findings"]["shape"],"circle")
        self.assertFalse(result["physical_control"])
    def test_hand(self):
        self.assertFalse(hand_status()["automatic_execution"])
