"""Opt-in independent S-Brain bridge: no torch dependency in Continuum."""
import io
import json
import os
import unittest
from unittest.mock import patch

from continuum import s_brain_bridge as bridge


class MockResponse:
    def __init__(self, value):
        self.raw = io.BytesIO(json.dumps(value).encode("utf-8"))

    def __enter__(self):
        return self.raw

    def __exit__(self, exc_type, exc, tb):
        self.raw.close()


class BridgeTests(unittest.TestCase):
    def test_disabled_without_flag(self):
        self.assertFalse(bridge.enabled({}))
        self.assertFalse(bridge.enabled({bridge.ENABLE_VARIABLE: "0"}))

    def test_only_explicit_one_enables(self):
        self.assertTrue(bridge.enabled({bridge.ENABLE_VARIABLE: "1"}))
        self.assertFalse(bridge.enabled({bridge.ENABLE_VARIABLE: "true"}))

    def test_disabled_health_does_not_call_network(self):
        with patch.dict(os.environ, {bridge.ENABLE_VARIABLE: "0"}):
            with patch.object(bridge, "urlopen", side_effect=AssertionError("network called")):
                self.assertFalse(bridge.read_health()["ready"])

    def test_healthy_local_service(self):
        with patch.dict(os.environ, {bridge.ENABLE_VARIABLE: "1"}):
            with patch.object(
                bridge, "urlopen",
                return_value=MockResponse(
                    {"ready": True, "independent": True, "context": 128}
                ),
            ):
                self.assertTrue(bridge.available())

    def test_not_independent_not_ready(self):
        with patch.dict(os.environ, {bridge.ENABLE_VARIABLE: "1"}):
            with patch.object(
                bridge, "urlopen",
                return_value=MockResponse({"ready": True, "independent": False}),
            ):
                self.assertFalse(bridge.available())

    def test_generation_disabled_by_default(self):
        with patch.dict(os.environ, {bridge.ENABLE_VARIABLE: "0"}):
            with self.assertRaises(bridge.BrainBridgeError):
                bridge.generate("S")

    def test_generation_sends_only_loopback_payload(self):
        captured = []

        def respond(request, timeout):
            captured.append((request.full_url, json.loads(request.data)))
            return MockResponse({"response": "answer", "usage": {"generated_tokens": 6}})

        with patch.dict(os.environ, {bridge.ENABLE_VARIABLE: "1"}):
            with patch.object(bridge, "urlopen", side_effect=respond):
                result = bridge.generate("S", max_new_tokens=6)
        self.assertEqual(result["response"], "answer")
        self.assertTrue(result["experimental"])
        self.assertEqual(captured[0][0], "http://127.0.0.1:8766/generate")
        self.assertTrue(captured[0][1]["use_cache"])

    def test_missing_response_is_error(self):
        with patch.dict(os.environ, {bridge.ENABLE_VARIABLE: "1"}):
            with patch.object(
                bridge, "urlopen", return_value=MockResponse({"unexpected": True})
            ):
                with self.assertRaises(bridge.BrainBridgeError):
                    bridge.generate("hello")

    def test_unrecognized_path_refused(self):
        with self.assertRaises(ValueError):
            bridge._request("/danger")

    def test_large_prompt_refused(self):
        with patch.dict(os.environ, {bridge.ENABLE_VARIABLE: "1"}):
            with self.assertRaises(ValueError):
                bridge.generate("x" * 16001)

    def test_nonempty_prompt_required(self):
        with patch.dict(os.environ, {bridge.ENABLE_VARIABLE: "1"}):
            with self.assertRaises(ValueError):
                bridge.generate("")

    def test_configuration_has_no_cloud(self):
        summary = bridge.configuration_summary({bridge.ENABLE_VARIABLE: "1"})
        self.assertFalse(summary["requires_ollama"])
        self.assertFalse(summary["automatic_fallback"])
        self.assertFalse(summary["public_internet"])

    def test_engine_guard_refuses_disabled_brain(self):
        from continuum import engine
        class Store:
            def get_session(self, sid):
                return {"project_id": "p"}
            def add_message(self, *args, **kwargs):
                return {"role": args[1], "content": args[2]}
        with patch.dict(os.environ, {bridge.ENABLE_VARIABLE: "0"}):
            with self.assertRaises(engine.LocalModelError):
                engine.chat(Store(), "s", "hello", model=bridge.MODEL)


if __name__ == "__main__":
    unittest.main()
