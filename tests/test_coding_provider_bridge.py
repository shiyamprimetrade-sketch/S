import hashlib
import unittest
from unittest.mock import patch

from s_brain.coding_provider_bridge import benchmark_with_provider, extract_python_source


class ProviderBridgeTests(unittest.TestCase):
    def test_source_required(self):
        with self.assertRaises(ValueError):
            extract_python_source({"test_app.py": "pass"})

    def test_approval_required(self):
        with self.assertRaises(PermissionError):
            benchmark_with_provider(spec={}, provider_id="native", config={}, images={})

    def test_template_cannot_impersonate_neural_provider(self):
        with self.assertRaises(ValueError):
            benchmark_with_provider(spec={}, provider_id="template", config={},
                                    images={}, approved=True)

    def test_real_provider_adapter(self):
        spec = {"name": "Benchmark", "answers": {"purpose": "Solve coding tasks"}}
        with patch("s_developer.providers.provider") as factory, \
             patch("s_brain.coding_provider_bridge.improve_candidate") as loop:
            engine = factory.return_value
            engine.provenance.return_value = {"provider": "native"}
            engine.generate.return_value = {"app.py": "def add(a,b): return a+b",
                                            "test_app.py": "pass"}
            digest = hashlib.sha256(b"candidate").hexdigest()
            loop.return_value = {
                "passed": False,
                "attempts": [{"number": 1, "source_sha256": digest,
                              "passed": False, "status": "failed"}],
                "final_source_sha256": digest,
                "benchmark": "nine-independent-python-cases-v1",
                "execution": "pinned-docker",
            }
            outcome = benchmark_with_provider(
                spec=spec, provider_id="native", config={}, images={}, approved=True
            )
            self.assertEqual(outcome["provider"]["provider"], "native")
            self.assertFalse(outcome["benchmark"]["passed"])
            self.assertFalse(outcome["report"]["final_passed"])
            self.assertTrue(callable(loop.call_args.kwargs["revise"]))


if __name__ == "__main__":
    unittest.main()
