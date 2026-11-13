import unittest
from unittest.mock import patch

from s_brain.coding_agent import improve_candidate


class CodingAgentTests(unittest.TestCase):
    def test_explicit_approval(self):
        with self.assertRaises(PermissionError):
            improve_candidate("pass", images={})

    def test_success_stops_after_one_attempt(self):
        with patch("s_brain.coding_agent.run_benchmark") as run:
            run.return_value = {"status": "toolchain_passed", "passed": True,
                                "scoring": {"reported_tests": 9}}
            result = improve_candidate("pass", images={}, approved=True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["attempt_count"], 1)

    def test_failed_candidate_revision_and_success(self):
        source = "def add(a,b): return 0"
        with patch("s_brain.coding_agent.run_benchmark") as run:
            run.side_effect = [
                {"status": "failed", "passed": False, "scoring": {"reported_tests": 9}},
                {"status": "toolchain_passed", "passed": True, "scoring": {"reported_tests": 9}},
            ]
            result = improve_candidate(source, images={}, approved=True,
                                       revise=lambda old, result: "def add(a,b): return a+b")
        self.assertTrue(result["passed"])
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual(result["attempts"][1]["revision_origin"], "revision-callback")

    def test_no_reviser_stops_on_failure(self):
        with patch("s_brain.coding_agent.run_benchmark") as run:
            run.return_value = {"status": "failed", "passed": False}
            result = improve_candidate("def add(a,b): return 0", images={}, approved=True)
        self.assertFalse(result["passed"])
        self.assertEqual(result["attempt_count"], 1)

    def test_static_repair(self):
        with patch("s_brain.coding_agent.run_benchmark") as run:
            run.side_effect = [
                {"status": "syntax-error", "passed": False},
                {"status": "toolchain_passed", "passed": True, "scoring": {"reported_tests": 9}},
            ]
            result = improve_candidate("def add(a,b)\n    return a+b\n", images={}, approved=True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["attempts"][1]["revision_origin"], "static-syntax-repair")

    def test_final_source_is_last_tested_source(self):
        with patch("s_brain.coding_agent.run_benchmark") as run:
            run.return_value = {"status": "failed", "passed": False}
            result = improve_candidate(
                "def add(a,b): return 0", images={}, approved=True,
                revise=lambda old, feedback: "def add(a,b): return a+b",
                max_attempts=1,
            )
        self.assertEqual(result["final_source"], "def add(a,b): return 0")
        self.assertEqual(result["attempt_count"], 1)

    def test_invalid_attempt_budget(self):
        with self.assertRaises(ValueError):
            improve_candidate("pass", images={}, approved=True, max_attempts=0)


if __name__ == "__main__":
    unittest.main()
