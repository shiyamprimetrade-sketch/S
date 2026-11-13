import ast
import unittest

from s_brain.coding_benchmark import (
    benchmark_spec, generate_test_source, prepare_benchmark, score_unittest_output,
)


class CodingBenchmarkTests(unittest.TestCase):
    def test_independent_cases(self):
        spec = benchmark_spec()
        self.assertEqual(spec["case_count"], 9)
        self.assertEqual(spec["tasks"], ["add", "factorial", "reverse_text"])

    def test_test_source_parses(self):
        source = generate_test_source()
        ast.parse(source)
        self.assertEqual(source.count("def test_"), 9)

    def test_prepare_does_not_execute(self):
        result = prepare_benchmark("def add(a,b): return a+b\n")
        self.assertEqual(result["status"], "ready-for-isolated-execution")
        self.assertFalse(result["executed"])
        self.assertIn("test_app.py", result["files"])

    def test_syntax_failure(self):
        result = prepare_benchmark("def add(:")
        self.assertEqual(result["status"], "syntax-error")

    def test_execution_requires_approval(self):
        from s_brain.coding_benchmark import run_benchmark
        with self.assertRaises(PermissionError):
            run_benchmark("def add(a,b): return a+b", images={})

    def test_execution_calls_isolated_runner(self):
        from unittest.mock import patch
        from s_brain.coding_benchmark import run_benchmark
        sample = "def add(a,b): return a+b\\ndef reverse_text(s): return s[::-1]\\ndef factorial(n):\\n    import math\\n    return math.factorial(n)\\n"
        with patch("s_developer.sandbox.DockerRunner.run") as run:
            run.return_value = {
                "status": "toolchain_passed",
                "output": "Ran 9 tests in 0.001s\\n\\nOK",
                "isolation": "docker-network-none-readonly",
            }
            result = run_benchmark(sample, images={}, approved=True)
        self.assertTrue(result["passed"])
        self.assertEqual(run.call_args.args[1], "python-cli")
        self.assertIn("app.py", run.call_args.args[0])
        self.assertIn("test_app.py", run.call_args.args[0])

    def test_score_pass(self):
        result = score_unittest_output(".........\nRan 9 tests in 0.001s\n\nOK\n")
        self.assertTrue(result["suite_passed"])

    def test_score_rejects_partial(self):
        result = score_unittest_output("Ran 2 tests in 0.001s\n\nOK\n")
        self.assertFalse(result["suite_passed"])

    def test_score_rejects_failure(self):
        result = score_unittest_output("Ran 9 tests in 0.001s\nFAILED (failures=1)")
        self.assertFalse(result["suite_passed"])

    def test_score_rejects_no_summary(self):
        self.assertFalse(score_unittest_output("OK")["suite_passed"])


if __name__ == "__main__":
    unittest.main()
