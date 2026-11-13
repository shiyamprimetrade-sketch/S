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
        self.assertIn("test_solution.py", result["files"])

    def test_syntax_failure(self):
        result = prepare_benchmark("def add(:")
        self.assertEqual(result["status"], "syntax-error")

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
