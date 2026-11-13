import ast
import unittest

from s_brain.practical_benchmark import (
    PRACTICAL_CASES, practical_spec, practical_test_source,
)


class PracticalBenchmarkTests(unittest.TestCase):
    def test_case_count_and_skill_coverage(self):
        spec = practical_spec()
        self.assertEqual(spec["case_count"], 15)
        self.assertEqual(len(spec["functions"]), 5)

    def test_generated_suite_parses(self):
        source = practical_test_source()
        ast.parse(source)
        self.assertEqual(source.count("def test_"), 15)

    def test_csv_cases_use_real_newlines(self):
        cases = [case for case in PRACTICAL_CASES if case.name.startswith("csv_")]
        self.assertEqual(len(cases), 3)
        for case in cases:
            self.assertIn("\n", case.args[0])
            self.assertNotIn("\\n", case.args[0])

    def test_tests_import_candidate_not_reference_solution(self):
        source = practical_test_source()
        self.assertIn("import app as solution", source)
        self.assertNotIn("def shortest_path(", source)


if __name__ == "__main__":
    unittest.main()
