import ast
import unittest

from s_brain.coding_cycle import (
    cycle_report, generate_smoke_tests, repair_files, repair_missing_colon,
)


class CodingCycleTests(unittest.TestCase):
    def test_repair_colon(self):
        fixed, repair = repair_missing_colon("sample.py", "def hello()\n    return 3\n")
        self.assertIsNotNone(repair)
        self.assertIn("def hello():", fixed)
        ast.parse(fixed)

    def test_refuse_ambiguous(self):
        original = "def hello() # comment\n    return 3\n"
        fixed, repair = repair_missing_colon("sample.py", original)
        self.assertIsNone(repair)
        self.assertEqual(fixed, original)

    def test_no_changes_to_valid_code(self):
        files = {"a.py": "def f():\n    return 1\n"}
        result = repair_files(files)
        self.assertEqual(result["files"], files)
        self.assertEqual(result["repair_count"], 0)

    def test_report_tracks_repair(self):
        result = cycle_report({"a.py": "def f()\n    return 1\n"})
        self.assertEqual(result["changed_paths"], ["a.py"])
        self.assertTrue(result["review"]["syntax_passed"])
        self.assertFalse(result["verified_by_execution"])

    def test_smoke_test_has_digest_and_compile(self):
        source = "def f():\n    return 1\n"
        generated = generate_smoke_tests("sample", source)
        ast.parse(generated)
        self.assertIn("sha256", generated)
        self.assertIn("ast.parse", generated)

    def test_invalid_module_rejected(self):
        with self.assertRaises(ValueError):
            generate_smoke_tests("../bad", "pass")

    def test_repair_budget(self):
        with self.assertRaises(ValueError):
            repair_files({"a.py": "pass"}, max_repairs=-1)


if __name__ == "__main__":
    unittest.main()
