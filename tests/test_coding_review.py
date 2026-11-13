import unittest
from s_brain.coding_review import inspect_python, review_files


class CodingReviewTests(unittest.TestCase):
    def test_valid_code(self):
        result = review_files({"app.py": "def add(a,b):\n    return a+b\n"})
        self.assertTrue(result["syntax_passed"])
        self.assertFalse(result["executed"])

    def test_syntax_error(self):
        result = review_files({"bad.py": "def broken(:\n pass"})
        self.assertFalse(result["syntax_passed"])
        self.assertEqual(result["diagnostics"][0]["rule"], "PY_SYNTAX")

    def test_risky_calls(self):
        findings = inspect_python("app.py", "eval('2+2')\n")
        self.assertEqual(findings[0].rule, "PY_REVIEW_CALL")

    def test_no_execution(self):
        result = review_files({"app.py": "raise RuntimeError('not executed')"})
        self.assertTrue(result["syntax_passed"])

    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            review_files({"../bad.py": "pass"})

    def test_non_python_ignored(self):
        result = review_files({"index.html": "<html></html>"})
        self.assertEqual(result["diagnostics"], [])


if __name__ == "__main__":
    unittest.main()
