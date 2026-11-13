"""Independent, opt-in coding benchmark harness for S-Brain.

Runs candidate Python solutions in an isolated Docker container, not on the
host. Test cases are specified independently of generated candidate source.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Mapping

from .coding_review import review_files


@dataclass(frozen=True)
class CodingCase:
    name: str
    function: str
    arguments: tuple
    expected: object


CASES = (
    CodingCase("add_positive", "add", (2, 3), 5),
    CodingCase("add_negative", "add", (-8, 3), -5),
    CodingCase("add_zero", "add", (0, 0), 0),
    CodingCase("reverse_ascii", "reverse_text", ("abc",), "cba"),
    CodingCase("reverse_empty", "reverse_text", ("",), ""),
    CodingCase("reverse_unicode", "reverse_text", ("café",), "éfac"),
    CodingCase("factorial_zero", "factorial", (0,), 1),
    CodingCase("factorial_five", "factorial", (5,), 120),
    CodingCase("factorial_one", "factorial", (1,), 1),
)


def benchmark_spec() -> dict:
    return {
        "version": 1,
        "language": "python",
        "tasks": sorted({case.function for case in CASES}),
        "case_count": len(CASES),
        "execution": "Docker-only; explicitly enabled",
        "cases": [{"name": c.name, "function": c.function,
                   "arguments": list(c.arguments), "expected": c.expected} for c in CASES],
    }


def generate_test_source() -> str:
    """Construct an independent unittest suite for a candidate solution.py."""
    lines = [
        "import unittest",
        "import solution",
        "",
        "class CandidateTests(unittest.TestCase):",
    ]
    for case in CASES:
        if not re.fullmatch(r"[a-z_][a-z_0-9]*", case.name):
            raise ValueError("Invalid benchmark case name")
        lines.extend([
            f"    def test_{case.name}(self):",
            f"        self.assertEqual(getattr(solution, {case.function!r})(*{case.arguments!r}), {case.expected!r})",
            "",
        ])
    lines.extend(["if __name__ == '__main__':", "    unittest.main(verbosity=2)", ""])
    return "\n".join(lines)


def prepare_benchmark(candidate: str) -> dict:
    """Prepare files for the existing permissioned Docker runner."""
    review = review_files({"solution.py": candidate})
    return {
        "files": {
            "solution.py": candidate,
            "test_solution.py": generate_test_source(),
        },
        "review": review,
        "case_count": len(CASES),
        "status": "ready-for-isolated-execution" if review["syntax_passed"] else "syntax-error",
        "executed": False,
    }


def score_unittest_output(output: str) -> dict:
    """Parse a Python unittest summary, without treating absence as success."""
    if not isinstance(output, str) or len(output) > 1_000_000:
        raise ValueError("Invalid test output")
    match = re.search(r"Ran (\d+) tests? in [\d.]+s", output)
    passed = bool(re.search(r"\bOK\b", output)) and not re.search(
        r"\bFAILED\b|\bERROR\b", output
    )
    count = int(match.group(1)) if match else 0
    return {
        "reported_tests": count,
        "suite_passed": passed and count == len(CASES),
        "expected_tests": len(CASES),
        "source": "unittest-output",
    }
