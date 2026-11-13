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
        "import app as solution",
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
    return "
".join(lines)


def prepare_benchmark(candidate: str, *, suite: str = "basic") -> dict:
    """Prepare files for the existing permissioned Docker runner."""
    if suite not in ("basic", "practical"):
        raise ValueError("Unknown benchmark suite")
    if suite == "practical":
        from .practical_benchmark import practical_test_source, PRACTICAL_CASES
        test_source = practical_test_source()
        count = len(PRACTICAL_CASES)
    else:
        test_source = generate_test_source()
        count = len(CASES)
    review = review_files({"solution.py": candidate})
    return {
        "files": {
            "app.py": candidate,
            "test_app.py": test_source,
        },
        "review": review,
        "case_count": count,
        "status": "ready-for-isolated-execution" if review["syntax_passed"] else "syntax-error",
        "executed": False,
    }


def score_unittest_output(output: str, *, expected_tests: int = len(CASES)) -> dict:
    """Parse a Python unittest summary, without treating absence as success."""
    if type(expected_tests) is not int or not 1 <= expected_tests <= 1000:
        raise ValueError("Invalid expected test count")
    if not isinstance(output, str) or len(output) > 1_000_000:
        raise ValueError("Invalid test output")
    match = re.search(r"Ran (\d+) tests? in [\d.]+s", output)
    passed = bool(re.search(r"\bOK\b", output)) and not re.search(
        r"\bFAILED\b|\bERROR\b", output
    )
    count = int(match.group(1)) if match else 0
    return {
        "reported_tests": count,
        "suite_passed": passed and count == expected_tests,
        "expected_tests": expected_tests,
        "source": "unittest-output",
    }


def run_benchmark(candidate: str, *, images: dict[str, str], approved: bool = False, suite: str = "basic") -> dict:
    """Execute candidate against independent cases using the existing DockerRunner.

    The caller must explicitly authorize execution and supply a pinned Python
    image through the established S Developer runtime-preparation workflow.
    """
    if not approved:
        raise PermissionError("Explicit benchmark execution approval required")
    prepared = prepare_benchmark(candidate, suite=suite)
    if prepared["status"] != "ready-for-isolated-execution":
        return {"status": "syntax-error", "passed": False,
                "review": prepared["review"], "executed": False}
    from s_developer.sandbox import DockerRunner
    execution = DockerRunner(images).run(prepared["files"], "python-cli")
    scoring = score_unittest_output(execution.get("output", ""), expected_tests=prepared["case_count"])
    return {
        "status": execution["status"],
        "passed": execution["status"] == "toolchain_passed" and scoring["suite_passed"],
        "scoring": scoring,
        "isolation": execution.get("isolation"),
        "executed": True,
    }
