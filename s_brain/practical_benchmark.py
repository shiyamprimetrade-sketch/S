"""Practical programming challenge suite for independently evaluated Python code.

These tests are fixed outside the candidate source. Candidate implementations
are not supplied by this module; passing requires isolated execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class PracticalCase:
    name: str
    function: str
    args: tuple
    expected: object


PRACTICAL_CASES = (
    PracticalCase("dedupe_empty", "unique_ordered", ([],), []),
    PracticalCase("dedupe_order", "unique_ordered", ([3, 1, 3, 2, 1],), [3, 1, 2]),
    PracticalCase("dedupe_strings", "unique_ordered", (["a", "b", "a"],), ["a", "b"]),
    PracticalCase("word_count_basic", "word_counts", ("Red blue red",), {"red": 2, "blue": 1}),
    PracticalCase("word_count_punctuation", "word_counts", ("Hi, hi! Bye.",), {"hi": 2, "bye": 1}),
    PracticalCase("word_count_empty", "word_counts", ("",), {}),
    PracticalCase("csv_total_basic", "sum_csv_column", ("name,amount\na,3\nb,7\n", "amount"), 10),
    PracticalCase("csv_total_empty", "sum_csv_column", ("name,amount\n", "amount"), 0),
    PracticalCase("csv_total_negative", "sum_csv_column", ("amount\n-2\n5\n", "amount"), 3),
    PracticalCase("brackets_valid", "balanced_brackets", ("([]{})",), True),
    PracticalCase("brackets_invalid", "balanced_brackets", ("([)]",), False),
    PracticalCase("brackets_text", "balanced_brackets", ("a(b[c]{d})",), True),
    PracticalCase("shortest_basic", "shortest_path", ({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}, "a", "d"), ["a", "b", "d"]),
    PracticalCase("shortest_same", "shortest_path", ({"a": ["b"]}, "a", "a"), ["a"]),
    PracticalCase("shortest_unreachable", "shortest_path", ({"a": [], "b": []}, "a", "b"), None),
)


def practical_spec() -> dict:
    return {
        "version": 2,
        "case_count": len(PRACTICAL_CASES),
        "functions": sorted({case.function for case in PRACTICAL_CASES}),
        "contract": {
            "unique_ordered": "Return first occurrences in order; list elements are strings or integers.",
            "word_counts": "Count case-insensitive ASCII words separated by non-letter characters.",
            "sum_csv_column": "Parse CSV text with a header and integer column; return the sum.",
            "balanced_brackets": "Ignore non-bracket characters; validate (), [], {} nesting.",
            "shortest_path": "Return shortest node list in directed graph, BFS using neighbor list order; None if unreachable.",
        },
        "evaluation": "Independent Python unittest suite in pinned isolated Docker runtime",
    }


def practical_test_source() -> str:
    lines = ["import unittest", "import app as solution", "", "class PracticalTests(unittest.TestCase):"]
    for case in PRACTICAL_CASES:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", case.name):
            raise ValueError("Unsafe test case identifier")
        lines.extend([
            f"    def test_{case.name}(self):",
            f"        self.assertEqual(getattr(solution, {case.function!r})(*{case.args!r}), {case.expected!r})",
            "",
        ])
    return "\n".join(lines)
