"""S-Brain coding cycle: static review, bounded syntax repair, and test scaffolding.

Candidate source is never executed here. Repairs are conservative and only
apply when Python's parser identifies a missing colon on a block header.
"""
from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Mapping

from .coding_review import inspect_python, review_files

BLOCK_PREFIXES = (
    "def ", "async def ", "class ", "if ", "elif ", "else",
    "for ", "async for ", "while ", "try", "except", "finally",
    "with ", "async with ", "match ", "case ",
)


@dataclass(frozen=True)
class Repair:
    path: str
    line: int
    description: str
    before: str
    after: str


def repair_missing_colon(path: str, source: str) -> tuple[str, Repair | None]:
    try:
        ast.parse(source, filename=path)
        return source, None
    except SyntaxError as error:
        if error.lineno is None or "expected ':'" not in error.msg:
            return source, None
        lines = source.splitlines(keepends=True)
        index = error.lineno - 1
        if not 0 <= index < len(lines):
            return source, None
        line = lines[index]
        body = line.rstrip("\r\n")
        stripped = body.lstrip()
        if not any(stripped.startswith(prefix) for prefix in BLOCK_PREFIXES):
            return source, None
        if "#" in body or body.endswith(":") or body.endswith("\\"):
            return source, None
        replacement = body + ":" + line[len(body):]
        candidate = list(lines)
        candidate[index] = replacement
        updated = "".join(candidate)
        try:
            ast.parse(updated, filename=path)
        except SyntaxError:
            return source, None
        return updated, Repair(path, error.lineno, "Inserted missing block colon", line, replacement)


def repair_files(files: Mapping[str, str], *, max_repairs: int = 2) -> dict:
    if type(max_repairs) is not int or not 0 <= max_repairs <= 10:
        raise ValueError("Repair budget must be 0..10")
    before = review_files(files)
    revised = dict(files)
    repairs: list[Repair] = []
    for name in sorted(revised):
        if not name.endswith(".py"):
            continue
        for _ in range(max_repairs - len(repairs)):
            updated, change = repair_missing_colon(name, revised[name])
            if change is None:
                break
            revised[name] = updated
            repairs.append(change)
        if len(repairs) >= max_repairs:
            break
    after = review_files(revised)
    return {
        "files": revised,
        "before": before,
        "after": after,
        "repairs": [asdict(change) for change in repairs],
        "executed": False,
        "repair_count": len(repairs),
    }


def generate_smoke_tests(module_name: str, source: str) -> str:
    """Generate import/compilation smoke tests, not claims of semantic correctness."""
    if not module_name.isidentifier():
        raise ValueError("Module name must be a single Python identifier")
    ast.parse(source, filename=module_name + ".py")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    # Compile-only avoids side effects of importing untrusted generated modules.
    return (
        "import ast\nimport hashlib\nfrom pathlib import Path\n\n"
        "def test_source_compiles():\n"
        f"    path = Path(__file__).resolve().parent / {module_name + '.py'!r}\n"
        "    source = path.read_text(encoding='utf-8')\n"
        f"    assert hashlib.sha256(source.encode('utf-8')).hexdigest() == {digest!r}\n"
        "    ast.parse(source, filename=str(path))\n"
    )


def cycle_report(files: Mapping[str, str], *, max_repairs: int = 2) -> dict:
    result = repair_files(files, max_repairs=max_repairs)
    return {
        "review": result["after"],
        "repair_count": result["repair_count"],
        "repairs": result["repairs"],
        "changed_paths": sorted(name for name in files if files[name] != result["files"][name]),
        "verified_by_execution": False,
        "candidate_files": result["files"],
    }
