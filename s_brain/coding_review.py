"""Static Python coding feedback for S-Brain's iterative coding loop.

The analyzer never executes candidate code. It provides actionable diagnostics
for syntax, unresolved imports, risky calls, and structural complexity.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Diagnostic:
    path: str
    line: int
    severity: str
    rule: str
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


DANGEROUS_CALLS = {
    "eval": "Avoid evaluating untrusted strings",
    "exec": "Avoid executing dynamic source strings",
    "os.system": "Avoid shell command strings",
    "pickle.loads": "Untrusted pickle data can execute code",
    "subprocess.Popen": "Review subprocess invocation and input boundaries",
}


def qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def inspect_python(path: str, source: str) -> list[Diagnostic]:
    if not isinstance(source, str) or len(source.encode("utf-8")) > 1_000_000:
        raise ValueError("Python source exceeds review budget")
    findings: list[Diagnostic] = []
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return [Diagnostic(path, exc.lineno or 1, "error", "PY_SYNTAX", exc.msg)]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = qualified_name(node.func)
            if name in DANGEROUS_CALLS:
                findings.append(Diagnostic(path, node.lineno, "warning", "PY_REVIEW_CALL", DANGEROUS_CALLS[name]))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            branches = sum(isinstance(item, (ast.If, ast.For, ast.While, ast.Try, ast.Match))
                           for item in ast.walk(node))
            if branches > 15:
                findings.append(Diagnostic(path, node.lineno, "advice", "PY_COMPLEXITY",
                                           f"Function contains {branches} branch structures; consider decomposition"))
    return sorted(findings, key=lambda item: (item.line, item.rule))


def review_files(files: Mapping[str, str]) -> dict:
    if len(files) > 100:
        raise ValueError("Maximum 100 source files per review")
    findings: list[Diagnostic] = []
    for name, source in sorted(files.items()):
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe source path")
        if name.endswith(".py"):
            findings.extend(inspect_python(name, source))
    errors = sum(item.severity == "error" for item in findings)
    return {"files_reviewed": len(files), "syntax_passed": errors == 0,
            "errors": errors, "diagnostics": [item.as_dict() for item in findings],
            "executed": False, "review_type": "static-ast"}
