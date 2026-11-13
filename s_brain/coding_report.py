"""Evidence-preserving coding benchmark reports.

Reports distinguish mocked tests, static checks and real Docker executions.
No inferred pass counts or fabricated runtime measurements.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def summarize_cycle(outcome: dict, *, provider: dict | None = None) -> dict:
    attempts = outcome.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("Cannot report a cycle without recorded attempts")
    for index, attempt in enumerate(attempts, 1):
        if attempt.get("number") != index:
            raise ValueError("Attempt numbering is inconsistent")
        digest = attempt.get("source_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Missing candidate SHA-256")
        if attempt.get("passed") not in (True, False):
            raise ValueError("Missing attempt result")
    if outcome.get("final_source_sha256") != attempts[-1]["source_sha256"]:
        raise ValueError("Final candidate must match the last tested attempt")
    if outcome.get("passed") is not attempts[-1]["passed"]:
        raise ValueError("Final result must match the last attempt")
    first = attempts[0]["passed"]
    final = attempts[-1]["passed"]
    return {
        "schema_version": 1,
        "benchmark": outcome.get("benchmark"),
        "execution": outcome.get("execution"),
        "attempt_count": len(attempts),
        "initial_passed": first,
        "final_passed": final,
        "repaired_to_pass": (not first and final),
        "attempts": attempts,
        "provider": provider,
        "evidence_type": "reported-execution-results; verify Docker logs independently",
        "general_programming_proficiency": "not established",
    }


def encode_report(report: dict) -> str:
    if report.get("schema_version") != 1:
        raise ValueError("Unknown report schema")
    return json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def report_fingerprint(report: dict) -> str:
    return hashlib.sha256(encode_report(report).encode("utf-8")).hexdigest()


def save_report(report: dict, destination: str | Path) -> dict:
    """Write a new report only; never overwrite earlier checkpoint evidence."""
    path = Path(destination)
    if path.suffix.lower() != ".json":
        raise ValueError("Report destination must end with .json")
    content = encode_report(report)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
    return {"path": str(path), "sha256": hashlib.sha256(content.encode()).hexdigest()}
