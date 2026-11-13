"""Bounded generate-test-repair loop for S-Brain coding candidates.

Caller supplies a revision function (neural model or approved provider).
This orchestration does not fabricate a model, execute host code, or claim
semantic correctness without a passing isolated benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable

from .coding_benchmark import run_benchmark
from .coding_cycle import repair_missing_colon


@dataclass(frozen=True)
class Attempt:
    number: int
    source_sha256: str
    status: str
    passed: bool
    revision_origin: str
    reported_tests: int


def improve_candidate(
    initial_source: str,
    *,
    images: dict[str, str],
    approved: bool = False,
    revise: Callable[[str, dict], str] | None = None,
    max_attempts: int = 3,
) -> dict:
    if not approved:
        raise PermissionError("Explicit coding execution approval required")
    if not isinstance(initial_source, str) or len(initial_source.encode("utf-8")) > 64000:
        raise ValueError("Candidate source exceeds 64 KB")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 5:
        raise ValueError("Attempt budget must be 1..5")
    source = initial_source
    history: list[Attempt] = []
    seen: set[str] = set()
    origin = "initial"
    for number in range(1, max_attempts + 1):
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if digest in seen:
            break
        seen.add(digest)
        result = run_benchmark(source, images=images, approved=True)
        score = result.get("scoring", {})
        passed = result.get("passed") is True
        history.append(Attempt(number, digest, result["status"], passed,
                               origin, score.get("reported_tests", 0)))
        if passed:
            break
        # Static syntax repair is only a first fallback. A capable model can
        # supply its own revised source through the optional callback.
        repaired, change = repair_missing_colon("app.py", source)
        if change is not None and repaired != source:
            next_source = repaired
            next_origin = "static-syntax-repair"
        elif revise is not None:
            next_source = revise(source, result)
            next_origin = "revision-callback"
        else:
            break
        if not isinstance(next_source, str) or len(next_source.encode("utf-8")) > 64000:
            raise ValueError("Revised candidate exceeds 64 KB")
        if next_source == source:
            break
        if number == max_attempts:
            break
        source = next_source
        origin = next_origin
    return {
        "passed": bool(history and history[-1].passed),
        "attempts": [vars(attempt) for attempt in history],
        "attempt_count": len(history),
        "final_source": source,
        "final_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "benchmark": "nine-independent-python-cases-v1",
        "execution": "pinned-docker",
    }
