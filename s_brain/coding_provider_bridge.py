"""Adapt S Developer's configured neural coding provider to S-Brain benchmarks.

The provider produces candidate source; independent tests determine success.
No template fallback is silently presented as model-generated code.
"""
from __future__ import annotations

import json
from typing import Callable

from .coding_agent import improve_candidate


def extract_python_source(files: dict[str, str]) -> str:
    if not isinstance(files, dict) or "app.py" not in files:
        raise ValueError("Coding provider must return app.py")
    source = files["app.py"]
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Coding provider returned empty Python source")
    return source


def benchmark_with_provider(
    *,
    spec: dict,
    provider_id: str,
    config: dict,
    images: dict[str, str],
    approved: bool = False,
    max_attempts: int = 3,
    suite: str = "basic",
) -> dict:
    """Generate, test and optionally revise via the real configured provider."""
    if suite not in ("basic", "practical"):
        raise ValueError("Unknown benchmark suite")
    if not approved:
        raise PermissionError("Explicit provider benchmark approval required")
    if provider_id not in ("native", "native-copy", "ollama-external"):
        raise ValueError("Choose an explicitly configured neural coding provider")
    from s_developer.providers import provider
    engine = provider(provider_id, config)
    provenance = engine.provenance()
    files = engine.generate(spec, "python-cli")
    candidate = extract_python_source(files)

    def revise(source: str, result: dict) -> str:
        diagnostic = json.dumps({
            "benchmark_status": result.get("status"),
            "scoring": result.get("scoring", {}),
            "feedback": result.get("output", "")[:4000],
        }, ensure_ascii=False)
        revised_files = engine.repair(
            spec, "python-cli",
            {"app.py": source, "test_app.py": files["test_app.py"]},
            diagnostic,
        )
        return extract_python_source(revised_files)

    outcome = improve_candidate(
        candidate, images=images, approved=True,
        revise=revise, max_attempts=max_attempts,
        **({"suite": suite} if suite != "basic" else {}),
    )
    from .coding_report import summarize_cycle
    report = summarize_cycle(outcome, provider=provenance)
    return {
        "provider": provenance,
        "benchmark": outcome,
        "report": report,
        "claim": "Measured on the configured benchmark only; not general coding proficiency",
    }
