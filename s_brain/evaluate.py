"""Reproducible language-model evaluation and task-level benchmark reports.

This module distinguishes statistical language-model quality (cross-entropy
and perplexity) from correctness on explicitly labeled questions. A model
trained on a small corpus does not automatically become a useful assistant.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class BenchmarkCase:
    identifier: str
    prompt: str
    expected: str
    category: str
    source: str

    @classmethod
    def from_dict(cls, row: dict) -> "BenchmarkCase":
        for key in ("id", "prompt", "expected", "category", "source"):
            if key not in row:
                raise ValueError(f"Benchmark missing {key}")
            if not row["id"] or not isinstance(row["prompt"], str):
                raise ValueError("Invalid benchmark case")
        return cls(
            identifier=str(row["id"]),
            prompt=row["prompt"],
            expected=str(row["expected"]),
            category=str(row["category"]),
            source=str(row["source"]),
        )


@dataclass(frozen=True)
class CaseResult:
    identifier: str
    category: str
    passed: bool
    answer: str
    expected: str
    latency_ms: float
    exact_match: bool
    normalized_match: bool
    keyword_recall: float


def normalized(text: str) -> str:
    import re
    return " ".join(re.findall(r"[\w]+", text.casefold()))


def keyword_recall(expected: str, actual: str) -> float:
    target = set(normalized(expected).split())
    if not target:
        return float(not normalized(actual))
    actual_tokens = set(normalized(actual).split())
    return len(target & actual_tokens) / len(target)


def score_answer(expected: str, actual: str,
                 strategy: str = "normalized") -> tuple[bool, dict]:
    if strategy not in ("exact", "normalized", "contains"):
        raise ValueError("Unsupported scoring strategy")
    direct = expected == actual
    normal = normalized(expected) == normalized(actual)
    contains = bool(normalized(expected)) and normalized(expected) in normalized(actual)
    passed = {"exact": direct, "normalized": normal, "contains": contains}[strategy]
    return passed, {
        "exact_match": direct,
        "normalized_match": normal,
        "contains_match": contains,
        "keyword_recall": keyword_recall(expected, actual),
    }


def load_cases(path: Path) -> list[BenchmarkCase]:
    cases = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                case = BenchmarkCase.from_dict(json.loads(line))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            if case.identifier in seen:
                raise ValueError(f"Duplicate benchmark ID: {case.identifier}")
            seen.add(case.identifier)
            cases.append(case)
    if not cases:
        raise ValueError("Benchmark must contain cases")
    return cases


def save_cases(path: Path, cases: Iterable[BenchmarkCase]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as out:
        for case in cases:
            value = asdict(case)
            value["id"] = value.pop("identifier")
            out.write(json.dumps(value, ensure_ascii=False) + "\n")
            n += 1
    return n


def case_fingerprint(cases: Iterable[BenchmarkCase]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        digest.update(json.dumps(asdict(case), sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def run_cases(
    cases: Iterable[BenchmarkCase],
    generate: Callable[[str], str],
    strategy: str = "normalized",
) -> tuple[list[CaseResult], dict]:
    rows = []
    for case in cases:
        start = time.perf_counter()
        try:
            answer = generate(case.prompt)
        except Exception as exc:
            answer = f"[GENERATION_ERROR:{type(exc).__name__}]"
        elapsed = (time.perf_counter() - start) * 1000
        passed, detail = score_answer(case.expected, answer, strategy)
        rows.append(CaseResult(
            case.identifier, case.category, passed, answer,
            case.expected, elapsed, detail["exact_match"],
            detail["normalized_match"], detail["keyword_recall"],
        ))
    return rows, summarize_cases(rows)


def summarize_cases(rows: Iterable[CaseResult]) -> dict:
    rows = list(rows)
    if not rows:
        raise ValueError("No evaluated cases")
    by_category = {}
    for category in sorted({row.category for row in rows}):
        items = [row for row in rows if row.category == category]
        by_category[category] = {
            "count": len(items),
            "correct": sum(row.passed for row in items),
            "accuracy": sum(row.passed for row in items) / len(items),
        }
    latencies = sorted(row.latency_ms for row in rows)
    p95 = latencies[min(len(latencies)-1, math.ceil(0.95*len(latencies))-1)]
    return {
        "count": len(rows),
        "correct": sum(row.passed for row in rows),
        "accuracy": sum(row.passed for row in rows) / len(rows),
        "mean_keyword_recall": statistics.mean(row.keyword_recall for row in rows),
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": p95,
        "by_category": by_category,
    }


def bootstrap_accuracy_ci(rows: list[CaseResult], seed: int = 42,
                          draws: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    """Nonparametric bootstrap estimate, not a guarantee of population accuracy."""
    import random
    if not rows or draws < 2 or not 0 < alpha < 1:
        raise ValueError("Invalid bootstrap inputs")
    rng = random.Random(seed)
    outcomes = [int(row.passed) for row in rows]
    samples = sorted(
        sum(rng.choice(outcomes) for _ in outcomes) / len(outcomes)
        for _ in range(draws)
    )
    left = max(0, int((alpha / 2) * draws))
    right = min(draws-1, int((1-alpha/2)*draws))
    return samples[left], samples[right]


@torch.inference_mode()
def evaluate_token_stream(model, tokens: torch.Tensor,
                          context: int | None = None,
                          device: str = "cpu") -> dict:
    """Score every next token exactly once with a sliding nonoverlap target.

    Earlier tokens are reprocessed as context; loss is weighted by target count.
    This is deliberately slower than a sophisticated batched evaluator.
    """
    if tokens.ndim != 1 or len(tokens) < 2:
        raise ValueError("Expected at least 2 token IDs")
    model.eval()
    window = context or model.cfg.context
    if not 2 <= window <= model.cfg.context:
        raise ValueError("Invalid context window")
    total_loss = 0.0
    total_tokens = 0
    start = time.perf_counter()
    for target_position in range(1, len(tokens)):
        first = max(0, target_position-window)
        segment = tokens[first:target_position+1].to(device)
        inputs = segment[:-1][None, :]
        target = segment[-1].reshape(1)
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits[:, -1, :].float(), target, reduction="sum")
        total_loss += float(loss)
        total_tokens += 1
    mean_loss = total_loss / total_tokens
    return {
        "token_count": total_tokens,
        "nll": mean_loss,
        "perplexity": math.exp(mean_loss) if mean_loss < 700 else float("inf"),
        "elapsed_seconds": time.perf_counter() - start,
        "device": device,
        "note": "Teacher-forced next-token metric, not a reasoning score",
    }


@torch.inference_mode()
def evaluate_byte_file(model, path: Path,
                       max_bytes: int = 25000,
                       device: str = "cpu") -> dict:
    if max_bytes < 2:
        raise ValueError("max_bytes too small")
    raw = path.read_bytes()[:max_bytes]
    if len(raw) < 2:
        raise ValueError("Byte file has too little content")
    tokens = torch.tensor(list(raw), dtype=torch.long)
    report = evaluate_token_stream(model, tokens, device=device)
    report["source"] = str(path)
    report["evaluated_bytes"] = len(raw)
    report["sha256"] = hashlib.sha256(raw).hexdigest()
    return report


def compare_reports(baseline: dict, candidate: dict,
                    metric: str, lower_is_better: bool) -> dict:
    if metric not in baseline or metric not in candidate:
        raise ValueError("Metric missing from comparison")
    a, b = float(baseline[metric]), float(candidate[metric])
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Metrics must be finite")
    change = b-a
    relative = None if a == 0 else change/abs(a)
    improved = change < 0 if lower_is_better else change > 0
    return {
        "metric": metric,
        "baseline": a,
        "candidate": b,
        "absolute_delta": change,
        "relative_delta": relative,
        "improved": improved,
        "note": "Improvement on one metric does not prove general intelligence",
    }


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sensitivity_by_category(rows: list[CaseResult]) -> dict:
    """Find the lowest-performing evaluated categories for training priorities."""
    summary = summarize_cases(rows)
    ranking = sorted(
        summary["by_category"].items(),
        key=lambda item: (item[1]["accuracy"], -item[1]["count"]),
    )
    return {"priorities": [name for name, _ in ranking], "metrics": dict(ranking)}


def regression_gate(baseline: dict, candidate: dict,
                    max_accuracy_drop: float = 0.02,
                    max_p95_latency_increase: float = 0.5) -> dict:
    if max_accuracy_drop < 0 or max_p95_latency_increase < 0:
        raise ValueError("Negative tolerance")
    accuracy_ok = candidate["accuracy"] >= baseline["accuracy"] - max_accuracy_drop
    base_latency = baseline["p95_latency_ms"]
    latency_ok = candidate["p95_latency_ms"] <= base_latency*(1+max_p95_latency_increase)
    return {
        "passed": accuracy_ok and latency_ok,
        "accuracy_ok": accuracy_ok,
        "latency_ok": latency_ok,
        "accuracy_delta": candidate["accuracy"]-baseline["accuracy"],
        "p95_latency_delta_ms": candidate["p95_latency_ms"]-base_latency,
    }
