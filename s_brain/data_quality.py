"""Independent data-quality gates for S-Brain training experiments.

Data quality is not a substitute for scale. It is, however, a testable
requirement: exact duplicates, short documents, split leakage, private-data
patterns and repetitive synthetic examples must be visible before training.

This module analyzes in-memory text or a corpus manifest without uploading it.
Only aggregate statistics and hashes belong in public CI reports. Raw personal
materials remain outside GitHub and are never automatically collected.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

from .dataset import CorpusRecord, read_jsonl


@dataclasses.dataclass(frozen=True)
class QualityPolicy:
    minimum_documents: int = 10
    minimum_unique_documents: int = 8
    minimum_total_bytes: int = 1024
    minimum_document_bytes: int = 20
    max_exact_duplicate_fraction: float = 0.02
    max_short_fraction: float = 0.10
    max_repeated_3gram_fraction: float = 0.95
    require_train_validation_test: bool = True
    reject_sensitive_patterns: bool = True

    def validate(self):
        if any(
            value < 0 for value in (
                self.minimum_documents, self.minimum_unique_documents,
                self.minimum_total_bytes, self.minimum_document_bytes
            )
        ):
            raise ValueError("Minimum thresholds cannot be negative")
        for name in (
            "max_exact_duplicate_fraction",
            "max_short_fraction",
            "max_repeated_3gram_fraction",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be a fraction")


@dataclasses.dataclass(frozen=True)
class QualityFinding:
    category: str
    passed: bool
    observed: float | int
    threshold: float | int
    explanation: str


@dataclasses.dataclass
class QualityReport:
    records: int
    unique_records: int
    total_bytes: int
    split_counts: dict[str, int]
    findings: list[QualityFinding]

    @property
    def accepted(self) -> bool:
        return all(item.passed for item in self.findings)

    def as_dict(self) -> dict:
        return {
            "records": self.records,
            "unique_records": self.unique_records,
            "total_bytes": self.total_bytes,
            "split_counts": self.split_counts,
            "accepted": self.accepted,
            "findings": [
                dataclasses.asdict(item) for item in self.findings
            ],
            "note": (
                "Data screening is statistical and cannot prove permission, "
                "authorship, privacy or intellectual-property compliance"
            ),
        }


SECRET_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|password|access[_-]?token|private[_-]?key)"
    r"\s*(?:=|:)\s*\S+"
)
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")


def sentence_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def ngram_fraction(text: str, n: int = 3) -> float:
    """Proportion of repeated n-gram occurrences, useful for loops."""
    if n < 1:
        raise ValueError("n must be positive")
    tokens = sentence_tokens(text)
    count = max(0, len(tokens) - n + 1)
    if count == 0:
        return 0.0
    observed = [
        tuple(tokens[index:index+n]) for index in range(count)
    ]
    return 1 - len(set(observed)) / len(observed)


def shannon_entropy(data: bytes) -> float:
    """Observed byte entropy, in bits per byte, without model assumptions."""
    if not data:
        return 0.0
    counts = collections.Counter(data)
    size = len(data)
    return -sum(
        (count / size) * math.log2(count / size)
        for count in counts.values()
    )


def duplicate_fraction(records: Sequence[CorpusRecord]) -> float:
    if not records:
        return 0.0
    return 1 - len({item.sha256 for item in records}) / len(records)


def short_document_fraction(records: Sequence[CorpusRecord],
                            minimum: int) -> float:
    if not records:
        return 0.0
    return sum(row.bytes_utf8 < minimum for row in records) / len(records)


def find_sensitive_indicators(records: Iterable[CorpusRecord]) -> dict:
    """Count *possible* secret/email patterns without including matched values."""
    secret = 0
    emails = 0
    for row in records:
        secret += len(SECRET_PATTERN.findall(row.text))
        emails += len(EMAIL_PATTERN.findall(row.text))
    return {
        "possible_secret_assignments": secret,
        "possible_emails": emails,
    }


def detect_cross_split_duplicates(records: Iterable[CorpusRecord]) -> list[str]:
    first_split = {}
    violations = []
    for row in records:
        prior = first_split.setdefault(row.sha256, row.split)
        if prior != row.split:
            violations.append(row.identifier)
    return violations


def evaluate_quality(
    records: Sequence[CorpusRecord],
    policy: QualityPolicy | None = None,
) -> QualityReport:
    policy = policy or QualityPolicy()
    policy.validate()
    rows = list(records)
    splits = collections.Counter(row.split for row in rows)
    unique_count = len({row.sha256 for row in rows})
    total_bytes = sum(row.bytes_utf8 for row in rows)
    exact_fraction = duplicate_fraction(rows)
    short_fraction = short_document_fraction(
        rows, policy.minimum_document_bytes
    )
    overlap = detect_cross_split_duplicates(rows)
    sensitive = find_sensitive_indicators(rows)
    repeated = ngram_fraction(
        " ".join(row.text for row in rows), 3
    )
    findings = [
        QualityFinding(
            "document_count", len(rows) >= policy.minimum_documents,
            len(rows), policy.minimum_documents,
            "Source documents included in the audited manifest",
        ),
        QualityFinding(
            "unique_documents", unique_count >= policy.minimum_unique_documents,
            unique_count, policy.minimum_unique_documents,
            "Distinct SHA-256 document contents",
        ),
        QualityFinding(
            "total_bytes", total_bytes >= policy.minimum_total_bytes,
            total_bytes, policy.minimum_total_bytes,
            "Combined UTF-8 document size",
        ),
        QualityFinding(
            "exact_duplicates",
            exact_fraction <= policy.max_exact_duplicate_fraction,
            exact_fraction, policy.max_exact_duplicate_fraction,
            "Fraction of duplicated content hashes",
        ),
        QualityFinding(
            "short_documents", short_fraction <= policy.max_short_fraction,
            short_fraction, policy.max_short_fraction,
            "Fraction of documents below the minimum length",
        ),
        QualityFinding(
            "repeated_ngrams",
            repeated <= policy.max_repeated_3gram_fraction,
            repeated, policy.max_repeated_3gram_fraction,
            "Repeated 3-gram fraction across corpus text",
        ),
        QualityFinding(
            "cross_split_overlap", not overlap, len(overlap), 0,
            "Exact content appearing in multiple splits",
        ),
        QualityFinding(
            "split_coverage",
            not policy.require_train_validation_test
            or all(splits[name] > 0 for name in ("train", "validation", "test")),
            sum(splits[name] > 0 for name in ("train", "validation", "test")),
            3 if policy.require_train_validation_test else 0,
            "Train/validation/test population",
        ),
        QualityFinding(
            "obvious_sensitive_patterns",
            not policy.reject_sensitive_patterns or sum(sensitive.values()) == 0,
            sum(sensitive.values()), 0,
            "Possible email addresses and assigned secrets; manual review still needed",
        ),
    ]
    return QualityReport(
        records=len(rows),
        unique_records=unique_count,
        total_bytes=total_bytes,
        split_counts=dict(splits),
        findings=findings,
    )


def inspect_manifest(
    path: Path,
    policy: QualityPolicy | None = None,
) -> dict:
    rows = list(read_jsonl(path))
    report = evaluate_quality(rows, policy)
    data = report.as_dict()
    data["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return data


def compare_quality(before: QualityReport, after: QualityReport) -> dict:
    previous = {item.category: item for item in before.findings}
    current = {item.category: item for item in after.findings}
    return {
        "accepted_before": before.accepted,
        "accepted_after": after.accepted,
        "documents_delta": after.records - before.records,
        "unique_delta": after.unique_records - before.unique_records,
        "newly_failing": [
            name for name in previous
            if previous[name].passed and not current[name].passed
        ],
        "newly_passing": [
            name for name in previous
            if not previous[name].passed and current[name].passed
        ],
    }


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(
        description="Screen an approved S-Brain corpus manifest"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--min-documents", type=int, default=10)
    parser.add_argument("--min-bytes", type=int, default=1024)
    args = parser.parse_args(argv)
    report = inspect_manifest(
        args.manifest,
        QualityPolicy(
            minimum_documents=args.min_documents,
            minimum_unique_documents=min(args.min_documents, 8),
            minimum_total_bytes=args.min_bytes,
        ),
    )
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
