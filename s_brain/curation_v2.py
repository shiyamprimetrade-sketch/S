"""Curation of consented text, reviewed teacher examples and blind holdouts.

Pure Python, offline and repeatable. The output is a manifest of *metadata*
plus optional local text payloads explicitly exported by the operator.
No crawling, no network, no inference, no automatic license assumptions.
Exact and near-duplicate screening is deterministic and split-aware.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Iterator

SPLITS = ("train", "validation", "test")
WORD = re.compile(r"\w+", flags=re.UNICODE)
SECRET = re.compile(
    r"(?i)(?:api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*\S+"
)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


@dataclasses.dataclass(frozen=True)
class CurationPolicy:
    minimum_bytes: int = 24
    maximum_bytes: int = 32768
    near_duplicate_threshold: float = 0.90
    ngram_size: int = 3
    reject_possible_secrets: bool = True
    reject_possible_emails: bool = True
    reject_other_splits_on_overlap: bool = True
    require_documented_permission: bool = True

    def validate(self):
        if not 1 <= self.minimum_bytes < self.maximum_bytes:
            raise ValueError("Invalid length bounds")
        if not 0 < self.near_duplicate_threshold <= 1:
            raise ValueError("Invalid duplicate similarity threshold")
        if not 1 <= self.ngram_size <= 12:
            raise ValueError("Invalid ngram_size")


@dataclasses.dataclass(frozen=True)
class SourceRecord:
    identifier: str
    text: str
    source: str
    permission: str
    split: str
    category: str
    origin: str = "human"
    review: str = "pending"

    def validate(self):
        if not self.identifier or not self.identifier.strip():
            raise ValueError("Missing ID")
        if not self.source or not self.category:
            raise ValueError("Missing source or category")
        if not self.permission or not self.permission.strip():
            raise ValueError("Explicit permission required")
        if self.split not in SPLITS:
            raise ValueError("Invalid split")
        if self.review not in ("pending", "approved", "rejected"):
            raise ValueError("Invalid review state")
        if self.origin not in ("human", "local-teacher", "synthetic-test"):
            raise ValueError("Unknown record origin")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Empty text")


@dataclasses.dataclass(frozen=True)
class CurationDecision:
    identifier: str
    accepted: bool
    reasons: tuple[str, ...]
    content_sha256: str
    duplicate_of: str | None = None

    def metadata(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class CuratedCorpus:
    records: list[SourceRecord]
    decisions: list[CurationDecision]
    policy: CurationPolicy

    def report(self):
        accepted = [x for x in self.decisions if x.accepted]
        reasons = collections.Counter(
            reason for row in self.decisions for reason in row.reasons
        )
        return {
            "submitted": len(self.decisions),
            "accepted": len(accepted),
            "rejected": len(self.decisions) - len(accepted),
            "rejection_reasons": dict(reasons),
            "accepted_splits": dict(collections.Counter(x.split for x in self.records)),
            "accepted_categories": dict(collections.Counter(x.category for x in self.records)),
            "accepted_bytes": sum(len(x.text.encode("utf-8")) for x in self.records),
            "policy": dataclasses.asdict(self.policy),
            "note": "Permission metadata must be verified by a human",
        }


def canonical(text: str) -> str:
    """Normalize whitespace and case, not the meaning or quoted content."""
    return " ".join(WORD.findall(text.casefold()))


def digest(text: str) -> str:
    return hashlib.sha256(canonical(text).encode("utf-8")).hexdigest()


def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = WORD.findall(text.casefold())
    return {tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))}


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def near_duplicate(a: str, b: str, threshold: float, n: int) -> bool:
    left, right = word_ngrams(a, n), word_ngrams(b, n)
    # Small texts with no shingles must be checked by normalized equality.
    if not left or not right:
        return canonical(a) == canonical(b)
    return jaccard(left, right) >= threshold


def sensitive_indicators(text: str) -> dict:
    return {
        "possible_secrets": len(SECRET.findall(text)),
        "possible_emails": len(EMAIL.findall(text)),
    }


def reject_reasons(row: SourceRecord, policy: CurationPolicy) -> list[str]:
    row.validate()
    policy.validate()
    problems = []
    length = len(row.text.encode("utf-8"))
    if length < policy.minimum_bytes:
        problems.append("too_short")
    if length > policy.maximum_bytes:
        problems.append("too_large")
    indicators = sensitive_indicators(row.text)
    if policy.reject_possible_secrets and indicators["possible_secrets"]:
        problems.append("possible_secret")
    if policy.reject_possible_emails and indicators["possible_emails"]:
        problems.append("possible_email")
    if row.review != "approved":
        problems.append("not_approved")
    if policy.require_documented_permission and row.permission in (
        "unknown", "unspecified", "none", "not-checked"
    ):
        problems.append("permission_unverified")
    return problems


def curate(
    records: Iterable[SourceRecord],
    policy: CurationPolicy | None = None,
) -> CuratedCorpus:
    """Reject duplicates across splits instead of leaking holdout answers."""
    policy = policy or CurationPolicy()
    policy.validate()
    accepted = []
    decisions = []
    seen_ids = set()
    accepted_digests = {}
    shingles: list[tuple[SourceRecord, set]] = []
    for row in records:
        row.validate()
        reasons = reject_reasons(row, policy)
        duplicate_of = None
        hash_value = digest(row.text)
        if row.identifier in seen_ids:
            reasons.append("duplicate_identifier")
        seen_ids.add(row.identifier)
        previous = accepted_digests.get(hash_value)
        if previous is not None:
            duplicate_of = previous.identifier
            reasons.append(
                "cross_split_exact_duplicate"
                if previous.split != row.split else "exact_duplicate"
            )
        elif not reasons:
            current = word_ngrams(row.text, policy.ngram_size)
            for previous_row, previous_shingles in shingles:
                if not current or not previous_shingles:
                    similar = canonical(row.text) == canonical(previous_row.text)
                else:
                    similar = jaccard(current, previous_shingles) >= policy.near_duplicate_threshold
                if similar:
                    duplicate_of = previous_row.identifier
                    reasons.append(
                        "cross_split_near_duplicate"
                        if previous_row.split != row.split else "near_duplicate"
                    )
                    break
        accepted_flag = not reasons
        decisions.append(
            CurationDecision(row.identifier, accepted_flag, tuple(reasons), hash_value, duplicate_of)
        )
        if accepted_flag:
            accepted.append(row)
            accepted_digests[hash_value] = row
            shingles.append((row, word_ngrams(row.text, policy.ngram_size)))
    return CuratedCorpus(accepted, decisions, policy)


def group_safe_splits(records: Iterable[SourceRecord]) -> dict[str, list[SourceRecord]]:
    values = list(records)
    groups = {name: [] for name in SPLITS}
    for row in values:
        row.validate()
        groups[row.split].append(row)
    hashes = collections.defaultdict(set)
    for row in values:
        hashes[digest(row.text)].add(row.split)
    overlap = [key for key, splits in hashes.items() if len(splits) > 1]
    if overlap:
        raise ValueError(f"Exact cross-split leakage detected: {len(overlap)} fingerprints")
    return groups


def source_fingerprint(records: Iterable[SourceRecord]) -> str:
    h = hashlib.sha256()
    for row in sorted(records, key=lambda x: x.identifier):
        h.update(json.dumps(
            {
                "id": row.identifier, "sha": digest(row.text),
                "split": row.split, "license": row.permission,
                "review": row.review, "origin": row.origin,
            }, sort_keys=True
        ).encode("utf-8"))
    return h.hexdigest()


def read_source_jsonl(path: Path) -> list[SourceRecord]:
    result = []
    with path.open("r", encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("Expected JSON object")
                row = SourceRecord(**payload)
                row.validate()
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            result.append(row)
    return result


def write_source_jsonl(path: Path, records: Iterable[SourceRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as out:
        for row in records:
            row.validate()
            out.write(json.dumps(dataclasses.asdict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def write_audit(path: Path, corpus: CuratedCorpus) -> None:
    """Audit excludes raw text and any matched secret values."""
    report = corpus.report()
    report["source_fingerprint"] = source_fingerprint(corpus.records)
    report["decisions"] = [item.metadata() for item in corpus.decisions]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def categories(records: Iterable[SourceRecord]) -> dict[str, int]:
    return dict(collections.Counter(row.category for row in records))


def assess_holdout_independence(
    train: Iterable[SourceRecord], holdout: Iterable[SourceRecord],
    policy: CurationPolicy | None = None,
) -> dict:
    policy = policy or CurationPolicy()
    train_rows, holdout_rows = list(train), list(holdout)
    matches = []
    for candidate in holdout_rows:
        for reference in train_rows:
            if near_duplicate(
                reference.text, candidate.text,
                policy.near_duplicate_threshold, policy.ngram_size
            ):
                matches.append({
                    "train_id": reference.identifier,
                    "holdout_id": candidate.identifier,
                    "train_split": reference.split,
                    "holdout_split": candidate.split,
                })
    return {
        "independent_by_ngram_check": len(matches) == 0,
        "matches": matches,
        "train_count": len(train_rows), "holdout_count": len(holdout_rows),
        "limitations": "Lexical n-grams cannot detect all semantic paraphrases",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline, permission-aware corpus curation")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args(argv)
    corpus = curate(
        read_source_jsonl(args.input),
        CurationPolicy(near_duplicate_threshold=args.threshold),
    )
    write_source_jsonl(args.accepted, corpus.records)
    write_audit(args.audit, corpus)
    print(json.dumps(corpus.report(), indent=2))
    return 0 if corpus.records else 2


if __name__ == "__main__":
    raise SystemExit(main())
