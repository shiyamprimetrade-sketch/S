"""Deterministic, provenance-aware corpora and byte-level training windows.

The dataset format is deliberately independent of hosted model providers.
Every input record carries source, license/permission, split, and a SHA-256
fingerprint. No private conversations are published by this module.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
EMAIL = re.compile(r"(?<![\w.])(?:[A-Za-z0-9._%+-]+)@(?:[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)")
TOKEN = re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*[^\s,;]+")
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class CorpusRecord:
    identifier: str
    text: str
    source: str
    license: str
    split: str
    sha256: str
    bytes_utf8: int

    @staticmethod
    def from_text(identifier: str, text: str, source: str,
                  license: str, split: str) -> "CorpusRecord":
        if split not in SPLITS:
            raise ValueError("Invalid split: " + split)
        if not identifier.strip() or not source.strip() or not license.strip():
            raise ValueError("Record identifier, source and license required")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Corpus text cannot be blank")
        encoded = text.encode("utf-8")
        return CorpusRecord(identifier, text, source, license, split,
                            hashlib.sha256(encoded).hexdigest(), len(encoded))

    @staticmethod
    def parse(row: dict) -> "CorpusRecord":
        result = CorpusRecord.from_text(
            str(row["identifier"]), str(row["text"]),
            str(row["source"]), str(row["license"]), str(row["split"])
        )
        if result.sha256 != row.get("sha256"):
            raise ValueError("Content hash mismatch")
        return result


def scrub(text: str, redact_contacts: bool = True) -> str:
    """Remove NUL/control bytes and obvious contact/secret patterns.

    This is a safeguard rather than a guarantee of de-identification.
    """
    if not isinstance(text, str):
        raise TypeError("Expected text")
    result = CONTROL.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    result = TOKEN.sub("[REDACTED_SECRET]", result)
    if redact_contacts:
        result = EMAIL.sub("[REDACTED_EMAIL]", result)
        result = PHONE.sub("[REDACTED_PHONE]", result)
    return result


def normalize(text: str) -> str:
    """Normalize line breaks, trailing whitespace and empty-line floods."""
    lines = [line.rstrip() for line in scrub(text).splitlines()]
    cleaned = "\n".join(lines).strip()
    return re.sub(r"\n{4,}", "\n\n\n", cleaned)


def stable_split(identifier: str, seed: int = 42,
                 train_ratio: float = 0.9,
                 validation_ratio: float = 0.05) -> str:
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
        raise ValueError("Ratios must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Leave room for test split")
    raw = f"{seed}:{identifier}".encode("utf-8")
    fraction = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / (2**64)
    if fraction < train_ratio:
        return "train"
    if fraction < train_ratio + validation_ratio:
        return "validation"
    return "test"


def iter_text_files(root: Path, suffixes=(".txt", ".md")) -> Iterator[Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            if path.is_symlink():
                continue
            yield path


def read_documents(root: Path, max_bytes: int = 2_000_000,
                   suffixes=(".txt", ".md")) -> Iterator[tuple[str, str]]:
    """Read small explicit text sources, without following symlinked files."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    for path in iter_text_files(root, suffixes=suffixes):
        if path.stat().st_size > max_bytes:
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        text = normalize(raw)
        if text:
            yield path.relative_to(root).as_posix(), text


def build_manifest(root: Path, source: str, license: str,
                   seed: int = 42,
                   redact: bool = True,
                   min_bytes: int = 40,
                   max_bytes: int = 2_000_000) -> tuple[list[CorpusRecord], dict]:
    """Enumerate a corpus and reject exact duplicates across splits."""
    if not source or not license:
        raise ValueError("Explicit source and permission/license required")
    records = []
    seen = set()
    rejected = Counter()
    for relative, original in read_documents(root, max_bytes=max_bytes):
        text = normalize(original) if redact else original.strip()
        raw = text.encode("utf-8")
        if len(raw) < min_bytes:
            rejected["short"] += 1
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(digest)
        record = CorpusRecord.from_text(
            relative, text, source, license, stable_split(relative, seed)
        )
        records.append(record)
    stats = summarize(records)
    stats["rejected"] = dict(rejected)
    stats["seed"] = seed
    return records, stats


def summarize(records: Iterable[CorpusRecord]) -> dict:
    rows = list(records)
    return {
        "records": len(rows),
        "bytes": sum(row.bytes_utf8 for row in rows),
        "by_split": dict(Counter(row.split for row in rows)),
        "by_source": dict(Counter(row.source for row in rows)),
        "duplicate_hashes": len(rows) - len(set(row.sha256 for row in rows)),
    }


def write_jsonl(path: Path, records: Iterable[CorpusRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            total += 1
    return total


def read_jsonl(path: Path, strict: bool = True) -> Iterator[CorpusRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Not a JSON object")
                yield CorpusRecord.parse(value)
            except (ValueError, TypeError, KeyError) as exc:
                if strict:
                    raise ValueError(f"{path}:{lineno}: {exc}") from exc


def assert_nonoverlap(records: Iterable[CorpusRecord]) -> None:
    splits_by_hash = {}
    for record in records:
        previous = splits_by_hash.setdefault(record.sha256, record.split)
        if previous != record.split:
            raise ValueError(f"Split leakage for {record.identifier}")


def write_byte_streams(root: Path, records: Iterable[CorpusRecord],
                       separator: bytes = b"\n\n") -> dict:
    """Encode one file per split; reject duplicate content across splits."""
    rows = list(records)
    assert_nonoverlap(rows)
    root.mkdir(parents=True, exist_ok=True)
    output = {}
    for split in SPLITS:
        target = root / f"{split}.bin"
        count = 0
        with target.open("wb") as handle:
            for row in rows:
                if row.split != split:
                    continue
                handle.write(row.text.encode("utf-8"))
                handle.write(separator)
                count += 1
        output[split] = {"path": str(target), "documents": count, "bytes": target.stat().st_size}
    return output


def audit_manifest(path: Path) -> dict:
    rows = list(read_jsonl(path))
    assert_nonoverlap(rows)
    return summarize(rows)


def merge_manifests(paths: Iterable[Path], unique: bool = True) -> list[CorpusRecord]:
    rows = []
    fingerprints = set()
    for path in paths:
        for record in read_jsonl(path):
            if unique and record.sha256 in fingerprints:
                continue
            fingerprints.add(record.sha256)
            rows.append(record)
    assert_nonoverlap(rows)
    return rows


def sample_records(records: list[CorpusRecord], count: int,
                   seed: int = 42, split: str | None = None) -> list[CorpusRecord]:
    if count < 0:
        raise ValueError("count must be nonnegative")
    candidates = [record for record in records if split is None or record.split == split]
    generator = random.Random(seed)
    if count >= len(candidates):
        generator.shuffle(candidates)
        return candidates
    return generator.sample(candidates, count)


def export_review_csv(path: Path, records: Iterable[CorpusRecord]) -> int:
    """Export metadata only, not the underlying sensitive document text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["identifier", "source", "license", "split", "sha256", "bytes_utf8"]
    count = 0
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({name: getattr(record, name) for name in fields})
            count += 1
    return count


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Prepare an auditable private S-Brain corpus")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--source", required=True)
    build.add_argument("--license", required=True)
    build.add_argument("--seed", type=int, default=42)
    verify = sub.add_parser("audit")
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "audit":
        print(json.dumps(audit_manifest(args.manifest), indent=2))
        return 0
    records, stats = build_manifest(args.input, args.source, args.license, args.seed)
    write_jsonl(args.output / "manifest.jsonl", records)
    write_byte_streams(args.output, records)
    (args.output / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
