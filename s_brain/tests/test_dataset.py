"""Dependency-light regression suite for the S-Brain data provenance module."""
import json
import tempfile
import unittest
from pathlib import Path

from s_brain.dataset import (
    CorpusRecord,
    assert_nonoverlap,
    audit_manifest,
    build_manifest,
    export_review_csv,
    merge_manifests,
    normalize,
    read_documents,
    read_jsonl,
    sample_records,
    scrub,
    stable_split,
    summarize,
    write_byte_streams,
    write_jsonl,
)


class RecordTests(unittest.TestCase):
    def make(self, identifier="record1", text="A meaningful training document."):
        return CorpusRecord.from_text(
            identifier, text, "local-approved", "user-owned", "train"
        )

    def test_record_has_digest(self):
        record = self.make()
        self.assertEqual(len(record.sha256), 64)
        self.assertEqual(record.bytes_utf8, len(record.text.encode("utf-8")))

    def test_record_does_not_accept_blank_text(self):
        with self.assertRaises(ValueError):
            self.make(text="   ")

    def test_record_requires_permission_metadata(self):
        with self.assertRaises(ValueError):
            CorpusRecord.from_text("x", "Text", "source", "", "train")

    def test_record_requires_valid_split(self):
        with self.assertRaises(ValueError):
            CorpusRecord.from_text("x", "Text", "source", "owned", "other")

    def test_record_detects_tampering(self):
        record = self.make()
        row = vars(record).copy()
        row["text"] = "Changed without updating digest"
        with self.assertRaises(ValueError):
            CorpusRecord.parse(row)

    def test_scrub_email(self):
        self.assertNotIn("example.org", scrub("Contact user@example.org now"))

    def test_scrub_secret(self):
        self.assertNotIn("abc123", scrub("api_key=abc123"))

    def test_control_chars(self):
        self.assertEqual(scrub("a\x00b\x01c"), "abc")

    def test_line_endings(self):
        self.assertEqual(normalize("a\r\n\r\n\r\n\r\n\r\nb"), "a\n\n\nb")

    def test_stable_split_reproducibility(self):
        one = [stable_split(f"id:{i}", seed=11) for i in range(100)]
        two = [stable_split(f"id:{i}", seed=11) for i in range(100)]
        self.assertEqual(one, two)

    def test_stable_split_ratios(self):
        with self.assertRaises(ValueError):
            stable_split("x", train_ratio=0.9, validation_ratio=0.2)

    def test_all_splits_are_reachable(self):
        seen = {stable_split(str(i)) for i in range(2000)}
        self.assertEqual(seen, {"train", "validation", "test"})

    def test_summary_counts(self):
        records = [self.make("a", "Alpha"), self.make("b", "Beta")]
        report = summarize(records)
        self.assertEqual(report["records"], 2)
        self.assertEqual(report["by_split"]["train"], 2)

    def test_hash_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("Same text for deduplication. " * 3)
            (root / "b.txt").write_text("Same text for deduplication. " * 3)
            records, report = build_manifest(root, "local", "owned")
            self.assertEqual(len(records), 1)
            self.assertEqual(report["rejected"]["duplicate"], 1)

    def test_short_documents_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "short.txt").write_text("too short")
            records, report = build_manifest(root, "local", "owned")
            self.assertEqual(records, [])
            self.assertEqual(report["rejected"]["short"], 1)

    def test_manifest_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            records = [self.make("a", "Text A"), self.make("b", "Text B")]
            self.assertEqual(write_jsonl(path, records), 2)
            self.assertEqual(list(read_jsonl(path)), records)
            self.assertEqual(audit_manifest(path)["records"], 2)

    def test_corrupt_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text(json.dumps({"identifier": "bad"}))
            with self.assertRaises(ValueError):
                list(read_jsonl(path))

    def test_explicit_split_overlap_error(self):
        left = self.make("a", "Repeated")
        right = CorpusRecord.from_text("b", "Repeated", "local", "owned", "test")
        with self.assertRaises(ValueError):
            assert_nonoverlap([left, right])

    def test_byte_streams_have_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [
                CorpusRecord.from_text("a", "Train", "local", "owned", "train"),
                CorpusRecord.from_text("b", "Holdout", "local", "owned", "validation"),
                CorpusRecord.from_text("c", "Test", "local", "owned", "test"),
            ]
            report = write_byte_streams(root, records)
            self.assertEqual(report["train"]["documents"], 1)
            self.assertEqual((root/"train.bin").read_bytes(), b"Train\n\n")
            self.assertEqual((root/"validation.bin").read_bytes(), b"Holdout\n\n")

    def test_review_export_does_not_include_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"review.csv"
            export_review_csv(path, [self.make(text="SECRET BODY")])
            self.assertNotIn("SECRET BODY", path.read_text())

    def test_sample_is_seeded(self):
        records = [self.make(str(i), "Text " + str(i)) for i in range(30)]
        self.assertEqual(
            sample_records(records, 5, seed=2),
            sample_records(records, 5, seed=2)
        )

    def test_sample_validation(self):
        with self.assertRaises(ValueError):
            sample_records([], -1)

    def test_merge_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"m.jsonl"
            write_jsonl(path, [self.make()])
            result = merge_manifests([path, path])
            self.assertEqual(len(result), 1)

    def test_iterating_source_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"example.txt").write_text("a\n\nb")
            (root/"ignore.exe").write_text("not corpus")
            self.assertEqual(list(read_documents(root)), [("example.txt", "a\n\nb")])


if __name__ == "__main__":
    unittest.main()
