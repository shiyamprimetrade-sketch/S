"""Dataset quality, leakage and privacy indicator regression tests."""
import tempfile
import unittest
from pathlib import Path

from s_brain.data_quality import (
    QualityPolicy, compare_quality, detect_cross_split_duplicates,
    duplicate_fraction, evaluate_quality, find_sensitive_indicators,
    ngram_fraction, sentence_tokens, shannon_entropy,
    short_document_fraction, inspect_manifest,
)
from s_brain.dataset import CorpusRecord, write_jsonl


def row(number, text=None, split="train"):
    return CorpusRecord.from_text(
        str(number),
        text or f"Document {number} with varied English technical details and unique examples.",
        "owned-test-material", "user-owned", split,
    )


class DataQualityTests(unittest.TestCase):
    def test_tokenizing_words(self):
        self.assertEqual(sentence_tokens("Hello, World!"), ["hello", "world"])

    def test_ngram_repetition(self):
        self.assertGreater(ngram_fraction("a b c a b c a b c"), 0)

    def test_ngram_unique(self):
        self.assertEqual(ngram_fraction("a b c d e f"), 0)

    def test_ngram_bad_size(self):
        with self.assertRaises(ValueError):
            ngram_fraction("hello", 0)

    def test_entropy_all_equal(self):
        self.assertEqual(shannon_entropy(b"aaaaa"), 0)

    def test_entropy_mixed(self):
        self.assertGreater(shannon_entropy(b"abcde"), 0)

    def test_duplicate_fraction(self):
        items = [row(1, "repeated"), row(2, "repeated"), row(3, "unique")]
        self.assertAlmostEqual(duplicate_fraction(items), 1/3)

    def test_short_fraction(self):
        items = [row(1, "a"), row(2, "long enough")]
        self.assertEqual(short_document_fraction(items, 5), 0.5)

    def test_secret_indicator_never_returns_content(self):
        record = row(1, "api_key=TOPSECRET user@example.com")
        findings = find_sensitive_indicators([record])
        self.assertEqual(findings["possible_secret_assignments"], 1)
        self.assertEqual(findings["possible_emails"], 1)
        self.assertNotIn("TOPSECRET", str(findings))

    def test_cross_split_duplicate_detection(self):
        values = [row(1, "same", "train"), row(2, "same", "test")]
        self.assertEqual(detect_cross_split_duplicates(values), ["2"])

    def test_good_corpus_is_accepted(self):
        values = [
            row(
                i,
                f"Engineering document number {i} discusses separate "
                f"concepts and gives demonstration example {i*i} in context.",
                ["train", "validation", "test"][i % 3],
            )
            for i in range(15)
        ]
        report = evaluate_quality(values)
        self.assertTrue(report.accepted, str(report.as_dict()))

    def test_small_corpus_rejected(self):
        report = evaluate_quality([row(1)])
        self.assertFalse(report.accepted)
        self.assertIn(
            "document_count",
            [item.category for item in report.findings if not item.passed],
        )

    def test_synthetic_repetition_fails_gate(self):
        values = [
            row(i, f"same same same same same same {i}", "train")
            for i in range(20)
        ]
        report = evaluate_quality(values)
        self.assertFalse(report.accepted)

    def test_bad_threshold_rejected(self):
        with self.assertRaises(ValueError):
            QualityPolicy(max_short_fraction=2).validate()

    def test_split_coverage_optional(self):
        values = [row(i) for i in range(5)]
        report = evaluate_quality(
            values,
            QualityPolicy(
                minimum_documents=5,
                minimum_unique_documents=5,
                minimum_total_bytes=20,
                require_train_validation_test=False,
            ),
        )
        self.assertTrue(
            next(x for x in report.findings if x.category == "split_coverage").passed
        )

    def test_manifest_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "records.jsonl"
            write_jsonl(manifest, [row(1)])
            result = inspect_manifest(manifest)
            self.assertEqual(result["records"], 1)
            self.assertEqual(len(result["manifest_sha256"]), 64)

    def test_compare_gates(self):
        first = evaluate_quality([row(1)])
        rows = [
            row(
                i, f"Distinct technical material {i} in independent part {i*i}",
                ["train", "validation", "test"][i % 3],
            )
            for i in range(15)
        ]
        second = evaluate_quality(rows)
        delta = compare_quality(first, second)
        self.assertGreater(delta["documents_delta"], 0)


if __name__ == "__main__":
    unittest.main()
