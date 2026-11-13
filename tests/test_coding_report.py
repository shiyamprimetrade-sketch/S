import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from s_brain.coding_report import (
    encode_report, report_fingerprint, save_report, summarize_cycle,
)


def outcome(initial=False, final=True):
    a = hashlib.sha256(b"initial").hexdigest()
    b = hashlib.sha256(b"final").hexdigest()
    return {
        "attempts": [
            {"number": 1, "source_sha256": a, "passed": initial, "status": "failed"},
            {"number": 2, "source_sha256": b, "passed": final, "status": "toolchain_passed"},
        ],
        "final_source_sha256": b,
        "passed": final,
        "benchmark": "nine-independent-python-cases-v1",
        "execution": "pinned-docker",
    }


class CodingReportTests(unittest.TestCase):
    def test_repair_metric(self):
        result = summarize_cycle(outcome())
        self.assertTrue(result["repaired_to_pass"])
        self.assertFalse(result["initial_passed"])
        self.assertTrue(result["final_passed"])

    def test_no_improvement_claim(self):
        self.assertFalse(summarize_cycle(outcome(False, False))["repaired_to_pass"])

    def test_reject_untested_final_source(self):
        data = outcome()
        data["final_source_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            summarize_cycle(data)

    def test_reject_mismatched_final_result(self):
        data = outcome()
        data["passed"] = False
        with self.assertRaises(ValueError):
            summarize_cycle(data)

    def test_report_encoding_is_deterministic(self):
        report = summarize_cycle(outcome())
        self.assertEqual(report_fingerprint(report), report_fingerprint(report))
        self.assertEqual(json.loads(encode_report(report))["attempt_count"], 2)

    def test_checkpoint_is_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.json"
            report = summarize_cycle(outcome())
            saved = save_report(report, path)
            self.assertEqual(saved["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                save_report(report, path)


if __name__ == "__main__":
    unittest.main()
