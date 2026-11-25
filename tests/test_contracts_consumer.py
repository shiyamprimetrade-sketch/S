import unittest
from project_s.contracts import request_for, validate, receipt
class ContractTests(unittest.TestCase):
    def test_supported_requests(self):
        for capability in ("calculator","website","inventory"):
            r=request_for("Create "+capability,capability)
            self.assertEqual(validate(r),r)
    def test_invalid_requests(self):
        r=request_for("Create calculator","calculator")
        for patch in ({"schema_version":True},{"deadline_ms":0},{"deadline_ms":True},{"operation_id":"../x"},{"capability":"shell"},{"actor":"admin"},{"payload":{"requirement":"x","authorization":True}},{"payload":{"requirement":" "*10}}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate(dict(r,**patch))
    def test_receipt_requires_evidence(self):
        op=request_for("Create calculator","calculator")["operation_id"]
        with self.assertRaises(ValueError): receipt(op,"COMPLETED")
        with self.assertRaises(ValueError): receipt(op,"FAILED")
        with self.assertRaises(ValueError): receipt(op,"COMPLETED",artifact_sha256="bad",evidence_ids=["test"])
        self.assertEqual(receipt(op,"COMPLETED",artifact_sha256="a"*64,evidence_ids=["test"])["status"],"COMPLETED")
