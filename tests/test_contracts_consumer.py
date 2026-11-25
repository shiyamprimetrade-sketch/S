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

    def test_trusted_context_execution(self):
        import tempfile
        from project_s.secure import SecureEngine
        from project_s.contracts import trusted_context
        with tempfile.TemporaryDirectory() as root:
            engine=SecureEngine(root)
            task,plan=engine.create("Create a calculator")
            token=engine.issue(task)
            scope=token["scope"]
            self.assertEqual(scope["actor"],"local-owner")
            self.assertEqual(scope["capability"],"calculator")
            for actor,workspace,capability in (("other",scope["workspace"],"calculator"),("local-owner","wrong","calculator"),("local-owner",scope["workspace"],"website")):
                with self.assertRaises(ValueError):
                    trusted_context(actor,token,workspace,capability)
            self.assertFalse((engine.root/task).exists())
            self.assertTrue(engine.execute_token(task,token)["passed"])

    def test_raw_length_limit(self):
        with self.assertRaises(ValueError):
            request_for(" "*512+"calculator","calculator")
        self.assertEqual(len(request_for("x"*512,"calculator")["payload"]["requirement"]),512)
