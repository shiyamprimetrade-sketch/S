import json
import unittest
from pathlib import Path
from jsonschema import Draft202012Validator
from project_s.contracts import request_for, validate, receipt, CAPABILITIES, STATES

class SchemaConformance(unittest.TestCase):
    def validator(self, name):
        schema=json.loads((Path(__file__).resolve().parents[1]/"schemas"/name).read_text())
        Draft202012Validator.check_schema(schema)
        return Draft202012Validator(schema)

    def test_requests(self):
        v=self.validator("sbi-request-v1.json")
        for capability in sorted(CAPABILITIES):
            request=request_for("Create "+capability,capability)
            v.validate(request)
            for patch in ({"deadline_ms":0},{"deadline_ms":120001},{"operation_id":"../x"},{"capability":"shell"},{"payload":{"requirement":" "*10}},{"payload":{"requirement":"x"*513}},{"extra":True}):
                bad=dict(request,**patch)
                self.assertFalse(v.is_valid(bad))
                with self.assertRaises(ValueError):
                    validate(bad)

    def test_receipts(self):
        v=self.validator("scc-receipt-v1.json")
        for state in sorted(STATES):
            kwargs={"artifact_sha256":"b"*64,"evidence_ids":["verification"]} if state=="COMPLETED" else {"error_code":"REJECTED"} if state in ("FAILED","DENIED","BLOCKED") else {}
            v.validate(receipt("a"*32,state,**kwargs))
        good=receipt("a"*32,"COMPLETED",artifact_sha256="b"*64,evidence_ids=["verification"])
        for patch in ({"artifact_sha256":None},{"evidence_ids":[]},{"error_code":"ERROR"},{"operation_id":"bad"}):
            self.assertFalse(v.is_valid(dict(good,**patch)))
