"""Cycle 8 stronger-learning and retrieval-quality regression tests.

Distribution-shift fixtures are intentionally authored in the test itself.
The model is randomly initialized; there is no claim of gained reasoning
from the presence of an evaluation framework.
"""
from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

import torch

from s_brain.model import Config,SBrain
from s_brain.supervised_data import encode_record
from s_brain.shift_evaluation import (
    ShiftProtocol,ShiftGroup,validate_group_isolation,
    weighted_record_interval,score_group,evaluate_shift,
    compare_model_shift_reports,category_regressions,write_report,
)
from s_brain.evaluate_learning import RecordScore
from s_brain.memory_store import MemoryStore
from s_brain.memory_evaluation import (
    RetrievalCase,RetrievalPolicy,evaluate_case,evaluate_retrieval,
    cross_scope_probe,compare_retrieval,labeled_query_digest,
    write_report as write_memory_report,
)


def row(number,split="validation",category="math"):
    return encode_record(
        identifier=f"{split}-{number}",
        question=f"Explain controlled test topic {number} with evidence.",
        answer=f"Answer number {number} is a self-authored text example.",
        category=category,
        source_hash=f"source-{split}-{number}",
        permission="owned",split=split,
    )


class ShiftEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(801)
        cls.model=SBrain(Config(
            dim=32,layers=1,heads=4,kv_heads=2,context=16
        ))
        cls.model.eval()

    def groups(self):
        base=ShiftGroup(
            "baseline","Equivalent task distribution",
            (row(1),row(2)),
        )
        shift=ShiftGroup(
            "novel","New question domain",
            (row(101,category="coding"),row(102,category="coding")),
        )
        return base,shift

    def test_protocol_valid(self):
        ShiftProtocol().validate()

    def test_protocol_rejects_invalid_confidence(self):
        with self.assertRaises(ValueError):
            ShiftProtocol(confidence=1).validate()

    def test_protocol_requires_minimum_examples(self):
        with self.assertRaises(ValueError):
            ShiftProtocol(minimum_records_per_group=0).validate()

    def test_empty_shift_group_rejected(self):
        with self.assertRaises(ValueError):
            ShiftGroup("empty","none",()).validate()

    def test_training_records_not_evaluable(self):
        with self.assertRaises(ValueError):
            ShiftGroup("bad","train",(row(1,"train"),)).validate()

    def test_isolation_passes_distinct_records(self):
        base,shift=self.groups()
        report=validate_group_isolation(base,[shift],ShiftProtocol())
        self.assertTrue(report["isolated"])
        self.assertEqual(report["groups"],2)

    def test_duplicate_prompt_rejected(self):
        base,shift=self.groups()
        collision=dataclasses.replace(
            shift.records[0],prompt_bytes=base.records[0].prompt_bytes
        )
        changed=dataclasses.replace(shift,records=(collision,shift.records[1]))
        report=validate_group_isolation(base,[changed],ShiftProtocol())
        self.assertFalse(report["isolated"])
        self.assertTrue(any("shared_prompt" in x for x in report["issues"]))

    def test_duplicate_source_hash_rejected(self):
        base,shift=self.groups()
        collision=dataclasses.replace(
            shift.records[0],source_hash=base.records[0].source_hash
        )
        changed=dataclasses.replace(shift,records=(collision,shift.records[1]))
        report=validate_group_isolation(base,[changed],ShiftProtocol())
        self.assertFalse(report["isolated"])

    def test_duplicate_group_name_rejected(self):
        base,shift=self.groups()
        changed=dataclasses.replace(shift,name="baseline")
        with self.assertRaises(ValueError):
            validate_group_isolation(base,[changed],ShiftProtocol())

    def test_score_real_model(self):
        base,_=self.groups()
        score=score_group(self.model,base,ShiftProtocol(bootstrap_repeats=50))
        self.assertEqual(score.records,2)
        self.assertGreater(score.token_count,0)
        self.assertGreater(score.cross_entropy,0)
        self.assertTrue(0<=score.accuracy<=1)

    def test_shift_report_has_fingerprints(self):
        base,shift=self.groups()
        report=evaluate_shift(
            self.model,base,[shift],
            policy=ShiftProtocol(bootstrap_repeats=50)
        )
        self.assertIn("baseline",report["isolation"]["fingerprints"])
        self.assertIn("novel",report["comparisons"])

    def test_identical_model_shift_report_stable(self):
        base,shift=self.groups()
        settings=ShiftProtocol(bootstrap_repeats=50)
        first=evaluate_shift(self.model,base,[shift],policy=settings)
        second=evaluate_shift(self.model,base,[shift],policy=settings)
        result=compare_model_shift_reports(first,second)
        self.assertTrue(result["same_protocol"])
        self.assertAlmostEqual(
            result["comparisons"]["novel"]["accuracy_drop_reduction"],0
        )

    def test_compare_different_dataset_rejected(self):
        base,shift=self.groups()
        report=evaluate_shift(
            self.model,base,[shift],policy=ShiftProtocol(bootstrap_repeats=50)
        )
        altered={**report,"isolation":{
            **report["isolation"],"fingerprints":{"other":"different"}
        }}
        with self.assertRaises(ValueError):
            compare_model_shift_reports(report,altered)

    def test_shift_missing_data_rejected(self):
        base,_=self.groups()
        with self.assertRaises(ValueError):
            score_group(self.model,base,ShiftProtocol(minimum_records_per_group=3))

    def test_bootstrap_interval_within_range(self):
        scores=[
            RecordScore("a","math","test",10,7,12.0,[]),
            RecordScore("b","math","test",10,3,18.0,[]),
        ]
        result=weighted_record_interval(
            scores,"accuracy",ShiftProtocol(bootstrap_repeats=50),
        )
        self.assertLessEqual(result["lower"],result["upper"])
        self.assertGreaterEqual(result["lower"],0)
        self.assertLessEqual(result["upper"],1)

    def test_write_report_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            destination=Path(folder)/"report.json"
            write_report(destination,{"test":"authored"})
            with self.assertRaises(FileExistsError):
                write_report(destination,{"test":"changed"})

    def test_category_regressions(self):
        base,_=self.groups()
        policy=ShiftProtocol(bootstrap_repeats=50)
        result=score_group(self.model,base,policy)
        groups=category_regressions(result,result)
        self.assertIn("math",groups)
        self.assertTrue(groups["math"]["passes"])


class RetrievalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=MemoryStore(Path(self.tmp.name)/"memory.sqlite3")
        self.good=self.store.remember(
            scope="S",content="Neural weights update using gradient descent",
            source="user-owned note",consent="explicit"
        )
        self.unrelated=self.store.remember(
            scope="S",content="Apples oranges grapes arrive in cargo",
            source="test material",consent="explicit"
        )
        self.private=self.store.remember(
            scope="Prime",content="Neural private gradient in another project",
            source="other",consent="explicit"
        )

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def case(self):
        return RetrievalCase(
            "c1","S","neural gradient",(self.good.id,),"learning"
        )

    def test_retrieval_case_valid(self):
        self.case().validate()

    def test_duplicate_gold_ids_rejected(self):
        case=dataclasses.replace(
            self.case(),expected_ids=(self.good.id,self.good.id)
        )
        with self.assertRaises(ValueError):
            case.validate()

    def test_policy_accepts_reasonable_minimums(self):
        RetrievalPolicy(k=3,minimum_precision=0.1).validate()

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError):
            RetrievalPolicy(k=0).validate()

    def test_case_relevance(self):
        score=evaluate_case(self.store,self.case(),RetrievalPolicy(k=3))
        self.assertGreater(score.precision_at_k,0)
        self.assertEqual(score.recall_at_k,1)
        self.assertTrue(score.citation_complete)

    def test_search_no_cross_scope_leakage(self):
        result=cross_scope_probe(self.store,"S","Prime","neural gradient")
        self.assertTrue(result["isolated"])
        self.assertEqual(result["private_records_exposed"],0)

    def test_scope_probe_must_be_distinct(self):
        with self.assertRaises(ValueError):
            cross_scope_probe(self.store,"S","S","neural")

    def test_retrieval_report_passes(self):
        policy=RetrievalPolicy(k=3,minimum_precision=0.5,minimum_recall=1)
        result=evaluate_retrieval(self.store,[self.case()],policy)
        self.assertTrue(result["accepted"],str(result))
        self.assertEqual(result["cases"],1)

    def test_unattainable_recall_fails(self):
        case=dataclasses.replace(self.case(),expected_ids=("unknown",))
        result=evaluate_retrieval(self.store,[case],
            RetrievalPolicy(minimum_recall=1.0))
        self.assertFalse(result["accepted"])

    def test_empty_retrieval_cases_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_retrieval(self.store,[],RetrievalPolicy())

    def test_duplicate_case_identifiers_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_retrieval(self.store,[self.case(),self.case()])

    def test_query_digest_stable(self):
        self.assertEqual(
            labeled_query_digest([self.case()]),
            labeled_query_digest([self.case()])
        )

    def test_retrieval_comparison_no_change(self):
        report=evaluate_retrieval(self.store,[self.case()])
        compared=compare_retrieval(report,report)
        self.assertEqual(compared["recall_gain"],0)

    def test_policy_mismatch_refused(self):
        before=evaluate_retrieval(self.store,[self.case()])
        after=evaluate_retrieval(
            self.store,[self.case()],RetrievalPolicy(k=2)
        )
        with self.assertRaises(ValueError):
            compare_retrieval(before,after)

    def test_report_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"report.json"
            write_memory_report(path,{"run":"first"})
            with self.assertRaises(FileExistsError):
                write_memory_report(path,{"run":"second"})


if __name__=="__main__":
    unittest.main()
