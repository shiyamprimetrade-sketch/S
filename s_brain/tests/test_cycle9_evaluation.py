"""Cycle 9: paired answer benchmark and anti-leakage regression tests.

This test suite does not invent real-model quality evidence: paired responses
come from an explicitly labeled fixture callback and are compared with gold
answers. Neural execution itself is covered by test_cycle9_pipeline.
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from s_brain.memory_store import MemoryStore
from s_brain.rag_evaluation import (
    AnswerCase,AnswerEvalPolicy,PairedAnswerScore,
    normalize_answer,exact_match,token_f1,
    citation_precision,paired_confidence_interval,score_ablation_case,
    aggregate_paired_scores,run_answer_evaluation,
    compare_evaluations,read_cases,write_cases,write_evaluation,
)
from s_brain.rag_inference import run_ablation_pair


class Model:
    class Cfg:
        context=2048
    cfg=Cfg()


def controlled_generator(model,prompt,*,max_new_tokens=32,seed=7):
    # Proves retrieval can make this exact fixture answer visible to a
    # generator. This is not an advanced S model or human language benchmark.
    if "Mercury has an orbital period of 88 days" in prompt:
        answer="Mercury has an orbital period of 88 days."
    else:
        answer="I do not know the requested orbital period."
    return {"response":answer,"model":"fixture","usage":{"generated_tokens":5}}


class AnswerEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.store=MemoryStore(Path(self.temp.name)/"memory.db")
        self.known=self.store.remember(
            scope="S",content="Mercury has an orbital period of 88 days.",
            source="self-authored astronomy note",consent="explicit",
        )
        self.model=Model()

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def case(self,identifier="q1"):
        return AnswerCase(
            identifier=identifier,
            question="What is the orbital period of Mercury?",
            expected_answer="Mercury has an orbital period of 88 days.",
            scope="S",category="astronomy",
            expected_source_ids=(self.known.id,),
        )

    def test_case_valid(self):
        self.case().validate()

    def test_case_training_split_rejected(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(self.case(),split="train").validate()

    def test_case_needs_permission(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(self.case(),permission="").validate()

    def test_duplicate_gold_id_rejected(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(
                self.case(),
                expected_source_ids=(self.known.id,self.known.id),
            ).validate()

    def test_eval_policy(self):
        AnswerEvalPolicy().validate()

    def test_invalid_policy_cases(self):
        with self.assertRaises(ValueError):
            AnswerEvalPolicy(max_cases=0).validate()

    def test_invalid_policy_tokens(self):
        with self.assertRaises(ValueError):
            AnswerEvalPolicy(max_new_tokens=0).validate()

    def test_exact_normalization(self):
        self.assertEqual(exact_match("Mercury, 88 days!","mercury 88 DAYS"),1)

    def test_different_answer_not_exact(self):
        self.assertEqual(exact_match("88 days","87 days"),0)

    def test_token_f1_perfect(self):
        self.assertEqual(token_f1("88 orbital days","88 orbital days"),1)

    def test_token_f1_partial(self):
        score=token_f1("Mercury orbits","Mercury orbits the sun")
        self.assertGreater(score,0)
        self.assertLess(score,1)

    def test_token_f1_zero(self):
        self.assertEqual(token_f1("apple","satellite"),0)

    def test_normalization_removes_citations(self):
        marker=f"[mem:{self.known.id}]"
        self.assertEqual(normalize_answer("Mercury "+marker),"mercury")

    def test_citation_precision_known(self):
        text=f"Mercury [mem:{self.known.id}]"
        self.assertEqual(citation_precision(text,(self.known.id,)),1)

    def test_citation_precision_missing(self):
        self.assertEqual(citation_precision("No citations",(self.known.id,)),0)

    def test_ablations_have_same_seed(self):
        result=run_ablation_pair(
            self.model,self.case().question,self.store,
            scope="S",seed=12,generator=controlled_generator,
        )
        self.assertEqual(result["seed"],12)
        self.assertTrue(result["same_generator"])

    def test_real_memory_fixture_improves_exact_score(self):
        result=run_ablation_pair(
            self.model,self.case().question,self.store,
            scope="S",generator=controlled_generator,
        )
        score=score_ablation_case(self.case(),result)
        self.assertEqual(score.no_memory_exact,0)
        self.assertEqual(score.memory_exact,1)
        self.assertEqual(score.exact_gain,1)

    def test_mismatched_question_hash_refused(self):
        result=run_ablation_pair(
            self.model,self.case().question,self.store,
            scope="S",generator=controlled_generator,
        )
        with self.assertRaises(ValueError):
            score_ablation_case(
                dataclasses.replace(self.case(),question="Different question"),
                result,
            )

    def test_paired_interval(self):
        result=paired_confidence_interval(
            [1,0,1,-1],seed=19,iterations=50
        )
        self.assertLessEqual(result["lower"],result["upper"])

    def test_paired_interval_requires_data(self):
        with self.assertRaises(ValueError):
            paired_confidence_interval([],iterations=50)

    def test_aggregate_paired(self):
        result=run_ablation_pair(
            self.model,self.case().question,self.store,
            scope="S",generator=controlled_generator,
        )
        scores=[score_ablation_case(self.case(),result)]
        report=aggregate_paired_scores(
            scores,AnswerEvalPolicy(bootstrap_iterations=50)
        )
        self.assertEqual(report["cases"],1)
        self.assertEqual(report["better_cases"],1)
        self.assertEqual(report["mean_exact_gain"],1)

    def test_scoring_does_not_claim_intelligence(self):
        report=run_answer_evaluation(
            self.model,self.store,[self.case()],
            generator=controlled_generator,
            policy=AnswerEvalPolicy(bootstrap_iterations=50),
        )
        self.assertFalse(report["summary"]["model_quality_proven"])
        self.assertTrue(report["gold_data_used_only_for_scoring"])

    def test_evaluation_with_no_relevant_memory(self):
        self.store.forget("S",self.known.id)
        report=run_answer_evaluation(
            self.model,self.store,[self.case()],
            generator=controlled_generator,
            policy=AnswerEvalPolicy(bootstrap_iterations=50),
        )
        self.assertEqual(report["summary"]["mean_exact_gain"],0)

    def test_duplicate_case_identifiers_rejected(self):
        with self.assertRaises(ValueError):
            run_answer_evaluation(
                self.model,self.store,[self.case(),self.case()],
                generator=controlled_generator,
                policy=AnswerEvalPolicy(bootstrap_iterations=50),
            )

    def test_too_many_cases_rejected(self):
        with self.assertRaises(ValueError):
            run_answer_evaluation(
                self.model,self.store,[self.case("1"),self.case("2")],
                generator=controlled_generator,
                policy=AnswerEvalPolicy(max_cases=1,bootstrap_iterations=50),
            )

    def test_evaluations_are_comparable(self):
        settings=AnswerEvalPolicy(bootstrap_iterations=50)
        first=run_answer_evaluation(
            self.model,self.store,[self.case()],
            generator=controlled_generator,policy=settings,
        )
        second=run_answer_evaluation(
            self.model,self.store,[self.case()],
            generator=controlled_generator,policy=settings,
        )
        delta=compare_evaluations(first,second)
        self.assertEqual(delta["memory_exact_change"],0)

    def test_different_gold_data_not_comparable(self):
        settings=AnswerEvalPolicy(bootstrap_iterations=50)
        first=run_answer_evaluation(
            self.model,self.store,[self.case()],
            generator=controlled_generator,policy=settings,
        )
        second=run_answer_evaluation(
            self.model,self.store,
            [dataclasses.replace(self.case(),expected_answer="Wrong answer")],
            generator=controlled_generator,policy=settings,
        )
        with self.assertRaises(ValueError):
            compare_evaluations(first,second)

    def test_score_output_hides_answer_text(self):
        report=run_answer_evaluation(
            self.model,self.store,[self.case()],
            generator=controlled_generator,
            policy=AnswerEvalPolicy(bootstrap_iterations=50),
        )
        text=json.dumps(report)
        self.assertNotIn("Mercury has an orbital period",text)
        self.assertIn("memory_response_hash",text)

    def test_gold_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"gold.jsonl"
            write_cases(path,[self.case()])
            loaded=read_cases(path)
            self.assertEqual(loaded,[self.case()])

    def test_gold_file_is_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"gold.jsonl"
            write_cases(path,[self.case()])
            with self.assertRaises(FileExistsError):
                write_cases(path,[self.case()])

    def test_evaluation_file_is_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"eval.json"
            write_evaluation(path,{"run":"first"})
            with self.assertRaises(FileExistsError):
                write_evaluation(path,{"run":"second"})

    def test_benchmark_output_can_be_json(self):
        report=run_answer_evaluation(
            self.model,self.store,[self.case()],
            generator=controlled_generator,
            policy=AnswerEvalPolicy(bootstrap_iterations=50),
        )
        json.dumps(report,allow_nan=False)
        self.assertEqual(len(report["question_set_sha256"]),64)


if __name__=="__main__":
    unittest.main()
