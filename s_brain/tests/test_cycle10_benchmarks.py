"""Cycle 10: checkpoint-backed RAG benchmark and tool answer quality tests.

Checkpoints used in tests are randomly initialized toy S models with
verified safetensors manifests; execution is genuine, but the fixture does
not claim training quality, memory benefit, or frontier-level capability.
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import torch

from s_brain.model import Config,SBrain
from s_brain.checkpoint import save_checkpoint
from s_brain.memory_store import MemoryStore
from s_brain.rag_evaluation import AnswerCase
from s_brain.real_rag_benchmark import (
    RealRagPlan,VerifiedCheckpoint,case_fingerprint,inspect_cases,
    prepare_benchmark,open_checkpoint,real_model_ablation,
    compare_real_benchmarks,verify_benchmark_provenance,write_benchmark,
)
from s_brain.tool_answer_benchmark import (
    ToolAnswerCase,ToolQuestion,ToolAnswerPolicy,ToolAnswerResult,
    canonical_json,value_matches,case_digest,fixture_controller,
    execute_case,score_tool_answer,evaluate_tool_answers,
    aggregate_tool_answers,compare_tool_reports,read_tool_cases,
    write_tool_cases,write_tool_report,public_tool_question,
)


class RealModelRagTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        torch.manual_seed(220)
        self.model=SBrain(Config(
            dim=32,layers=1,heads=4,kv_heads=2,context=1024,
        ))
        self.checkpoint=self.root/"checkpoint"
        save_checkpoint(
            self.checkpoint,self.model,self.model.cfg,
            step=0,train_tokens=0,train_loss=5.5,
            validation_loss=5.5,seed=220,
            dataset_fingerprint="controlled-fixture-not-trained",
        )
        self.database=self.root/"memory.sqlite3"
        with MemoryStore(self.database) as memory:
            memory.remember(
                scope="S",content="Newton described classical gravity.",
                source="self-authored-test",consent="explicit",
            )
            memory.remember(
                scope="S",content="Mercury is a planet near the sun.",
                source="self-authored-test",consent="explicit",
            )
        self.cases=[
            AnswerCase(
                identifier="q1",question="Who described classical gravity?",
                expected_answer="Newton",scope="S",category="science",
                permission="self-authored",split="test",
            ),
            AnswerCase(
                identifier="q2",question="Which planet is near the sun?",
                expected_answer="Mercury",scope="S",category="science",
                permission="self-authored",split="test",
            ),
        ]

    def tearDown(self):
        self.temp.cleanup()

    def plan(self,**kwargs):
        return dataclasses.replace(
            RealRagPlan(minimum_cases=1,max_new_tokens=2,
                        bootstrap_iterations=50,minimum_context=512),
            **kwargs,
        )

    def test_plan_valid(self):
        self.plan().validate()

    def test_plan_rejects_no_generation(self):
        with self.assertRaises(ValueError):
            self.plan(max_new_tokens=0).validate()

    def test_plan_rejects_empty_case_cap(self):
        with self.assertRaises(ValueError):
            self.plan(minimum_cases=5,maximum_cases=1).validate()

    def test_case_fingerprint_stable(self):
        self.assertEqual(
            case_fingerprint(self.cases),
            case_fingerprint(list(reversed(self.cases))),
        )

    def test_gold_case_audit(self):
        report=inspect_cases(self.cases,self.plan())
        self.assertEqual(report["cases"],2)
        self.assertTrue(report["all_test"])

    def test_reject_training_split(self):
        wrong=dataclasses.replace(self.cases[0],split="validation")
        with self.assertRaises(ValueError):
            inspect_cases([wrong],self.plan())

    def test_duplicate_question_rejected(self):
        duplicate=dataclasses.replace(
            self.cases[1],question=self.cases[0].question
        )
        with self.assertRaises(ValueError):
            inspect_cases([self.cases[0],duplicate],self.plan())

    def test_missing_permission_rejected(self):
        bad=dataclasses.replace(self.cases[0],permission="unknown")
        with self.assertRaises(ValueError):
            inspect_cases([bad],self.plan())

    def test_checkpoint_loaded_and_verified(self):
        selected=open_checkpoint(self.checkpoint)
        self.assertEqual(selected.context,1024)
        self.assertEqual(selected.training_step,0)
        self.assertGreater(selected.parameter_count,0)
        self.assertEqual(len(selected.weights_sha256),64)

    def test_unknown_checkpoint_rejected(self):
        with self.assertRaises(ValueError):
            open_checkpoint(self.root/"missing")

    def test_corrupted_weights_rejected(self):
        # Clone raw files into separate scratch folder before corruption.
        target=self.root/"tampered"
        target.mkdir()
        for name in ("manifest.json","model.safetensors"):
            (target/name).write_bytes((self.checkpoint/name).read_bytes())
        with (target/"model.safetensors").open("ab") as handle:
            handle.write(b"unauthorized modification")
        with self.assertRaises(ValueError):
            open_checkpoint(target)

    def test_short_context_refused(self):
        with self.assertRaises(ValueError):
            prepare_benchmark(128,self.cases,self.plan())

    def test_too_long_gold_question_refused(self):
        huge=dataclasses.replace(self.cases[0],question="q"*600)
        with self.assertRaises(ValueError):
            prepare_benchmark(1024,[huge],self.plan())

    def test_real_model_forward_executed_in_both_arms(self):
        result=real_model_ablation(
            self.checkpoint,self.database,self.cases[:1],
            plan=self.plan(),
        )
        self.assertTrue(result["same_real_model_in_both_arms"])
        self.assertTrue(result["same_seed_in_both_arms"])
        self.assertFalse(result["teacher_called"])
        self.assertFalse(result["tool_called"])
        self.assertEqual(result["summary"]["cases"],1)
        self.assertFalse(result["independent_general_intelligence_established"])
        self.assertTrue(verify_benchmark_provenance(result)["valid"])

    def test_never_claim_benefit_without_positive_lower_bound(self):
        fake={
            "summary":{
                "mean_exact_gain":0,
                "exact_confidence_interval":{"lower":0},
                "better_cases":0,"worse_cases":0,
            }
        }
        self.assertFalse(
            fake["summary"]["exact_confidence_interval"]["lower"]>0
        )

    def test_benchmark_results_are_write_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"measurement.json"
            write_benchmark(path,{"synthetic":True})
            with self.assertRaises(FileExistsError):
                write_benchmark(path,{"synthetic":False})

    def test_benchmark_compare_requires_same_cases(self):
        before={
            "cases":{"fingerprint":"a"},"plan":{"seed":2},
            "checkpoint":{"weights_sha256":"1"},
            "summary":{},
        }
        after={
            "cases":{"fingerprint":"b"},"plan":{"seed":2},
            "checkpoint":{"weights_sha256":"2"},
            "summary":{},
        }
        with self.assertRaises(ValueError):
            compare_real_benchmarks(before,after)

    def test_checkpoint_provenance_audit_flags_teacher(self):
        report={
            "checkpoint":{
                "weights_sha256":"abc","training_step":0,
            },
            "cases":{"fingerprint":"x"*64,"all_test":True},
            "same_real_model_in_both_arms":True,
            "same_seed_in_both_arms":True,
            "teacher_called":True,"tool_called":False,
            "gold_answer_used_for_generation":False,
        }
        self.assertFalse(verify_benchmark_provenance(report)["valid"])


class ToolAnswerBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.case=ToolAnswerCase(
            identifier="arith-1",
            question="What is 7 multiplied by 8?",
            scope="S",tool_name="calculator",
            arguments={"operation":"multiply","left":7,"right":8},
            expected_output={
                "value":"56","operation":"multiply","exact_decimal":True,
            },
            expected_answer="56",
            category="arithmetic",
        )

    def answerer(self,question:ToolQuestion,execution):
        self.assertIsInstance(question,ToolQuestion)
        self.assertFalse(hasattr(question,"expected_answer"))
        self.assertFalse(hasattr(question,"expected_output"))
        if execution is None or execution.result is None:
            return "I cannot calculate that."
        return execution.result["value"]

    def test_case_valid(self):
        self.case.validate()

    def test_wrong_tool_refused(self):
        bad=dataclasses.replace(self.case,tool_name="run_shell")
        with self.assertRaises(ValueError):
            bad.validate()

    def test_training_split_refused(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(self.case,split="train").validate()

    def test_canonical_json_ignores_order(self):
        self.assertEqual(canonical_json({"a":1,"b":2}),
                         canonical_json({"b":2,"a":1}))

    def test_value_matches(self):
        self.assertTrue(value_matches({"a":1},{"a":1}))
        self.assertFalse(value_matches({"a":1},{"a":2}))

    def test_case_digest_stable(self):
        self.assertEqual(case_digest(self.case),case_digest(self.case))

    def test_public_tool_question_excludes_gold(self):
        question=public_tool_question(self.case)
        self.assertEqual(question.question,self.case.question)
        self.assertFalse(hasattr(question,"expected_answer"))
        self.assertFalse(hasattr(question,"expected_output"))

    def test_execute_readonly_calculator(self):
        trace,execution=execute_case(self.case)
        self.assertIsNotNone(execution)
        self.assertTrue(execution.success)
        self.assertEqual(execution.result["value"],"56")

    def test_denied_without_host_authorization(self):
        trace,execution=execute_case(self.case,authorize=False)
        self.assertIsNone(execution)
        self.assertEqual(len(trace.denied),1)

    def test_wrong_controller_scope_denied(self):
        controller=fixture_controller("Prime")
        trace,execution=execute_case(
            self.case,controller=controller,
        )
        self.assertIsNone(execution)
        self.assertEqual(len(trace.denied),1)

    def test_correct_output_scoring(self):
        _,execution=execute_case(self.case)
        result=score_tool_answer(self.case,execution,"56")
        self.assertTrue(result.output_matches)
        self.assertEqual(result.answer_exact,1)
        self.assertEqual(result.answer_f1,1)

    def test_incorrect_answer_does_not_hide_correct_tool(self):
        _,execution=execute_case(self.case)
        result=score_tool_answer(self.case,execution,"57")
        self.assertTrue(result.output_matches)
        self.assertEqual(result.answer_exact,0)

    def test_no_tool_result_scores_zero_fidelity(self):
        result=score_tool_answer(self.case,None,"No data")
        self.assertFalse(result.executed)
        self.assertFalse(result.output_matches)

    def test_tool_evaluation_fixture(self):
        report=evaluate_tool_answers(
            [self.case],answerer=self.answerer,
            policy=ToolAnswerPolicy(
                minimum_output_fidelity=1,
                minimum_answer_accuracy=1,
            ),
        )
        self.assertTrue(report["summary"]["accepted"])
        self.assertEqual(report["summary"]["answer_exact"],1)
        self.assertFalse(report["model_answer_ability_proven"])

    def test_host_denial_reduces_execution(self):
        report=evaluate_tool_answers(
            [self.case],answerer=self.answerer,authorize=False,
        )
        self.assertEqual(report["summary"]["execution_rate"],0)

    def test_quality_gate_fails_bad_answer(self):
        report=evaluate_tool_answers(
            [self.case],answerer=lambda q,e:"Incorrect",
            policy=ToolAnswerPolicy(minimum_answer_accuracy=1),
        )
        self.assertFalse(report["summary"]["accepted"])

    def test_duplicate_case_ids_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_tool_answers(
                [self.case,self.case],answerer=self.answerer
            )

    def test_too_many_cases_rejected(self):
        other=dataclasses.replace(self.case,identifier="second")
        with self.assertRaises(ValueError):
            evaluate_tool_answers(
                [self.case,other],answerer=self.answerer,
                policy=ToolAnswerPolicy(max_cases=1),
            )

    def test_missing_results_handled(self):
        result=score_tool_answer(self.case,None,"")
        self.assertEqual(result.answer_exact,0)

    def test_tool_report_has_no_raw_gold_answer(self):
        report=evaluate_tool_answers(
            [self.case],answerer=self.answerer,
        )
        self.assertNotIn('"expected_answer"',json.dumps(report))
        self.assertNotIn('"expected_output"',json.dumps(report))

    def test_benchmark_comparison_same_cases(self):
        old=evaluate_tool_answers(
            [self.case],answerer=lambda q,e:"wrong",
        )
        new=evaluate_tool_answers(
            [self.case],answerer=self.answerer,
        )
        delta=compare_tool_reports(old,new)
        self.assertEqual(delta["answer_accuracy_gain"],1)

    def test_benchmark_comparison_different_cases_denied(self):
        old=evaluate_tool_answers(
            [self.case],answerer=self.answerer,
        )
        new=evaluate_tool_answers(
            [dataclasses.replace(self.case,identifier="changed")],
            answerer=self.answerer,
        )
        with self.assertRaises(ValueError):
            compare_tool_reports(old,new)

    def test_tool_case_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"cases.jsonl"
            write_tool_cases(path,[self.case])
            loaded=read_tool_cases(path)
            self.assertEqual(loaded,[self.case])

    def test_case_file_write_once(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"cases.jsonl"
            write_tool_cases(path,[self.case])
            with self.assertRaises(FileExistsError):
                write_tool_cases(path,[self.case])

    def test_report_write_once(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"scores.json"
            write_tool_report(path,{"test":True})
            with self.assertRaises(FileExistsError):
                write_tool_report(path,{"test":False})


if __name__=="__main__":
    unittest.main()
