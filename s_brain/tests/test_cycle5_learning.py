"""Cycle 5 regression tests: real optimizer updates and held-out improvements.

Test fixtures are tiny self-authored synthetic examples. A passing test proves
the data path, gradient steps and independent model evaluation are executable,
NOT that the model can compete with a commercial frontier assistant.
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import torch

from s_brain.model import Config, SBrain
from s_brain.checkpoint import load_weights, load_manifest
from s_brain.supervised_data import (
    encode_record, dense_completion_windows, build_windows,
    fingerprint_dataset, masked_cross_entropy, WindowSampler,
)
from s_brain.train_supervised import (
    SupervisedTrainConfig, check_split_integrity, cosine_lr, eval_windows,
    optimizer_step, run_supervised_training, weight_update_l2, model_weights,
)
from s_brain.evaluate_learning import (
    EvalSettings, TokenError, check_holdout, score_record,
    aggregate_scores, bootstrap_metric, evaluate_checkpoint,
    paired_evaluate, pass_improvement_gate, compare_saved_checkpoints,
)


def record(number:int,split:str="train"):
    # Source text is authored for a tiny deterministic functional test.
    return encode_record(
        identifier=f"{split}-{number}",
        question=f"What result does the number {number} indicate?",
        answer=f"The example result is {number} units.",
        category="math",
        source_hash=f"{split}-content-{number}",
        permission="self-authored",
        split=split,
    )


def tiny_config(steps=2):
    return SupervisedTrainConfig(
        steps=steps,batch_size=2,context=16,
        dim=32,layers=1,heads=4,kv_heads=2,
        gradient_accumulation=1,
        validate_every=1,validation_batches=1,
        minimum_train_windows=2,minimum_holdout_windows=2,
        lr=1e-3,
    )


def tiny_model():
    torch.manual_seed(32)
    model=SBrain(Config(
        dim=32,layers=1,heads=4,kv_heads=2,context=16
    ))
    model.eval()
    return model


class TrainConfigTests(unittest.TestCase):
    def test_normal_config(self):
        tiny_config().validate()

    def test_invalid_head_width(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(tiny_config(),dim=30).validate()

    def test_invalid_learning_rate(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(tiny_config(),lr=-1).validate()

    def test_schedule_is_bounded(self):
        settings=tiny_config()
        values=[cosine_lr(step,10,settings.lr,settings.min_lr) for step in range(11)]
        self.assertLessEqual(max(values),settings.lr)
        self.assertGreaterEqual(min(values),settings.min_lr)

    def test_split_integrity(self):
        report=check_split_integrity([record(1)],[record(2,"validation")])
        self.assertEqual(report["question_overlap"],0)

    def test_training_requires_train_split(self):
        with self.assertRaises(ValueError):
            check_split_integrity(
                [record(1,"validation")],[record(2,"validation")]
            )

    def test_holdout_cannot_be_training(self):
        with self.assertRaises(ValueError):
            check_split_integrity([record(1)],[record(2)])

    def test_identical_question_cannot_leak(self):
        first=encode_record(
            "train","Same question?","The first answer.",
            category="test",source_hash="train",
            permission="owned",split="train"
        )
        second=encode_record(
            "val","Same question?","The second answer.",
            category="test",source_hash="validation",
            permission="owned",split="validation"
        )
        with self.assertRaises(ValueError):
            check_split_integrity([first],[second])

    def test_source_hash_overlap_rejected(self):
        before=record(1)
        after=dataclasses.replace(
            record(2,"validation"),source_hash=before.source_hash
        )
        with self.assertRaises(ValueError):
            check_split_integrity([before],[after])

    def test_weight_snapshot_and_update(self):
        net=tiny_model()
        optimizer=torch.optim.AdamW(net.parameters(),lr=0.001)
        before=model_weights(net)
        sample=torch.randint(0,256,(2,16))
        _,loss=net(sample,sample)
        loss.backward()
        optimizer.step()
        self.assertGreater(weight_update_l2(before,net),0)

    def test_optimizer_step_metrics(self):
        net=tiny_model()
        optimizer=torch.optim.AdamW(net.parameters(),lr=0.001)
        windows=build_windows([record(1)],16)
        sampler=WindowSampler(windows,16)
        result=optimizer_step(net,optimizer,sampler,tiny_config(),0.001)
        self.assertGreater(result["weight_change_l2"],0)
        self.assertGreater(result["supervised_tokens"],0)
        self.assertGreater(result["gradient_norm"],0)

    def test_validation_loss_finite(self):
        net=tiny_model()
        windows=build_windows([record(2,"validation")],16)
        report=eval_windows(net,windows,tiny_config())
        self.assertTrue(torch.isfinite(torch.tensor(report["cross_entropy"])))

    def test_real_training_creates_best_checkpoint_or_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"experiment"
            report=run_supervised_training(
                [record(1),record(2)],
                [record(10,"validation"),record(11,"validation")],
                output,config=tiny_config(steps=2),
            )
            self.assertTrue(report["weights_changed_each_step"])
            self.assertEqual(report["optimizer_steps"],2)
            self.assertTrue((output/"baseline"/"model.safetensors").exists())
            self.assertTrue((output/"final"/"model.safetensors").exists())
            self.assertTrue((output/"metrics.jsonl").exists())
            self.assertTrue((output/"report.json").exists())

    def test_baseline_and_final_weights_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"out"
            run_supervised_training(
                [record(1)], [record(10,"validation")], output,
                config=tiny_config(steps=1),
            )
            old=load_manifest(output/"baseline")
            new=load_manifest(output/"final")
            self.assertNotEqual(old.weights_sha256,new.weights_sha256)

    def test_saved_checkpoint_can_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"experiment"
            run_supervised_training(
                [record(1)], [record(9,"validation")],output,
                config=tiny_config(steps=1),
            )
            net,manifest=load_weights(output/"final")
            self.assertEqual(manifest.step,1)
            self.assertEqual(net.cfg.context,16)


class HeldOutEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.net=tiny_model()
        self.records=[record(10,"validation"),record(11,"validation")]

    def test_eval_settings_validation(self):
        EvalSettings(context=16).validate()
        with self.assertRaises(ValueError):
            EvalSettings(confidence=1.1).validate()

    def test_reject_training_records_as_holdout(self):
        with self.assertRaises(ValueError):
            check_holdout([record(1)])

    def test_holdout_fingerprint_stable(self):
        self.assertEqual(check_holdout(self.records),check_holdout(self.records))

    def test_scores_one_completion_byte_per_window(self):
        score=score_record(self.net,self.records[0],16,"cpu")
        count=len(dense_completion_windows(self.records[0],16))
        self.assertEqual(score.tokens,count)
        self.assertGreater(score.total_nll,0)

    def test_scores_are_finite(self):
        score=score_record(self.net,self.records[0],16,"cpu")
        self.assertTrue(torch.isfinite(torch.tensor(score.cross_entropy)))
        self.assertGreaterEqual(score.accuracy,0)
        self.assertLessEqual(score.accuracy,1)

    def test_aggregation(self):
        rows=[score_record(self.net,item,16,"cpu") for item in self.records]
        aggregated=aggregate_scores(rows)
        self.assertEqual(aggregated["records"],2)
        self.assertIn("math",aggregated["categories"])

    def test_bootstrap_interval(self):
        rows=[score_record(self.net,item,16,"cpu") for item in self.records]
        bounds=bootstrap_metric(rows,"accuracy",repetitions=50)
        self.assertLessEqual(bounds["lower"],bounds["upper"])

    def test_reject_bad_bootstrap_metric(self):
        with self.assertRaises(ValueError):
            bootstrap_metric([],"bad")

    def test_full_evaluation(self):
        report=evaluate_checkpoint(
            self.net,self.records,
            EvalSettings(context=16,repetitions=50),
        )
        self.assertEqual(report["records"],2)
        self.assertIn("loss_confidence_interval",report)

    def test_paired_same_model_has_zero_change(self):
        report=paired_evaluate(
            self.net,self.net,self.records,
            EvalSettings(context=16,repetitions=50),
        )
        self.assertAlmostEqual(report["loss_reduction"],0,places=5)
        self.assertFalse(report["proven_general_intelligence"])

    def test_gate_does_not_accept_fake_gain(self):
        report={
            "loss_reduction":0,
            "accuracy_gain":0.5,
            "weights_changed":True,
        }
        self.assertFalse(pass_improvement_gate(report)["passed"])

    def test_gate_rejects_unchanged_weights(self):
        report={
            "loss_reduction":0.5,
            "accuracy_gain":0,
            "weights_changed":False,
        }
        self.assertFalse(pass_improvement_gate(report)["passed"])

    def test_gate_accepts_valid_metric_improvement(self):
        report={
            "loss_reduction":0.4,
            "accuracy_gain":0.05,
            "weights_changed":True,
        }
        self.assertTrue(pass_improvement_gate(report)["passed"])

    def test_error_report_excludes_prompt_content(self):
        error=TokenError("private-record-identifier",1,100,101,0.1,2.3)
        scrubbed=error.redacted()
        self.assertNotIn("private-record-identifier",str(scrubbed))
        self.assertEqual(scrubbed["expected_byte"],100)


if __name__=="__main__":
    unittest.main()
