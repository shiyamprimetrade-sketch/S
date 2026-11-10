"""Cycle 7 acceptance: real curriculum and uniform weight updates with
measured validation improvements on intentionally simple held-out examples.

The controlled task repeats a short response across different question IDs.
It is a pipeline acceptance test, not a demonstration of reasoning, memorized
facts or generalization across domains.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from s_brain.curriculum_training import (
    CurriculumTrainingConfig,run_curriculum_training,
)
from s_brain.train_supervised import SupervisedTrainConfig
from s_brain.supervised_data import encode_record
from s_brain.checkpoint import load_weights,load_manifest
from s_brain.evaluate_learning import (
    EvalSettings,paired_evaluate,pass_improvement_gate,
)


def fixture(number,split):
    return encode_record(
        identifier=f"{split}-{number}",
        question=f"Explain a controlled experimental marker {number}.",
        answer="The answer is an independently learned repeated sequence.",
        category="controlled-acceptance",
        source_hash=f"owned-{split}-{number}",
        permission="self-authored",
        split=split,
    )


def tiny_training(steps=35):
    return SupervisedTrainConfig(
        steps=steps,context=16,dim=32,layers=1,
        heads=4,kv_heads=2,
        batch_size=3,gradient_accumulation=1,
        validate_every=5,validation_batches=2,
        minimum_train_windows=4,minimum_holdout_windows=4,
        lr=0.003,min_lr=0.0005,weight_decay=0.0,
        seed=23,device="cpu",
    )


class CurriculumLearningAcceptance(unittest.TestCase):
    def test_curriculum_training_improves_controlled_holdout(self):
        train=[fixture(i,"train") for i in range(4)]
        holdout=[fixture(i+100,"validation") for i in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"curriculum"
            report=run_curriculum_training(
                train,holdout,output,
                config=CurriculumTrainingConfig(
                    training=tiny_training(),strategy="curriculum"
                ),
            )
            self.assertTrue(report["weight_updates_verified"])
            self.assertGreater(report["improvement_over_baseline"],0.02)
            self.assertGreater(report["best_step"],0)
            baseline,old_meta=load_weights(output/"baseline")
            learned,new_meta=load_weights(output/"best")
            self.assertNotEqual(old_meta.weights_sha256,new_meta.weights_sha256)
            evidence=paired_evaluate(
                baseline,learned,holdout,
                EvalSettings(context=16,repetitions=50,max_error_examples=0),
            )
            self.assertGreater(evidence["loss_reduction"],0.02)
            self.assertFalse(evidence["proven_general_intelligence"])


if __name__=="__main__":
    unittest.main()
