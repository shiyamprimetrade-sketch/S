"""Controlled learning acceptance test for independent S-Brain weights.

This fixture has a deliberately simple, repeatable answer mapping with different
holdout questions. It verifies actual reduction in held-out next-byte loss
after AdamW updates. Such improvement is *not* a general intelligence claim.
"""
import tempfile
import unittest
from pathlib import Path

from s_brain.supervised_data import encode_record
from s_brain.train_supervised import SupervisedTrainConfig, run_supervised_training
from s_brain.evaluate_learning import (
    compare_saved_checkpoints, EvalSettings, pass_improvement_gate,
)


def generate_record(index,split):
    return encode_record(
        identifier=f"{split}-heldout-{index}",
        question=f"Explain the designated example for catalog item {index}.",
        answer="The sample response is consistent and independently learned.",
        category="controlled-generalization",
        source_hash=f"author-generated-{split}-{index}",
        permission="self-authored-fixture",
        split=split,
    )


class IndependentLearningAcceptance(unittest.TestCase):
    def test_train_improves_on_separate_questions(self):
        train=[generate_record(i,"train") for i in range(4)]
        holdout=[generate_record(i+100,"validation") for i in range(2)]
        config=SupervisedTrainConfig(
            steps=45,batch_size=4,context=16,
            dim=32,layers=1,heads=4,kv_heads=2,
            lr=0.004,min_lr=0.001,
            weight_decay=0.0,gradient_accumulation=1,
            validate_every=5,validation_batches=2,
            minimum_train_windows=8,
            minimum_holdout_windows=8,
            seed=13,device="cpu",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"experiment"
            report=run_supervised_training(train,holdout,root,config=config)
            self.assertTrue(report["weights_changed_each_step"])
            self.assertTrue(report["best_model_saved"],str(report))
            self.assertGreater(report["improvement_over_baseline"],0.02,str(report))
            # A separate evaluator reads checkpoints from disk and scores
            # each held-out byte exactly once, rather than trusting training logs.
            from s_brain.supervised_data import export_jsonl
            holdout_file=Path(temporary)/"holdout.jsonl"
            export_jsonl(holdout_file,holdout)
            evaluation=compare_saved_checkpoints(
                root/"baseline",root/"best",holdout_file,
                settings=EvalSettings(context=16,repetitions=50),
            )
            self.assertTrue(evaluation["weights_changed"])
            self.assertGreater(evaluation["loss_reduction"],0.02,str(evaluation))
            gate=pass_improvement_gate(
                evaluation,minimum_loss_reduction=0.02,
                max_accuracy_drop=0.10,
            )
            self.assertTrue(gate["passed"],str(gate))
            self.assertFalse(evaluation["proven_general_intelligence"])


if __name__=="__main__":
    unittest.main()
