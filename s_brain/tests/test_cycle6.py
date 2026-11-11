"""Cycle 6 deterministic regression coverage across curation, teacher review,
curriculum sampling, learning-curve summaries and checkpoint quality gates.

All generated examples are explicit fixtures; no Ollama request is sent.
These tests verify process correctness, not factual accuracy of teacher data.
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from s_brain.curation_v2 import (
    CurationPolicy, SourceRecord, curate, canonical, digest,
    word_ngrams, jaccard, near_duplicate, sensitive_indicators,
    reject_reasons, group_safe_splits, source_fingerprint,
    write_source_jsonl, read_source_jsonl, write_audit,
    assess_holdout_independence,
)
from s_brain.teacher_data import (
    TeacherSettings, TeacherQuestion, create_example, review_example,
)
from s_brain.teacher_review import (
    ReviewPolicy, ReviewAction, apply_actions, answer_flags,
    review_candidates, training_release, verify_ledger,
    write_actions, read_actions, write_ledger, read_ledger,
)
from s_brain.curriculum_v2 import (
    CurriculumStage, CurriculumPolicy, CurriculumSampler,
    weighted_choice, default_stages, validate_stages, stage_for_step,
    lexical_complexity, assign_difficulties, dry_run_curriculum,
    schedule_fingerprint,
)
from s_brain.supervised_data import encode_record
from s_brain.learning_curves import (
    CurvePlan, CurvePoint, run_directory, config_for_run, group_by_budget,
    bootstrap_mean, aggregate_curve, compare_budgets,
    read_metrics, learning_diagnostics, run_learning_curve,
)
from s_brain.train_supervised import SupervisedTrainConfig
from s_brain.checkpoint_selection import (
    SelectionPolicy, CheckpointCandidate, SelectionEvidence,
    select_candidate, rank_checkpoints, write_selection,
    prepare_promotion,
)


def source(number, text=None, split="train", review="approved"):
    return SourceRecord(
        identifier=f"s-{number}",
        text=text or f"Independent learning document {number} describes topic {number * 31}.",
        source="self-authored-test", permission="user-owned", split=split,
        category="education", review=review,
    )


def teacher(number=1, *, answer="The consistent test response is clear."):
    question = TeacherQuestion(
        identifier=f"q{number}", question=f"What is number {number}?",
        category="arithmetic", source="self-authored-test",
        permission="user-owned", split="train",
    )
    return create_example(
        question, TeacherSettings(),
        generator=lambda q, s: answer,
    )


def supervised(number, split="train", category="math"):
    return encode_record(
        identifier=f"{split}-{number}",
        question=f"Explain verified item {number} in this new exercise.",
        answer=f"Item {number} has a carefully documented synthetic answer.",
        category=category, source_hash=f"fixture-{split}-{number}",
        permission="user-owned", split=split,
    )


class CurationTests(unittest.TestCase):
    def test_canonical_whitespace(self):
        self.assertEqual(canonical(" Hello, WORLD! "), "hello world")

    def test_digest_is_stable(self):
        self.assertEqual(digest("HELLO"), digest("hello!"))

    def test_ngram_size(self):
        self.assertIn(("a", "b"), word_ngrams("a b c", 2))

    def test_empty_jaccard(self):
        self.assertEqual(jaccard(set(), set()), 1.0)

    def test_near_duplicates(self):
        text = "The same document describes a scientific experiment."
        self.assertTrue(near_duplicate(text, text, 0.9, 3))

    def test_distinct_texts_not_near_duplicates(self):
        self.assertFalse(near_duplicate(
            "Apples and fresh oranges arrive through the port",
            "Transformer tensors receive gradients from backward loss",
            0.9, 3,
        ))

    def test_secret_detection_is_metadata_only(self):
        result = sensitive_indicators("api_key=SECRET123 staff@example.com")
        self.assertEqual(result["possible_secrets"], 1)
        self.assertEqual(result["possible_emails"], 1)
        self.assertNotIn("SECRET123", str(result))

    def test_review_is_required(self):
        report = curate([source(1, review="pending")])
        self.assertEqual(report.report()["accepted"], 0)

    def test_curation_accepts_unique_reviewed(self):
        rows = [source(i) for i in range(4)]
        result = curate(rows)
        self.assertEqual(len(result.records), 4)
        self.assertTrue(all(item.accepted for item in result.decisions))

    def test_exact_duplicate_rejected(self):
        result = curate([source(1, "same long document text"), source(2, "same long document text")],
                        CurationPolicy(minimum_bytes=1))
        self.assertEqual(len(result.records), 1)
        self.assertIn("exact_duplicate", result.decisions[-1].reasons)

    def test_cross_split_duplicate_rejected(self):
        result = curate([
            source(1, "scientific computing creates repeatable results", "train"),
            source(2, "scientific computing creates repeatable results", "test"),
        ])
        self.assertEqual(len(result.records), 1)
        self.assertIn("cross_split_exact_duplicate", result.decisions[-1].reasons)

    def test_too_short_rejected(self):
        report = curate([source(1, "short")])
        self.assertIn("too_short", report.decisions[0].reasons)

    def test_permission_metadata_rejected(self):
        record = dataclasses.replace(source(1), permission="unknown")
        self.assertIn("permission_unverified", reject_reasons(record, CurationPolicy()))

    def test_email_rejected(self):
        record = source(1, "Person user@example.com wrote this private document.")
        self.assertIn("possible_email", reject_reasons(record, CurationPolicy()))

    def test_duplicate_identifier_rejected(self):
        report = curate([source(1), source(1)])
        self.assertIn("duplicate_identifier", report.decisions[-1].reasons)

    def test_safe_splits(self):
        groups = group_safe_splits([source(1), source(2, split="validation")])
        self.assertEqual(len(groups["validation"]), 1)

    def test_unsafe_splits(self):
        rows = [
            source(1, "Same content in two separate splits", "train"),
            source(2, "Same content in two separate splits", "test"),
        ]
        with self.assertRaises(ValueError):
            group_safe_splits(rows)

    def test_fingerprint_order_independent(self):
        rows = [source(1), source(2)]
        self.assertEqual(source_fingerprint(rows), source_fingerprint(list(reversed(rows))))

    def test_source_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approved.jsonl"
            rows = [source(1), source(2)]
            write_source_jsonl(path, rows)
            self.assertEqual(read_source_jsonl(path), rows)

    def test_audit_excludes_document_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = curate([source(1, "Special approved content never shown in audit")])
            path = Path(tmp) / "audit.json"
            write_audit(path, report)
            self.assertNotIn("Special approved content", path.read_text())

    def test_holdout_overlap_audit(self):
        shared = "Same research prose goes into training and validation splits."
        report = assess_holdout_independence(
            [source(1, shared)], [source(2, shared, split="validation")]
        )
        self.assertFalse(report["independent_by_ngram_check"])


class ReviewTests(unittest.TestCase):
    def action(self, item, decision="approve", reviewer="human", reason="Verified example"):
        return ReviewAction(
            item.identifier, decision, reviewer, reason,
            item.answer_sha256, item.question_sha256,
        )

    def test_candidate_is_pending(self):
        self.assertEqual(teacher().review_status, "pending")

    def test_review_requires_reviewer(self):
        action = self.action(teacher(), reviewer="")
        with self.assertRaises(ValueError):
            action.validate()

    def test_default_candidate_has_no_flags(self):
        self.assertEqual(answer_flags(teacher(), ReviewPolicy()), [])

    def test_external_link_is_flagged(self):
        item = teacher(answer="Read https://example.com and check it.")
        self.assertIn("unverified_external_link", answer_flags(item, ReviewPolicy()))

    def test_generate_review_card(self):
        rows = review_candidates([teacher()])
        self.assertEqual(rows[0]["status"], "pending")
        self.assertEqual(len(rows[0]["answer_sha256"]), 64)

    def test_approve(self):
        item = teacher()
        outcome = apply_actions([item], [self.action(item)])
        self.assertEqual(outcome.examples[0].review_status, "approved")
        self.assertEqual(outcome.events[0].reviewer, "human")

    def test_reject(self):
        item = teacher()
        outcome = apply_actions([item], [self.action(item, decision="reject")])
        self.assertEqual(outcome.examples[0].review_status, "rejected")

    def test_defer(self):
        item = teacher()
        outcome = apply_actions([item], [self.action(item, decision="defer", reason="")])
        self.assertEqual(outcome.examples[0].review_status, "pending")

    def test_stale_hash_fails(self):
        item = teacher()
        action = dataclasses.replace(self.action(item), expected_answer_sha256="changed")
        with self.assertRaises(ValueError):
            apply_actions([item], [action])

    def test_unresolved_link_cannot_be_approved(self):
        item = teacher(answer="Visit https://example.com for more details.")
        with self.assertRaises(ValueError):
            apply_actions([item], [self.action(item)])

    def test_holdout_cannot_be_approved(self):
        item = dataclasses.replace(teacher(), split="test")
        with self.assertRaises(ValueError):
            apply_actions([item], [self.action(item)])

    def test_missing_notes_fails(self):
        item = teacher()
        with self.assertRaises(ValueError):
            apply_actions([item], [self.action(item, reason="")])

    def test_training_release_only_approved(self):
        item = teacher()
        outcome = apply_actions([item], [self.action(item)])
        selected, audit = training_release(outcome)
        self.assertEqual(len(selected), 1)
        self.assertEqual(audit["eligible"], 1)

    def test_unapproved_not_eligible(self):
        item = teacher()
        with self.assertRaises(ValueError):
            training_release(apply_actions([item], []))

    def test_ledger_is_verifiable(self):
        item = teacher()
        outcome = apply_actions([item], [self.action(item)])
        report = verify_ledger(outcome.events, outcome.examples)
        self.assertTrue(report["valid"])

    def test_actions_and_events_roundtrip(self):
        item = teacher()
        with tempfile.TemporaryDirectory() as tmp:
            actions = Path(tmp) / "actions.jsonl"
            ledger = Path(tmp) / "ledger.jsonl"
            write_actions(actions, [self.action(item)])
            self.assertEqual(len(read_actions(actions)), 1)
            outcome = apply_actions([item], read_actions(actions))
            write_ledger(ledger, outcome.events)
            self.assertEqual(read_ledger(ledger), outcome.events)


class CurriculumTests(unittest.TestCase):
    def records(self):
        return [supervised(i, category=["math","coding","science"][i % 3]) for i in range(9)]

    def test_lexical_complexity(self):
        self.assertEqual(lexical_complexity("a a a"), 1/3)

    def test_stages_cover_full_schedule(self):
        validate_stages(default_stages())

    def test_stage_early(self):
        self.assertEqual(stage_for_step(0, 100, default_stages()).name, "foundation")

    def test_stage_late(self):
        self.assertEqual(stage_for_step(99, 100, default_stages()).name, "balanced")

    def test_stage_gap_rejected(self):
        bad = [CurriculumStage("a",0,0.4),CurriculumStage("b",0.5,1)]
        with self.assertRaises(ValueError):
            validate_stages(bad)

    def test_stage_weight(self):
        stage = CurriculumStage("a",0,1,easy_weight=2,category_weights={"math":3})
        self.assertEqual(stage.weight("easy","math"),6)

    def test_invalid_negative_weight(self):
        with self.assertRaises(ValueError):
            CurriculumStage("a",0,1,hard_weight=-1).validate()

    def test_assign_difficulty(self):
        assignment = assign_difficulties(self.records())
        self.assertEqual(len(assignment),9)

    def test_curriculum_sampler(self):
        sampler = CurriculumSampler(
            self.records(), context=16,
            policy=CurriculumPolicy(minimum_windows=1),
        )
        windows, ids = sampler.draw(0, 10, batch_size=3)
        self.assertEqual(len(windows),3)
        self.assertEqual(len(ids),3)

    def test_reproducible_draw(self):
        kwargs = {"context":16,"policy":CurriculumPolicy(seed=77,minimum_windows=1)}
        one = CurriculumSampler(self.records(),**kwargs)
        two = CurriculumSampler(self.records(),**kwargs)
        self.assertEqual(one.draw(0,20,5),two.draw(0,20,5))

    def test_holdout_refused(self):
        with self.assertRaises(ValueError):
            CurriculumSampler([supervised(1,split="validation")],context=16)

    def test_empty_dataset_refused(self):
        with self.assertRaises(ValueError):
            assign_difficulties([])

    def test_dry_run(self):
        sampler = CurriculumSampler(
            self.records(),context=16,
            policy=CurriculumPolicy(minimum_windows=1),
        )
        report = dry_run_curriculum(sampler,steps=12,batch_size=3)
        self.assertEqual(sum(report["stage_samples"].values()),36)

    def test_fingerprint_stable(self):
        self.assertEqual(
            schedule_fingerprint(default_stages(),CurriculumPolicy()),
            schedule_fingerprint(default_stages(),CurriculumPolicy()),
        )


def point(seed, steps, baseline, best):
    return CurvePoint(
        seed,steps,baseline,best,best,steps,steps,True,
        "train-hash","holdout-hash",f"runs/{seed}/{steps}"
    )


class LearningCurveTests(unittest.TestCase):
    def test_invalid_plan(self):
        with self.assertRaises(ValueError):
            CurvePlan(seeds=(),steps=(1,)).validate()

    def test_run_count_cap(self):
        with self.assertRaises(ValueError):
            CurvePlan(seeds=(1,2,3),steps=(10,20),maximum_runs=5).validate()

    def test_run_directory(self):
        self.assertIn("seed-000002",str(run_directory(Path("out"),2,10)))

    def test_bootstrap_interval(self):
        value = bootstrap_mean([0.1,0.2,0.3],draws=50)
        self.assertLessEqual(value["lower"],value["upper"])

    def test_aggregate(self):
        values = [point(1,10,4,3),point(2,10,4,3.5)]
        rows = aggregate_curve(values)
        self.assertEqual(rows[0]["steps"],10)
        self.assertEqual(rows[0]["improved_repeats"],2)

    def test_compare_budgets(self):
        values = [
            point(1,10,4,3),point(2,10,4,3.5),
            point(1,20,4,2.5),point(2,20,4,2.6),
        ]
        report = compare_budgets(values)
        self.assertTrue(report["more_steps_better_on_all_seeds"])

    def test_read_metrics_monotonic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"metrics.jsonl"
            path.write_text(
                '{"step":1,"train_loss":3,"best_validation_loss":3}\n'
                '{"step":2,"train_loss":2,"best_validation_loss":2}\n'
            )
            self.assertEqual(len(read_metrics(path)),2)

    def test_diagnostics(self):
        entries = [
            {"step":1,"train_loss":3,"best_validation_loss":2,
             "validation_loss":2.5,"parameter_update_l2":0.1},
            {"step":2,"train_loss":2,"best_validation_loss":2,
             "validation_loss":2.0,"parameter_update_l2":0.1},
        ]
        stats = learning_diagnostics(entries)
        self.assertTrue(stats["all_updates_nonzero"])

    def test_mock_trainer_no_heavy_training(self):
        train = [supervised(1)]
        holdout = [supervised(2,split="validation")]
        def fake_trainer(train_rows,holdout_rows,directory,config):
            return {
                "optimizer_steps":config.steps,
                "baseline_validation_loss":3.0,
                "best_validation_loss":2.0,
                "final_validation_loss":2.1,
                "best_step":config.steps,
                "weights_changed_each_step":True,
                "integrity":{
                    "train_fingerprint":"train-fp",
                    "holdout_fingerprint":"holdout-fp",
                },
            }
        with tempfile.TemporaryDirectory() as tmp:
            result = run_learning_curve(
                train,holdout,Path(tmp),
                plan=CurvePlan(seeds=(1,2),steps=(3,4)),
                base_config=SupervisedTrainConfig(
                    steps=1,batch_size=1,context=16,dim=32,
                    layers=1,heads=4,kv_heads=2,
                ),
                trainer=fake_trainer,
            )
            self.assertEqual(len(result["points"]),4)
            self.assertTrue((Path(tmp)/"learning_curve.json").exists())


class CheckpointSelectionTests(unittest.TestCase):
    def make_candidate(self, path, digest="a"*64, step=0, fingerprint="dataset"):
        return CheckpointCandidate(
            Path(path),digest,fingerprint,"architecture",step,2.0
        )

    def make_evidence(self,path,loss,accuracy=0.7,fp="test"):
        return SelectionEvidence(
            checkpoint_path=str(path),
            evaluation_sha256="1"*64,
            holdout_fingerprint=fp,
            accuracy=accuracy,cross_entropy=loss,
            records=20,tokens=500,
        )

    def decision(self, *, loss=1.5, accuracy=0.75, fingerprint="test"):
        before = self.make_candidate("baseline",step=0)
        after = self.make_candidate("candidate",digest="b"*64,step=10)
        return select_candidate(
            before,after,
            self.make_evidence(before.path,2.0),
            self.make_evidence(after.path,loss,accuracy,fp=fingerprint),
        )

    def test_selection_accepts_improvement(self):
        self.assertTrue(self.decision().accepted)

    def test_loss_regression_rejected(self):
        decision = self.decision(loss=2.2)
        self.assertFalse(decision.accepted)
        self.assertFalse(decision.checks["loss_improved"])

    def test_changed_holdout_rejected(self):
        decision = self.decision(fingerprint="new-holdout")
        self.assertFalse(decision.checks["same_holdout"])

    def test_accuracy_drop_rejected(self):
        decision = self.decision(accuracy=0.3)
        self.assertFalse(decision.checks["accuracy_not_regressed"])

    def test_unchanged_weights_rejected(self):
        before = self.make_candidate("baseline")
        after = self.make_candidate("candidate",step=10)
        result = select_candidate(
            before,after,
            self.make_evidence("baseline",2),
            self.make_evidence("candidate",1),
        )
        self.assertFalse(result.checks["weights_changed"])

    def test_different_architecture_rejected(self):
        before = self.make_candidate("baseline")
        after = dataclasses.replace(
            self.make_candidate("candidate",digest="b"*64,step=5),
            architecture_fingerprint="different",
        )
        result = select_candidate(
            before,after,
            self.make_evidence("baseline",2),
            self.make_evidence("candidate",1),
        )
        self.assertFalse(result.checks["same_architecture"])

    def test_human_approval_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PermissionError):
                prepare_promotion(self.decision(),Path(tmp)/"selected.json")

    def test_promotion_is_inert_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = prepare_promotion(
                self.decision(),Path(tmp)/"selected.json",
                operator_approved=True,
            )
            self.assertEqual(result["status"],"selected_not_deployed")

    def test_failed_gate_never_promotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                prepare_promotion(
                    self.decision(loss=3),Path(tmp)/"selected.json",
                    operator_approved=True,
                )

    def test_selection_report(self):
        report = self.decision().report()
        self.assertIn("deltas",report)

    def test_rank_checkpoints(self):
        candidates = [
            self.make_candidate("a"),self.make_candidate("b",digest="b"*64),
        ]
        evidence = {
            "a":self.make_evidence("a",2),
            "b":self.make_evidence("b",1),
        }
        ranking = rank_checkpoints(candidates,evidence)
        self.assertEqual(ranking[0]["checkpoint"]["path"],"b")

    def test_evidence_invalid_accuracy(self):
        with self.assertRaises(ValueError):
            self.make_evidence("x",1,accuracy=2).validate()


if __name__ == "__main__":
    unittest.main()
