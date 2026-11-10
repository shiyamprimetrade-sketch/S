"""Cycle 7 regression tests: actual curriculum training, immutable datasets,
paired arm comparison and blind holdout protocol.

Fixture models are intentionally tiny to keep Linux CPU tests inexpensive.
These checks establish learning code execution, *not* broad intelligence.
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import torch

from s_brain.supervised_data import (
    encode_record,export_jsonl,import_jsonl,
)
from s_brain.train_supervised import SupervisedTrainConfig
from s_brain.curriculum_training import (
    CurriculumTrainingConfig,ActualCurriculumBatcher,
    run_curriculum_training,optimize_one_step,
)
from s_brain.curriculum_experiments import (
    ComparisonPlan,ArmOutcome,PairedOutcome,summarize_pairs,
    choose_validation_arm,verify_paired_baselines,
    audit_test_boundary,validate_experiment_report,
    run_paired_experiment,parse_arm,
)
from s_brain.dataset_versions import (
    DatasetReleasePolicy,DatasetRegistry,audit_release,create_part,
    create_release,release_digest,DatasetRelease,
)
from s_brain.blind_benchmarks import (
    BenchmarkProtocol,BenchmarkSeal,seal_benchmark,seal_to_file,
    load_seal,verify_sealed_test,acceptance_checks,
    evaluate_blind_checkpoint,compare_blind_reports,
)


def record(i,split="train",category="math"):
    return encode_record(
        identifier=f"{split}-{i}",
        question=f"Explain independent example number {i}.",
        answer=f"The example {i} uses repeated concepts and verifiable steps.",
        category=category,
        source_hash=f"unique-{split}-{i}",
        permission="self-authored-test",
        split=split,
    )


def training_config(steps=2,seed=17):
    return SupervisedTrainConfig(
        steps=steps,batch_size=2,context=16,dim=32,
        layers=1,heads=4,kv_heads=2,
        gradient_accumulation=1,validate_every=1,
        validation_batches=1,seed=seed,
        minimum_train_windows=2,minimum_holdout_windows=2,
        lr=0.003,min_lr=0.001,
        device="cpu",
    )


def fixture_sets():
    return (
        [record(i,category=["math","coding","science"][i % 3]) for i in range(5)],
        [record(i+50,split="validation") for i in range(2)],
        [record(i+100,split="test") for i in range(2)],
    )


class ActualCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.train,self.validation,self.test=fixture_sets()

    def test_config_is_valid(self):
        CurriculumTrainingConfig(training=training_config()).validate()

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            CurriculumTrainingConfig(
                training=training_config(),strategy="random"
            ).validate()

    def test_sampler_is_deterministic(self):
        one=ActualCurriculumBatcher(
            self.train,16,seed=11,strategy="curriculum"
        )
        two=ActualCurriculumBatcher(
            self.train,16,seed=11,strategy="curriculum"
        )
        x1,y1,c1,m1=one.draw(0,8,3,"cpu")
        x2,y2,c2,m2=two.draw(0,8,3,"cpu")
        self.assertTrue(torch.equal(x1,x2))
        self.assertTrue(torch.equal(y1,y2))
        self.assertEqual(m1,m2)

    def test_curriculum_sampling_uses_stage(self):
        sampler=ActualCurriculumBatcher(
            self.train,16,seed=12,strategy="curriculum"
        )
        _,_,_,metadata=sampler.draw(0,10,3,"cpu")
        self.assertEqual(metadata["stage"],"foundation")
        _,_,_,metadata=sampler.draw(9,10,3,"cpu")
        self.assertEqual(metadata["stage"],"balanced")

    def test_uniform_control_has_no_stages(self):
        sampler=ActualCurriculumBatcher(
            self.train,16,seed=12,strategy="uniform"
        )
        _,_,_,metadata=sampler.draw(5,10,2,"cpu")
        self.assertEqual(metadata["stage"],"uniform")

    def test_holdout_cannot_enter_training_sampler(self):
        with self.assertRaises(ValueError):
            ActualCurriculumBatcher(
                self.train+self.validation,16,seed=1,
                strategy="curriculum",
            )

    def test_sampling_masks_answer_targets(self):
        sampler=ActualCurriculumBatcher(
            self.train,16,seed=15,strategy="curriculum"
        )
        x,y,count,meta=sampler.draw(0,3,4,"cpu")
        self.assertEqual(tuple(x.shape),(4,16))
        self.assertEqual(int((y!=-100).sum()),count)
        self.assertEqual(sum(meta["categories"].values()),4)

    def test_negative_step_rejected(self):
        sampler=ActualCurriculumBatcher(
            self.train,16,seed=15,strategy="curriculum"
        )
        with self.assertRaises(ValueError):
            sampler.draw(-1,5,2,"cpu")

    def test_sampler_summary_tracks_real_draws(self):
        sampler=ActualCurriculumBatcher(
            self.train,16,seed=15,strategy="curriculum"
        )
        sampler.draw(0,5,3,"cpu")
        self.assertEqual(sum(sampler.summary()["record_counts"].values()),3)

    def test_real_uniform_weights_change(self):
        with tempfile.TemporaryDirectory() as directory:
            report=run_curriculum_training(
                self.train,self.validation,
                Path(directory)/"uniform",
                config=CurriculumTrainingConfig(
                    training=training_config(),strategy="uniform",
                )
            )
            self.assertEqual(report["strategy"],"uniform")
            self.assertTrue(report["weight_updates_verified"])
            self.assertTrue((Path(directory)/"uniform"/"final"/"model.safetensors").exists())

    def test_real_curriculum_weights_change(self):
        with tempfile.TemporaryDirectory() as directory:
            report=run_curriculum_training(
                self.train,self.validation,
                Path(directory)/"curriculum",
                config=CurriculumTrainingConfig(
                    training=training_config(),strategy="curriculum",
                )
            )
            self.assertEqual(report["strategy"],"curriculum")
            self.assertEqual(report["optimizer_steps"],2)
            self.assertTrue(report["weight_updates_verified"])
            self.assertGreater(report["supervised_tokens"],0)
            self.assertTrue((Path(directory)/"curriculum"/"metrics.jsonl").exists())

    def test_existing_destination_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/"used"
            target.mkdir()
            (target/"keep").write_text("don't overwrite")
            with self.assertRaises(FileExistsError):
                run_curriculum_training(
                    self.train,self.validation,target,
                    config=CurriculumTrainingConfig(training=training_config()),
                )

    def test_metrics_record_actual_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/"model"
            run_curriculum_training(
                self.train,self.validation,target,
                config=CurriculumTrainingConfig(training=training_config()),
            )
            metrics=[json.loads(line) for line in
                     (target/"metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(metrics),2)
            self.assertTrue(all(row["changed_weight_l2"]>0 for row in metrics))
            self.assertTrue(all(row["gradient_norm"]>0 for row in metrics))
            selection=[json.loads(line) for line in
                       (target/"selections.jsonl").read_text().splitlines()]
            self.assertEqual(len(selection),2)


class DatasetReleaseTests(unittest.TestCase):
    def setUp(self):
        self.train,self.validation,self.test=fixture_sets()

    def files(self,directory):
        paths=[]
        for split,rows in zip(
            ("train","validation","test"),
            (self.train,self.validation,self.test),
        ):
            path=Path(directory)/f"{split}.jsonl"
            export_jsonl(path,rows)
            paths.append(path)
        return paths

    def release(self,paths):
        return create_release(
            *paths,name="experiment",version="v1",
            author="human",license_description="self-authored",
            review_protocol="manual-approval",timestamp=1234,
        )

    def test_split_audit_passes(self):
        self.assertTrue(audit_release(
            self.train,self.validation,self.test
        )["accepted"])

    def test_split_audit_rejects_duplicate_id(self):
        self.assertFalse(audit_release(
            self.train,self.validation,[dataclasses.replace(self.test[0],
                                      identifier=self.train[0].identifier)]
        )["accepted"])

    def test_split_audit_rejects_question_leakage(self):
        duplicate=dataclasses.replace(
            self.test[0],
            prompt_bytes=self.train[0].prompt_bytes,
        )
        self.assertFalse(audit_release(
            self.train,self.validation,[duplicate]
        )["accepted"])

    def test_split_audit_rejects_shared_source_hash(self):
        duplicate=dataclasses.replace(
            self.test[0],source_hash=self.train[0].source_hash,
        )
        self.assertFalse(audit_release(
            self.train,self.validation,[duplicate]
        )["accepted"])

    def test_release_has_stable_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            paths=self.files(folder)
            first=self.release(paths)
            second=self.release(paths)
            self.assertEqual(first.release_fingerprint,second.release_fingerprint)
            self.assertEqual(first.train.record_count,5)

    def test_version_write_once(self):
        with tempfile.TemporaryDirectory() as folder:
            release=self.release(self.files(folder))
            registry=DatasetRegistry(Path(folder)/"versions")
            registry.register(release)
            with self.assertRaises(FileExistsError):
                registry.register(release)

    def test_registry_read_and_verify(self):
        with tempfile.TemporaryDirectory() as folder:
            release=self.release(self.files(folder))
            registry=DatasetRegistry(Path(folder)/"versions")
            registry.register(release)
            loaded=registry.load("experiment","v1",verify_files=True)
            self.assertEqual(loaded.release_fingerprint,release.release_fingerprint)

    def test_tampered_data_file_fails_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            files=self.files(folder)
            registry=DatasetRegistry(Path(folder)/"versions")
            registry.register(self.release(files))
            files[0].write_text(files[0].read_text()+"\n")
            with self.assertRaises(ValueError):
                registry.load("experiment","v1",verify_files=True)

    def test_tampered_manifest_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            registry=DatasetRegistry(Path(folder)/"versions")
            path=registry.register(self.release(self.files(folder)))
            raw=json.loads(path.read_text())
            raw["version"]="v2"
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                registry.load("experiment","v1")

    def test_invalid_dataset_name_rejected(self):
        registry=DatasetRegistry(Path("/tmp")/"example")
        with self.assertRaises(ValueError):
            registry.path_for("../escape","v1")

    def test_version_list(self):
        with tempfile.TemporaryDirectory() as folder:
            registry=DatasetRegistry(Path(folder)/"versions")
            registry.register(self.release(self.files(folder)))
            self.assertEqual(registry.versions("experiment"),["v1"])

    def test_versions_compare(self):
        with tempfile.TemporaryDirectory() as folder:
            paths=self.files(folder)
            registry=DatasetRegistry(Path(folder)/"versions")
            first=self.release(paths)
            second=dataclasses.replace(first,version="v2",
                release_fingerprint=release_digest(
                    first.name,"v2",first.author,first.license_description,
                    first.review_protocol,first.parts()
                )
            )
            registry.register(first)
            registry.register(second)
            report=registry.compare("experiment","v1","v2")
            self.assertFalse(report["parts"]["test"]["content_changed"])


class BlindBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.train,self.validation,self.test=fixture_sets()

    def seal(self,folder):
        target=Path(folder)/"test.jsonl"
        export_jsonl(target,self.test)
        seal=seal_benchmark(
            target,name="private-blind",version="1.0",
            author="human",reviewer="independent-reviewer",
            license_description="self-authored",
        )
        return target,seal

    def test_seal_records_test_only(self):
        with tempfile.TemporaryDirectory() as folder:
            _,seal=self.seal(folder)
            self.assertEqual(seal.protocol.record_count,2)

    def test_seal_rejects_training_split(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"bad.jsonl"
            export_jsonl(path,self.train)
            with self.assertRaises(ValueError):
                seal_benchmark(
                    path,name="bad",version="v1",author="human",
                    reviewer="human",license_description="owned",
                )

    def test_seal_needs_reviewer(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"test.jsonl"
            export_jsonl(path,self.test)
            with self.assertRaises(ValueError):
                seal_benchmark(
                    path,name="bad",version="v1",author="human",
                    reviewer="",license_description="owned",
                )

    def test_seal_is_write_once(self):
        with tempfile.TemporaryDirectory() as folder:
            _,seal=self.seal(folder)
            path=Path(folder)/"seal.json"
            seal_to_file(path,seal)
            with self.assertRaises(FileExistsError):
                seal_to_file(path,seal)

    def test_seal_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            _,seal=self.seal(folder)
            path=Path(folder)/"seal.json"
            seal_to_file(path,seal)
            self.assertEqual(load_seal(path),seal)

    def test_seal_detects_modified_file(self):
        with tempfile.TemporaryDirectory() as folder:
            target,seal=self.seal(folder)
            target.write_text(target.read_text()+" ")
            self.assertFalse(verify_sealed_test(seal,target)["valid"])

    def test_benchmark_train_test_overlap_check(self):
        with tempfile.TemporaryDirectory() as folder:
            target,seal=self.seal(folder)
            result=verify_sealed_test(
                seal,target,train_records=self.train,
                validation_records=self.validation,
            )
            self.assertTrue(result["valid"])

    def test_benchmark_shared_source_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            target,seal=self.seal(folder)
            bad=dataclasses.replace(
                self.train[0],source_hash=self.test[0].source_hash
            )
            result=verify_sealed_test(
                seal,target,train_records=[bad],
                validation_records=self.validation,
            )
            self.assertFalse(result["valid"])

    def test_gate_accepts_valid_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            _,seal=self.seal(folder)
            metrics={
                "records":2,"tokens":100,"accuracy":0.3,
                "cross_entropy":4.0,
                "categories":{"math":{"accuracy":0.3}},
            }
            self.assertTrue(all(acceptance_checks(metrics,seal.protocol).values()))

    def test_gate_rejects_category_floor(self):
        with tempfile.TemporaryDirectory() as folder:
            _,seal=self.seal(folder)
            protocol=dataclasses.replace(
                seal.protocol,category_floors={"math":0.8}
            )
            metrics={
                "records":2,"tokens":100,"accuracy":0.5,
                "cross_entropy":4.0,
                "categories":{"math":{"accuracy":0.5}},
            }
            self.assertFalse(acceptance_checks(metrics,protocol)["category:math"])

    def test_test_set_cannot_be_used_for_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            _,seal=self.seal(folder)
            protocol=dataclasses.replace(
                seal.protocol,explicit_no_selection_usage=False
            )
            with self.assertRaises(ValueError):
                protocol.validate()

    def test_mismatched_blind_reports_rejected(self):
        baseline={
            "benchmark":{"name":"a","version":"1","fingerprint":"x"},
            "metrics":{"accuracy":0.5,"cross_entropy":2.0},
        }
        candidate={
            "benchmark":{"name":"b","version":"1","fingerprint":"y"},
            "metrics":{"accuracy":0.6,"cross_entropy":1.8},
        }
        with self.assertRaises(ValueError):
            compare_blind_reports(baseline,candidate)

    def test_real_checkpoint_blind_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            target,seal=self.seal(folder)
            report=run_curriculum_training(
                self.train,self.validation,folder/"training",
                config=CurriculumTrainingConfig(
                    training=training_config(steps=2),
                    strategy="curriculum",
                )
            )
            checked=evaluate_blind_checkpoint(
                folder/"training"/"final",target,seal,device="cpu",
                train_records=self.train,validation_records=self.validation,
            )
            self.assertTrue(checked["seal_verified"])
            self.assertFalse(checked["used_to_select_checkpoint"])
            self.assertEqual(checked["checkpoint"]["step"],2)


class PairedExperimentTests(unittest.TestCase):
    def setUp(self):
        self.train,self.validation,self.test=fixture_sets()

    def test_no_seeds_rejected(self):
        with self.assertRaises(ValueError):
            ComparisonPlan(seeds=()).validate()

    def test_no_budget_rejected(self):
        with self.assertRaises(ValueError):
            ComparisonPlan(steps=0).validate()

    def test_only_validation_allowed_for_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                run_paired_experiment(
                    self.train,self.test,Path(folder)/"experiment",
                    plan=ComparisonPlan(seeds=(1,),steps=1),
                    base_config=training_config(steps=1),
                )

    def test_aggregate_paired(self):
        uniform=ArmOutcome(1,"uniform",4,3,3,1,10,"a","dataset",True)
        curriculum=ArmOutcome(1,"curriculum",4,2.5,2.6,1.5,10,"b","dataset",True)
        paired=PairedOutcome(1,uniform,curriculum,0.5)
        result=summarize_pairs([paired])
        self.assertEqual(result["curriculum_better_seeds"],1)

    def test_paired_seed_mismatch(self):
        left=ArmOutcome(1,"uniform",4,3,3,1,10,"a","dataset",True)
        right=ArmOutcome(2,"curriculum",4,2.5,2.6,1.5,10,"b","dataset",True)
        result=verify_paired_baselines([
            PairedOutcome(1,left,right,0.5)
        ])
        self.assertFalse(result["valid"])

    def test_validation_selection_does_not_use_test(self):
        report={
            "selection_scope":"validation_only",
            "pairs":[{"curriculum_loss_advantage":-0.2},
                     {"curriculum_loss_advantage":0.1}],
        }
        outcome=choose_validation_arm(report)
        self.assertEqual(outcome["selected_strategy"],"uniform")
        self.assertFalse(outcome["test_set_consulted"])

    def test_reject_selection_on_test(self):
        with self.assertRaises(ValueError):
            choose_validation_arm({
                "selection_scope":"test","pairs":[{"curriculum_loss_advantage":2}]
            })

    def test_release_boundary_for_third_split(self):
        result=audit_test_boundary(self.train,self.validation,self.test)
        self.assertTrue(result["accepted"])

    def test_real_paired_training_updates_both_arms(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run_paired_experiment(
                self.train,self.validation,Path(folder)/"paired",
                plan=ComparisonPlan(seeds=(17,),steps=2),
                base_config=training_config(steps=2),
            )
            self.assertEqual(len(result["pairs"]),1)
            self.assertFalse(result["test_used_for_selection"])
            self.assertTrue(result["pairs"][0]["uniform"]["gradient_updates_verified"])
            self.assertTrue(result["pairs"][0]["curriculum"]["gradient_updates_verified"])
            self.assertTrue((Path(folder)/"paired"/"comparison.json").exists())
            validate_experiment_report(result)


if __name__=="__main__":
    unittest.main()
