"""Paired uniform-vs-curriculum experiment runner with independent test gate.

Every training arm uses the same architecture, initialization seed, number of
optimizer steps and the same approved training/validation records. Each
arm trains its own neural weights. Paired results compare *fixed validation
loss*, then optionally evaluate a third test split after selection is frozen.

The held-out TEST split is for final reporting only. It must never be used for
curriculum allocation, checkpoint promotion, or choosing the winning arm.
Runs are never resumed silently and existing experiment folders are protected.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Callable

from .curriculum_training import CurriculumTrainingConfig, run_curriculum_training
from .dataset_versions import audit_release, create_release
from .supervised_data import SupervisedRecord, import_jsonl, fingerprint_dataset
from .train_supervised import SupervisedTrainConfig, check_split_integrity
from .checkpoint import load_manifest
from .evaluate_learning import (
    EvalSettings, evaluate_checkpoint, compare_saved_checkpoints,
)


@dataclasses.dataclass(frozen=True)
class ComparisonPlan:
    seeds: tuple[int, ...] = (13, 29, 47)
    steps: int = 40
    min_validation_gain: float = 0.0
    minimum_paired_seeds: int = 2
    max_runs: int = 40
    run_blind_test: bool = False

    def validate(self):
        if not self.seeds or len(set(self.seeds))!=len(self.seeds):
            raise ValueError("Seeds must be unique and nonempty")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("Invalid seed")
        if self.steps < 1 or self.minimum_paired_seeds < 1:
            raise ValueError("Invalid plan counters")
        if self.max_runs < 2*len(self.seeds):
            raise ValueError("Experiment exceeds run cap")
        if not math.isfinite(self.min_validation_gain):
            raise ValueError("Nonfinite validation threshold")


@dataclasses.dataclass(frozen=True)
class ArmOutcome:
    seed: int
    strategy: str
    baseline_loss: float
    best_loss: float
    final_loss: float
    improvement: float
    best_step: int
    checkpoint_path: str
    fingerprint: str
    gradient_updates_verified: bool

    def as_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class PairedOutcome:
    seed: int
    uniform: ArmOutcome
    curriculum: ArmOutcome
    curriculum_loss_advantage: float

    def as_dict(self):
        return dataclasses.asdict(self)


def eligible_checkpoint(report: dict, root: Path) -> Path:
    if report["best_step"] > 0 and (root/"best"/"manifest.json").exists():
        return root/"best"
    return root/"baseline"


def parse_arm(report: dict,seed:int,strategy:str,root:Path) -> ArmOutcome:
    if report["strategy"]!=strategy:
        raise ValueError("Trainer used unexpected strategy")
    if not report["weight_updates_verified"]:
        raise RuntimeError("Trainer failed to update S weights")
    checkpoint=eligible_checkpoint(report,root)
    manifest=load_manifest(checkpoint,verify_digest=True)
    if manifest.seed!=seed:
        raise RuntimeError("Checkpoint seed mismatch")
    if manifest.dataset_fingerprint!=report["dataset_fingerprint"]:
        raise RuntimeError("Checkpoint dataset mismatch")
    return ArmOutcome(
        seed=seed,strategy=strategy,
        baseline_loss=report["baseline_validation_loss"],
        best_loss=report["best_validation_loss"],
        final_loss=report["final_validation_loss"],
        improvement=report["improvement_over_baseline"],
        best_step=report["best_step"],
        checkpoint_path=str(checkpoint),
        fingerprint=manifest.dataset_fingerprint,
        gradient_updates_verified=True,
    )


def run_paired_experiment(
    train: list[SupervisedRecord],
    validation: list[SupervisedRecord],
    root: Path,
    *,
    plan: ComparisonPlan | None = None,
    base_config: SupervisedTrainConfig | None = None,
    trainer: Callable = run_curriculum_training,
) -> dict:
    plan=plan or ComparisonPlan()
    plan.validate()
    config=base_config or SupervisedTrainConfig()
    config.validate()
    integrity=check_split_integrity(train,validation)
    if any(record.split!="train" for record in train):
        raise ValueError("Invalid train split")
    if any(record.split!="validation" for record in validation):
        raise ValueError("Use validation-only data for arm selection")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Experiment directory already contains outputs")
    results=[]
    for seed in plan.seeds:
        arms={}
        for strategy in ("uniform","curriculum"):
            destination=root/f"seed-{seed:06d}"/strategy
            training=dataclasses.replace(config,steps=plan.steps,seed=seed)
            chosen=CurriculumTrainingConfig(
                training=training,strategy=strategy,
            )
            report=trainer(
                train,validation,destination,config=chosen
            )
            arms[strategy]=parse_arm(report,seed,strategy,destination)
        control,treatment=arms["uniform"],arms["curriculum"]
        if control.fingerprint!=treatment.fingerprint:
            raise RuntimeError("Paired experiment used different datasets")
        if abs(control.baseline_loss-treatment.baseline_loss)>1e-5:
            raise RuntimeError("Different initial weights or baseline evaluation")
        results.append(PairedOutcome(
            seed,control,treatment,control.best_loss-treatment.best_loss,
        ))
    report={
        "plan":dataclasses.asdict(plan),
        "base_config":dataclasses.asdict(config),
        "integrity":integrity,
        "pairs":[item.as_dict() for item in results],
        "summary":summarize_pairs(results),
        "selection_scope":"validation_only",
        "test_used_for_selection":False,
        "note":"No advantage is claimed unless paired measurements support it",
    }
    root.mkdir(parents=True,exist_ok=True)
    save_report(root/"comparison.json",report)
    return report


def summarize_pairs(pairs:list[PairedOutcome]) -> dict:
    if not pairs:
        raise ValueError("No paired experiments")
    deltas=[x.curriculum_loss_advantage for x in pairs]
    return {
        "pairs":len(pairs),
        "mean_curriculum_advantage":statistics.mean(deltas),
        "median_curriculum_advantage":statistics.median(deltas),
        "curriculum_better_seeds":sum(x>0 for x in deltas),
        "uniform_better_seeds":sum(x<0 for x in deltas),
        "ties":sum(x==0 for x in deltas),
        "min_advantage":min(deltas),
        "max_advantage":max(deltas),
        "advantage_by_seed":{
            str(x.seed):x.curriculum_loss_advantage for x in pairs
        },
    }


def choose_validation_arm(report:dict,minimum_gain:float=0.0)->dict:
    if report.get("selection_scope")!="validation_only":
        raise ValueError("Report does not preserve validation-only selection")
    values=report["pairs"]
    if not values:
        raise ValueError("No runs to choose from")
    differences=[row["curriculum_loss_advantage"] for row in values]
    mean=statistics.mean(differences)
    selected="curriculum" if mean>minimum_gain else "uniform"
    return {
        "selected_strategy":selected,
        "mean_validation_loss_advantage":mean,
        "criterion":"paired mean best-validation cross-entropy",
        "test_set_consulted":False,
        "minimum_curriculum_advantage":minimum_gain,
    }


def verify_paired_baselines(pairs:list[PairedOutcome],tolerance=1e-5):
    issues=[]
    for item in pairs:
        if item.uniform.seed!=item.curriculum.seed:
            issues.append(f"seed_mismatch:{item.seed}")
        if item.uniform.fingerprint!=item.curriculum.fingerprint:
            issues.append(f"dataset_mismatch:{item.seed}")
        if abs(item.uniform.baseline_loss-item.curriculum.baseline_loss)>tolerance:
            issues.append(f"initialization_mismatch:{item.seed}")
    return {"valid":not issues,"issues":issues}


def blind_test_check(
    checkpoint_path: Path,
    test: list[SupervisedRecord],
    *,
    minimum_records:int=1,
    device:str="cpu",
) -> dict:
    """One-time external test measurement; cannot determine checkpoint choice."""
    if len(test)<minimum_records:
        raise ValueError("Not enough blinded test examples")
    if any(item.split!="test" for item in test):
        raise ValueError("Blind test requires test-only records")
    from .checkpoint import load_weights
    model,manifest=load_weights(checkpoint_path,device)
    metrics=evaluate_checkpoint(
        model,test,EvalSettings(
            context=model.cfg.context,repetitions=50,max_error_examples=0
        ),
    )
    return {
        "checkpoint_weights_sha256":manifest.weights_sha256,
        "test_data_fingerprint":fingerprint_dataset(test),
        "test_metrics":metrics,
        "model_selection_performed":False,
        "note":"One-time blind report; not used to select the best arm",
    }


def audit_test_boundary(
    train: list[SupervisedRecord],
    validation: list[SupervisedRecord],
    test: list[SupervisedRecord],
) -> dict:
    from .dataset_versions import DatasetReleasePolicy
    return audit_release(
        train,validation,test,
        DatasetReleasePolicy(
            minimum_train=1,minimum_validation=1,minimum_test=1,
        ),
    )


def summarize_gradient_evidence(root:Path,seeds:tuple[int,...]):
    from .learning_curves import read_metrics,learning_diagnostics
    result={}
    for seed in seeds:
        for strategy in ("uniform","curriculum"):
            path=root/f"seed-{seed:06d}"/strategy/"metrics.jsonl"
            rows=read_metrics(path)
            result[f"{seed}:{strategy}"]=learning_diagnostics(rows)
    return result


def save_report(path:Path,report:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")


def validate_experiment_report(report:dict)->None:
    if report["test_used_for_selection"]:
        raise ValueError("Blind test was used for model selection")
    if report["selection_scope"]!="validation_only":
        raise ValueError("Unexpected checkpoint selection source")
    pairs=report["pairs"]
    if not pairs:
        raise ValueError("No measured model runs")
    if any(
        not pair["uniform"]["gradient_updates_verified"]
        or not pair["curriculum"]["gradient_updates_verified"]
        for pair in pairs
    ):
        raise ValueError("Some optimizer runs never changed weights")


def main(argv=None):
    p=argparse.ArgumentParser(description="Paired real training: curriculum vs uniform")
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[13,29])
    p.add_argument("--steps",type=int,default=40)
    p.add_argument("--context",type=int,default=64)
    p.add_argument("--dim",type=int,default=128)
    p.add_argument("--layers",type=int,default=2)
    args=p.parse_args(argv)
    report=run_paired_experiment(
        import_jsonl(args.train),import_jsonl(args.validation),
        args.output,
        plan=ComparisonPlan(seeds=tuple(args.seeds),steps=args.steps),
        base_config=SupervisedTrainConfig(
            context=args.context,dim=args.dim,layers=args.layers,
        ),
    )
    print(json.dumps({
        "summary":report["summary"],
        "selection":choose_validation_arm(report),
    },indent=2))


if __name__=="__main__":
    main()
