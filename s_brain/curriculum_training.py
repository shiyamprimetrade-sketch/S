"""Actual optimizer integration for curriculum-guided S-Brain training.

The Cycle 6 curriculum was only a dry-run sampler. This engine draws real
TrainingWindows from CurriculumSampler inside each gradient accumulation step,
computes answer-only cross-entropy and updates independent SBrain weights.
Uniform mode is a matched control: identical model initialization, optimizer,
validation windows, update count, learning-rate schedule and gradient clipping.
No test examples enter the sampler. Every run stores baseline/best/final
checkpoints and a provenance manifest.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch

from .checkpoint import save_checkpoint
from .curriculum_v2 import CurriculumPolicy, CurriculumSampler, default_stages
from .model import SBrain
from .supervised_data import (
    SupervisedRecord, WindowSampler, build_windows, fingerprint_dataset,
    import_jsonl, masked_cross_entropy,
)
from .train_supervised import (
    SupervisedTrainConfig, check_split_integrity, cosine_lr,
    eval_windows, model_weights, resolve_device, set_reproducibility,
    weight_update_l2,
)


@dataclasses.dataclass(frozen=True)
class CurriculumTrainingConfig:
    training: SupervisedTrainConfig = dataclasses.field(
        default_factory=SupervisedTrainConfig
    )
    strategy: str = "curriculum"
    category_balancing: bool = True
    record_selections: bool = True
    maximum_selection_log: int = 10000

    def validate(self):
        self.training.validate()
        if self.strategy not in ("curriculum", "uniform"):
            raise ValueError("strategy must be curriculum or uniform")
        if self.maximum_selection_log < 0:
            raise ValueError("Invalid selection log quota")


@dataclasses.dataclass
class OptimizationRecord:
    step: int
    strategy: str
    stage: str
    train_loss: float
    validation_loss: float | None
    gradient_norm: float
    changed_weight_l2: float
    learned_tokens: int
    learning_rate: float
    selected_categories: dict[str, int]
    selected_difficulty: dict[str, int]
    wall_time: float

    def as_dict(self):
        return dataclasses.asdict(self)


class ActualCurriculumBatcher:
    """Sample byte windows with or without difficulty scheduling.

    The curriculum and uniform samplers are both seeded. A uniform run
    samples from all windows with equal window probability, while
    curriculum mode samples a record by difficulty/category weighting,
    then one of its windows. This is a legitimate comparison of policies
    but not identical sampling distributions between modes.
    """

    def __init__(
        self, records: list[SupervisedRecord], context: int,
        *, seed: int, strategy: str,
        category_balancing: bool = True,
    ):
        if strategy not in ("uniform", "curriculum"):
            raise ValueError("Invalid training strategy")
        if any(item.split != "train" for item in records):
            raise ValueError("Holdout leakage: nontrain record in sampler")
        self.strategy = strategy
        self.context = context
        self.records = list(records)
        self.by_identifier = {row.identifier: row for row in records}
        if len(self.by_identifier) != len(records):
            raise ValueError("Duplicate training IDs")
        windows = build_windows(records, context, method="dense")
        if not windows:
            raise ValueError("Training records do not yield complete windows")
        self.uniform = WindowSampler(windows, context, seed=seed + 1)
        self.curriculum = CurriculumSampler(
            records, context=context,
            policy=CurriculumPolicy(
                seed=seed + 1, minimum_windows=1,
                preserve_category_coverage=category_balancing,
            ),
        )
        self.signals = self.curriculum.signals
        self.selection_counter = {}
        self.drawn_tokens = 0

    def draw(
        self, step: int, total_steps: int, batch_size: int, device: str,
    ) -> tuple[torch.Tensor, torch.Tensor, int, dict]:
        if not 0 <= step < total_steps:
            raise ValueError("Invalid training schedule position")
        if batch_size < 1:
            raise ValueError("Invalid batch size")
        if self.strategy == "curriculum":
            windows, identifiers = self.curriculum.draw(step, total_steps, batch_size)
            stage = next(
                stage.name for stage in self.curriculum.stages
                if stage.start_fraction <= step / total_steps < stage.end_fraction
            )
        else:
            windows = self.uniform.random.choices(
                self.uniform.windows, k=batch_size
            )
            identifiers = [item.identifier for item in windows]
            stage = "uniform"
        inputs = torch.tensor(
            [item.token_ids for item in windows],
            dtype=torch.long, device=device,
        )
        targets = torch.tensor(
            [item.target_ids for item in windows],
            dtype=torch.long, device=device,
        )
        learned = sum(item.supervised_tokens for item in windows)
        self.drawn_tokens += learned
        categories = {}
        difficulty = {}
        for identifier in identifiers:
            category = self.by_identifier[identifier].category
            categories[category] = categories.get(category, 0) + 1
            bucket = self.signals[identifier].bucket
            difficulty[bucket] = difficulty.get(bucket, 0) + 1
            self.selection_counter[identifier] = (
                self.selection_counter.get(identifier, 0) + 1
            )
        return inputs, targets, learned, {
            "stage": stage, "identifiers": identifiers,
            "categories": categories, "difficulty": difficulty,
        }

    def summary(self):
        return {
            "strategy": self.strategy,
            "drawn_tokens": self.drawn_tokens,
            "record_counts": dict(sorted(self.selection_counter.items())),
            "curriculum": self.curriculum.summary(),
        }


def _serialize(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _append(path: Path, item: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, allow_nan=False) + "\n")


def optimize_one_step(
    model: SBrain,
    optimizer,
    sampler: ActualCurriculumBatcher,
    config: SupervisedTrainConfig,
    *,
    index: int,
) -> dict:
    """Backpropagate curriculum-sampled answers, then update actual weights."""
    model.train()
    device = str(next(model.parameters()).device)
    optimizer.zero_grad(set_to_none=True)
    before = model_weights(model)
    loss_total = 0.0
    total_tokens = 0
    category_counts = {}
    difficulty_counts = {}
    stages = []
    identifiers = []
    for _ in range(config.gradient_accumulation):
        inputs, targets, tokens, selection = sampler.draw(
            index - 1, config.steps, config.batch_size, device,
        )
        logits, _ = model(inputs)
        loss = masked_cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite curriculum loss")
        (loss / config.gradient_accumulation).backward()
        loss_total += float(loss.detach()) / config.gradient_accumulation
        total_tokens += tokens
        for key, count in selection["categories"].items():
            category_counts[key] = category_counts.get(key, 0) + count
        for key, count in selection["difficulty"].items():
            difficulty_counts[key] = difficulty_counts.get(key, 0) + count
        stages.append(selection["stage"])
        identifiers.extend(selection["identifiers"])
    gradient = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.gradient_clip,
    )
    if not torch.isfinite(gradient):
        raise RuntimeError("Nonfinite gradient")
    optimizer.step()
    change = weight_update_l2(before, model)
    if not math.isfinite(change) or change <= 0:
        raise RuntimeError("No measurable optimizer weight update")
    return {
        "train_loss": loss_total,
        "gradient_norm": float(gradient),
        "changed_weight_l2": change,
        "learned_tokens": total_tokens,
        "stage": stages[0],
        "selected_categories": category_counts,
        "selected_difficulty": difficulty_counts,
        "selected_identifiers": identifiers,
    }


def run_curriculum_training(
    train: list[SupervisedRecord],
    validation: list[SupervisedRecord],
    destination: Path,
    *,
    config: CurriculumTrainingConfig | None = None,
) -> dict:
    """Fully train a new model, including fixed validation checkpoint gates.

    This is an independent training run, not an in-place modification to the
    existing S process or a live Ollama model. Each run begins from a seeded
    SBrain initialization. The final model is always saved; best is saved
    only when it beats the untrained baseline.
    """
    config = config or CurriculumTrainingConfig()
    config.validate()
    training = config.training
    integrity = check_split_integrity(train, validation)
    if len({row.identifier for row in train + validation}) != len(train + validation):
        raise ValueError("Duplicate identifiers between train and validation")
    holdout = build_windows(validation, training.context, method="dense")
    source_windows = build_windows(train, training.context, method="dense")
    if len(holdout) < training.minimum_holdout_windows:
        raise ValueError("Not enough held-out tokens")
    if len(source_windows) < training.minimum_train_windows:
        raise ValueError("Not enough approved training tokens")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Refusing to overwrite existing experiment")
    destination.mkdir(parents=True, exist_ok=True)
    set_reproducibility(training.seed)
    device = resolve_device(training.device)
    model = SBrain(training.model_config()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.lr, weight_decay=training.weight_decay,
    )
    sampler = ActualCurriculumBatcher(
        train, training.context, seed=training.seed, strategy=config.strategy,
        category_balancing=config.category_balancing,
    )
    dataset_identity = hashlib.sha256(
        (integrity["train_fingerprint"] + ":" +
         integrity["holdout_fingerprint"]).encode("utf-8")
    ).hexdigest()
    baseline = eval_windows(
        model, holdout, training, seed=training.seed + 7
    )["cross_entropy"]
    save_checkpoint(
        destination / "baseline", model, model.cfg,
        step=0, train_tokens=0, train_loss=baseline,
        validation_loss=baseline, seed=training.seed,
        dataset_fingerprint=dataset_identity,
        extras={"strategy": config.strategy, "phase": "baseline"},
    )
    best, best_step, learned = baseline, 0, 0
    records, started = [], time.perf_counter()
    for step in range(1, training.steps + 1):
        rate = cosine_lr(
            step - 1, training.steps, training.lr, training.min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        measurement = optimize_one_step(
            model, optimizer, sampler, training, index=step,
        )
        learned += measurement["learned_tokens"]
        holdout_loss = None
        if step % training.validate_every == 0 or step == training.steps:
            holdout_loss = eval_windows(
                model, holdout, training, seed=training.seed + 7
            )["cross_entropy"]
            if holdout_loss < best:
                best, best_step = holdout_loss, step
                save_checkpoint(
                    destination / "best", model, model.cfg,
                    step=step, train_tokens=learned,
                    train_loss=measurement["train_loss"],
                    validation_loss=best,
                    seed=training.seed, dataset_fingerprint=dataset_identity,
                    extras={"strategy":config.strategy,"phase":"best"},
                )
        row = OptimizationRecord(
            step=step, strategy=config.strategy,
            stage=measurement["stage"],
            train_loss=measurement["train_loss"],
            validation_loss=holdout_loss,
            gradient_norm=measurement["gradient_norm"],
            changed_weight_l2=measurement["changed_weight_l2"],
            learned_tokens=measurement["learned_tokens"],
            learning_rate=rate,
            selected_categories=measurement["selected_categories"],
            selected_difficulty=measurement["selected_difficulty"],
            wall_time=time.perf_counter() - started,
        )
        records.append(row.as_dict())
        _append(destination / "metrics.jsonl", row.as_dict())
        if config.record_selections and step <= config.maximum_selection_log:
            _append(destination / "selections.jsonl", {
                "step":step, "strategy":config.strategy,
                "identifiers":measurement["selected_identifiers"],
                "stage":measurement["stage"],
            })
    final_loss = eval_windows(
        model, holdout, training, seed=training.seed + 7
    )["cross_entropy"]
    save_checkpoint(
        destination / "final", model, model.cfg,
        step=training.steps, train_tokens=learned,
        train_loss=records[-1]["train_loss"],
        validation_loss=final_loss,
        seed=training.seed, dataset_fingerprint=dataset_identity,
        extras={"strategy":config.strategy,"phase":"final"},
    )
    report = {
        "strategy":config.strategy,
        "training_config":dataclasses.asdict(training),
        "integrity":integrity,
        "dataset_fingerprint":dataset_identity,
        "baseline_validation_loss":baseline,
        "best_validation_loss":best,
        "final_validation_loss":final_loss,
        "improvement_over_baseline":baseline - best,
        "best_step":best_step,
        "optimizer_steps":training.steps,
        "weight_updates_verified":all(
            row["changed_weight_l2"] > 0 for row in records
        ),
        "supervised_tokens":learned,
        "selection_stats":sampler.summary(),
        "teacher_required_for_inference":False,
        "blind_test_used_for_selection":False,
        "claim":"Small experimental answer-only model; not frontier intelligence",
    }
    _serialize(destination / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train S's own weights with a real curriculum or uniform control"
    )
    parser.add_argument("--train",type=Path,required=True)
    parser.add_argument("--validation",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--strategy",choices=("curriculum","uniform"),default="curriculum")
    parser.add_argument("--steps",type=int,default=100)
    parser.add_argument("--context",type=int,default=64)
    parser.add_argument("--dim",type=int,default=128)
    parser.add_argument("--layers",type=int,default=2)
    parser.add_argument("--seed",type=int,default=42)
    args = parser.parse_args(argv)
    training = SupervisedTrainConfig(
        steps=args.steps, context=args.context,
        dim=args.dim, layers=args.layers, seed=args.seed,
    )
    report = run_curriculum_training(
        import_jsonl(args.train),import_jsonl(args.validation),args.output,
        config=CurriculumTrainingConfig(training=training,strategy=args.strategy),
    )
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
