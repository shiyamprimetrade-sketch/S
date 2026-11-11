"""Repeatable S-Brain learning-curve experiments across independent seeds.

Reuses the checked Cycle 5 supervised training engine. Each run trains S's
own weights from scratch, stores validation checkpoints, and records final
metrics, fingerprints and config. The comparison does not contact Ollama
or silently combine test and training data.

Note: a learning curve from a tiny synthetic fixture proves that code runs,
not that a general-purpose intelligence was trained.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Callable, Iterable

from .train_supervised import (
    SupervisedTrainConfig, run_supervised_training, check_split_integrity,
)
from .supervised_data import SupervisedRecord, import_jsonl


@dataclasses.dataclass(frozen=True)
class CurvePlan:
    seeds: tuple[int, ...] = (11, 29, 47)
    steps: tuple[int, ...] = (10, 30, 60)
    maximum_runs: int = 30
    require_disjoint_holdout: bool = True
    fail_if_no_improvement: bool = False
    label: str = "s-brain-curve"

    def validate(self):
        if not self.seeds or not self.steps:
            raise ValueError("Curve needs seeds and training budgets")
        if any(step < 1 for step in self.steps):
            raise ValueError("Training budgets must be positive")
        if len(set(self.steps)) != len(self.steps):
            raise ValueError("Duplicate training step budget")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Duplicate random seed")
        if not 1 <= self.maximum_runs <= 10000:
            raise ValueError("Invalid maximum experiment count")
        if len(self.seeds) * len(self.steps) > self.maximum_runs:
            raise ValueError("Curve plan exceeds maximum run count")
        if not self.label or "/" in self.label or ".." in self.label:
            raise ValueError("Invalid experiment label")


@dataclasses.dataclass(frozen=True)
class CurvePoint:
    seed: int
    budget_steps: int
    baseline_validation_loss: float
    best_validation_loss: float
    final_validation_loss: float
    best_step: int
    optimizer_steps: int
    weights_changed_each_step: bool
    train_fingerprint: str
    validation_fingerprint: str
    directory: str

    @property
    def improvement(self):
        return self.baseline_validation_loss - self.best_validation_loss

    def as_dict(self):
        return {**dataclasses.asdict(self), "improvement": self.improvement}


def config_for_run(
    base: SupervisedTrainConfig, seed: int, steps: int,
) -> SupervisedTrainConfig:
    result = dataclasses.replace(base, seed=seed, steps=steps)
    result.validate()
    return result


def run_directory(root: Path, seed: int, steps: int) -> Path:
    if seed < 0 or steps < 1:
        raise ValueError("Invalid run coordinates")
    return root / f"seed-{seed:06d}" / f"steps-{steps:06d}"


def point_from_training(
    report: dict, *, seed: int, budget: int, directory: Path,
) -> CurvePoint:
    if report["optimizer_steps"] != budget:
        raise ValueError("Training engine performed wrong step budget")
    split = report["integrity"]
    return CurvePoint(
        seed=seed, budget_steps=budget,
        baseline_validation_loss=report["baseline_validation_loss"],
        best_validation_loss=report["best_validation_loss"],
        final_validation_loss=report["final_validation_loss"],
        best_step=report["best_step"],
        optimizer_steps=report["optimizer_steps"],
        weights_changed_each_step=report["weights_changed_each_step"],
        train_fingerprint=split["train_fingerprint"],
        validation_fingerprint=split["holdout_fingerprint"],
        directory=str(directory),
    )


def run_learning_curve(
    train: list[SupervisedRecord],
    holdout: list[SupervisedRecord],
    root: Path,
    *,
    plan: CurvePlan | None = None,
    base_config: SupervisedTrainConfig | None = None,
    trainer: Callable = run_supervised_training,
) -> dict:
    """Train every seed/budget independently to avoid hidden checkpoint reuse."""
    plan = plan or CurvePlan()
    plan.validate()
    base_config = base_config or SupervisedTrainConfig()
    base_config.validate()
    if plan.require_disjoint_holdout:
        check_split_integrity(train, holdout)
    points = []
    for seed in plan.seeds:
        for steps in sorted(plan.steps):
            directory = run_directory(root, seed, steps)
            config = config_for_run(base_config, seed, steps)
            # No checkpoint-resume; each curve point begins from a fresh model.
            report = trainer(train, holdout, directory, config=config)
            item = point_from_training(report, seed=seed, budget=steps, directory=directory)
            if not item.weights_changed_each_step:
                raise RuntimeError("Experiment failed to update every training step")
            if plan.fail_if_no_improvement and item.improvement <= 0:
                raise RuntimeError("Held-out performance failed to improve")
            points.append(item)
    result = {
        "label": plan.label,
        "plan": dataclasses.asdict(plan),
        "base_config": dataclasses.asdict(base_config),
        "points": [item.as_dict() for item in points],
        "aggregate": aggregate_curve(points),
        "note": "Learning curves are task-specific; no general-intelligence claim",
    }
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "learning_curve.json", result)
    return result


def group_by_budget(points: Iterable[CurvePoint]) -> dict[int, list[CurvePoint]]:
    groups = collections.defaultdict(list)
    for item in points:
        groups[item.budget_steps].append(item)
    return dict(sorted(groups.items()))


def bootstrap_mean(
    samples: list[float], *, seed: int = 42,
    draws: int = 1000, confidence: float = 0.95,
) -> dict:
    import random
    if not samples or draws < 50 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    rng = random.Random(seed)
    estimates = sorted(
        statistics.mean(rng.choices(samples, k=len(samples)))
        for _ in range(draws)
    )
    alpha = 1 - confidence
    low = int(alpha / 2 * (draws - 1))
    high = int((1 - alpha / 2) * (draws - 1))
    return {
        "mean": statistics.mean(samples),
        "lower": estimates[low],
        "upper": estimates[high],
        "draws": draws,
        "sampling_unit": "independent training seed",
    }


def aggregate_curve(points: Iterable[CurvePoint]) -> list[dict]:
    result = []
    for budget, rows in group_by_budget(points).items():
        baseline = [item.baseline_validation_loss for item in rows]
        best = [item.best_validation_loss for item in rows]
        improvements = [item.improvement for item in rows]
        result.append({
            "steps": budget,
            "repeats": len(rows),
            "mean_baseline": statistics.mean(baseline),
            "mean_best": statistics.mean(best),
            "mean_improvement": statistics.mean(improvements),
            "improved_repeats": sum(item.improvement > 0 for item in rows),
            "improvement_interval": bootstrap_mean(improvements, draws=200),
            "seed_to_best_loss": {
                str(item.seed): item.best_validation_loss for item in rows
            },
        })
    return result


def compare_budgets(points: Iterable[CurvePoint]) -> dict:
    grouped = group_by_budget(points)
    if len(grouped) < 2:
        raise ValueError("Need at least two step budgets")
    ordered = sorted(grouped)
    first, last = ordered[0], ordered[-1]
    early = {item.seed: item for item in grouped[first]}
    late = {item.seed: item for item in grouped[last]}
    if set(early) != set(late):
        raise ValueError("Training seeds differ between budgets")
    differences = {
        str(seed): early[seed].best_validation_loss - late[seed].best_validation_loss
        for seed in sorted(early)
    }
    return {
        "earliest_steps": first,
        "latest_steps": last,
        "paired_loss_reduction_by_seed": differences,
        "mean_reduction": statistics.mean(differences.values()),
        "more_steps_better_on_all_seeds": all(value > 0 for value in differences.values()),
    }


def read_metrics(path: Path) -> list[dict]:
    entries = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected object")
                for key in ("step", "train_loss", "best_validation_loss"):
                    if key not in row:
                        raise ValueError("Missing metric key")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            entries.append(row)
    if entries and any(
        a["step"] >= b["step"] for a, b in zip(entries, entries[1:])
    ):
        raise ValueError("Metric steps are not strictly increasing")
    return entries


def learning_diagnostics(metrics: list[dict]) -> dict:
    if not metrics:
        raise ValueError("No learning metrics")
    losses = [float(item["train_loss"]) for item in metrics]
    if not all(math.isfinite(x) for x in losses):
        raise ValueError("Training log has nonfinite losses")
    changes = [float(item.get("parameter_update_l2", 0)) for item in metrics]
    valid = [float(item["validation_loss"]) for item in metrics if item.get("validation_loss") is not None]
    return {
        "steps": len(metrics),
        "initial_train_loss": losses[0],
        "last_train_loss": losses[-1],
        "train_loss_delta": losses[-1] - losses[0],
        "best_validation_loss": min(valid) if valid else None,
        "all_updates_nonzero": all(x > 0 for x in changes),
        "validation_measurements": len(valid),
        "note": "Training loss is not a substitute for blind test results",
    }


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Repeated seed/budget learning curves")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29])
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 30])
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args(argv)
    result = run_learning_curve(
        import_jsonl(args.train), import_jsonl(args.holdout), args.output,
        plan=CurvePlan(seeds=tuple(args.seeds), steps=tuple(args.budgets)),
        base_config=SupervisedTrainConfig(
            context=args.context, dim=args.dim, layers=args.layers,
        ),
    )
    print(json.dumps(result["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
