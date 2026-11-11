"""Deterministic curriculum planning for S-Brain answer-completion windows.

Curriculum phases assign increasing relative weights to task categories and
difficulty buckets. A curriculum does NOT change the objective function or
prove better reasoning. All schedules, sampling decisions and seeds can be
reproduced and audited. Validation and test material are never sampled.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable, Sequence

from .supervised_data import SupervisedRecord, TrainingWindow, build_windows, import_jsonl


@dataclasses.dataclass(frozen=True)
class DifficultySignal:
    identifier: str
    length_score: float
    lexical_score: float
    category_score: float
    combined: float
    bucket: str


@dataclasses.dataclass(frozen=True)
class CurriculumStage:
    name: str
    start_fraction: float
    end_fraction: float
    easy_weight: float = 1.0
    medium_weight: float = 1.0
    hard_weight: float = 1.0
    category_weights: dict[str, float] = dataclasses.field(default_factory=dict)

    def validate(self):
        if not self.name:
            raise ValueError("Stage name required")
        if not 0 <= self.start_fraction < self.end_fraction <= 1:
            raise ValueError("Stage fractions out of order")
        weights = [self.easy_weight, self.medium_weight, self.hard_weight]
        weights.extend(self.category_weights.values())
        if any(not math.isfinite(x) or x < 0 for x in weights):
            raise ValueError("Weights must be finite and nonnegative")
        if not any(x > 0 for x in weights[:3]):
            raise ValueError("At least one difficulty bucket must be sampled")

    def weight(self, bucket: str, category: str) -> float:
        if bucket not in ("easy", "medium", "hard"):
            raise ValueError("Unknown difficulty bucket")
        difficulty = getattr(self, bucket + "_weight")
        return difficulty * self.category_weights.get(category, 1.0)


@dataclasses.dataclass(frozen=True)
class CurriculumPolicy:
    seed: int = 42
    easy_quantile: float = 0.33
    hard_quantile: float = 0.67
    token_budget_per_stage: int = 4096
    minimum_windows: int = 4
    preserve_category_coverage: bool = True

    def validate(self):
        if not 0 < self.easy_quantile < self.hard_quantile < 1:
            raise ValueError("Invalid difficulty quantiles")
        if self.token_budget_per_stage < 1 or self.minimum_windows < 1:
            raise ValueError("Invalid minimum curriculum size")


@dataclasses.dataclass(frozen=True)
class CurriculumAssignment:
    identifier: str
    category: str
    bucket: str
    score: float
    windows: int
    training_tokens: int
    selected_weight: float

    def as_dict(self):
        return dataclasses.asdict(self)


def lexical_complexity(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    unique = len({word.casefold() for word in words})
    return unique / len(words)


def score_difficulty(record: SupervisedRecord) -> tuple[float, float, float, float]:
    record.validate()
    completion = bytes(record.completion_bytes).decode("utf-8", errors="replace")
    total = record.total_tokens
    length = min(1.0, math.log1p(total) / math.log1p(8192))
    lexical = lexical_complexity(completion)
    category = {
        "arithmetic": 0.3,
        "math": 0.55,
        "coding": 0.75,
        "reasoning": 0.85,
        "research": 0.85,
    }.get(record.category.casefold(), 0.5)
    combined = 0.4 * length + 0.35 * lexical + 0.25 * category
    return length, lexical, category, combined


def assign_difficulties(
    records: Sequence[SupervisedRecord],
    policy: CurriculumPolicy | None = None,
) -> dict[str, DifficultySignal]:
    policy = policy or CurriculumPolicy()
    policy.validate()
    rows = list(records)
    if not rows:
        raise ValueError("Empty curriculum dataset")
    if any(row.split != "train" for row in rows):
        raise ValueError("Curriculum must not use holdout records")
    scores = [(row, score_difficulty(row)) for row in rows]
    ordered = sorted(score[-1] for _, score in scores)
    def quantile(q):
        index = min(len(ordered)-1, int(q * (len(ordered)-1)))
        return ordered[index]
    lower, upper = quantile(policy.easy_quantile), quantile(policy.hard_quantile)
    result = {}
    for row, factors in scores:
        bucket = "easy" if factors[-1] <= lower else (
            "hard" if factors[-1] > upper else "medium"
        )
        if row.identifier in result:
            raise ValueError("Duplicate curriculum record IDs")
        result[row.identifier] = DifficultySignal(
            row.identifier, factors[0], factors[1], factors[2], factors[3], bucket
        )
    return result


def default_stages() -> list[CurriculumStage]:
    return [
        CurriculumStage(
            "foundation", 0.0, 0.35,
            easy_weight=3.0, medium_weight=1.0, hard_weight=0.25,
        ),
        CurriculumStage(
            "expansion", 0.35, 0.7,
            easy_weight=1.0, medium_weight=2.0, hard_weight=1.0,
        ),
        CurriculumStage(
            "balanced", 0.7, 1.0,
            easy_weight=1.0, medium_weight=1.0, hard_weight=1.0,
        ),
    ]


def validate_stages(stages: Sequence[CurriculumStage]) -> None:
    if not stages:
        raise ValueError("Curriculum has no stages")
    for item in stages:
        item.validate()
    ordered = sorted(stages, key=lambda item: item.start_fraction)
    if abs(ordered[0].start_fraction) > 1e-9:
        raise ValueError("Curriculum must start at 0")
    if abs(ordered[-1].end_fraction - 1) > 1e-9:
        raise ValueError("Curriculum must end at 1")
    for previous, next_item in zip(ordered, ordered[1:]):
        if abs(previous.end_fraction - next_item.start_fraction) > 1e-9:
            raise ValueError("Curriculum stages have gaps or overlap")


def stage_for_step(
    step: int, total_steps: int,
    stages: Sequence[CurriculumStage],
) -> CurriculumStage:
    validate_stages(stages)
    if not 0 <= step < total_steps or total_steps < 1:
        raise ValueError("Step outside schedule")
    fraction = step / total_steps
    for stage in sorted(stages, key=lambda item: item.start_fraction):
        if stage.start_fraction <= fraction < stage.end_fraction:
            return stage
    raise RuntimeError("No stage applies to step")


def weighted_choice(
    values: Sequence[str],
    weights: Sequence[float],
    randomizer: random.Random,
) -> str:
    if len(values) != len(weights) or not values:
        raise ValueError("Value/weight alignment required")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Invalid weight")
    total = sum(weights)
    if total <= 0:
        raise ValueError("No selectable training examples")
    threshold = randomizer.random() * total
    running = 0.0
    for item, weight in zip(values, weights):
        running += weight
        if threshold < running:
            return item
    return values[-1]


class CurriculumSampler:
    """Window-index samples with deterministic per-step category weighting."""

    def __init__(
        self,
        records: Sequence[SupervisedRecord],
        *,
        context: int,
        stages: Sequence[CurriculumStage] | None = None,
        policy: CurriculumPolicy | None = None,
    ):
        self.policy = policy or CurriculumPolicy()
        self.policy.validate()
        self.stages = list(stages or default_stages())
        validate_stages(self.stages)
        self.records = list(records)
        self.signals = assign_difficulties(self.records, self.policy)
        self.windows_by_record = {}
        for record in self.records:
            values = build_windows([record], context, method="dense")
            if values:
                self.windows_by_record[record.identifier] = values
        if len(self.windows_by_record) < self.policy.minimum_windows:
            raise ValueError("Insufficient usable training records")
        self.randomizer = random.Random(self.policy.seed)
        self.category_counts = collections.Counter(row.category for row in self.records)

    def pick_record(self, stage: CurriculumStage) -> str:
        keys = sorted(self.windows_by_record)
        weight_list = []
        for key in keys:
            record = next(row for row in self.records if row.identifier == key)
            signal = self.signals[key]
            weight = stage.weight(signal.bucket, record.category)
            if self.policy.preserve_category_coverage:
                weight /= max(1, self.category_counts[record.category])
            weight_list.append(weight)
        return weighted_choice(keys, weight_list, self.randomizer)

    def draw(
        self, step: int, total_steps: int, batch_size: int = 1,
    ) -> tuple[list[TrainingWindow], list[str]]:
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        stage = stage_for_step(step, total_steps, self.stages)
        selections = []
        chosen_ids = []
        for _ in range(batch_size):
            identifier = self.pick_record(stage)
            windows = self.windows_by_record[identifier]
            selections.append(self.randomizer.choice(windows))
            chosen_ids.append(identifier)
        return selections, chosen_ids

    def summary(self):
        return {
            "records": len(self.records),
            "usable_records": len(self.windows_by_record),
            "windows": sum(len(x) for x in self.windows_by_record.values()),
            "difficulty_distribution": dict(collections.Counter(
                signal.bucket for signal in self.signals.values()
            )),
            "stages": [dataclasses.asdict(item) for item in self.stages],
        }


def dry_run_curriculum(
    sampler: CurriculumSampler,
    steps: int,
    batch_size: int = 4,
) -> dict:
    if steps < 1:
        raise ValueError("steps must be positive")
    categories = collections.Counter()
    stages = collections.Counter()
    for index in range(steps):
        stage = stage_for_step(index, steps, sampler.stages)
        selections, ids = sampler.draw(index, steps, batch_size)
        stages[stage.name] += len(selections)
        categories.update(ids)
    return {
        "steps": steps,
        "batch_size": batch_size,
        "stage_samples": dict(stages),
        "sampled_record_ids": dict(categories),
        "note": "Sampling plan only; no model training executed",
    }


def schedule_fingerprint(
    stages: Sequence[CurriculumStage],
    policy: CurriculumPolicy,
) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({
        "stages": [dataclasses.asdict(item) for item in stages],
        "policy": dataclasses.asdict(policy),
    }, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit the S-Brain curriculum schedule")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args(argv)
    sampler = CurriculumSampler(
        import_jsonl(args.train), context=args.context,
        policy=CurriculumPolicy(minimum_windows=1),
    )
    print(json.dumps({
        "summary": sampler.summary(),
        "dry_run": dry_run_curriculum(sampler, args.steps),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
