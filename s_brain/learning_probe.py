"""S-Brain learning proof: predictions, loss, gradients and weight updates.

Cycle 4 adds an auditable supervised next-byte learning experiment. Unlike
synthetic anecdotes, this module records every optimization step's pre-update
loss, gradient norm, exact parameter delta, post-update loss and unseen
validation loss. It provides no guarantee of human-level intelligence.

This is an experiment harness: calling optimizer.step() modifies local model
parameters for a measured example. S's permanent training infrastructure
from Cycle 1 remains unchanged.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
from collections.abc import Iterable
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import Config, SBrain


@dataclasses.dataclass(frozen=True)
class ProbeConfig:
    steps: int = 50
    lr: float = 1e-3
    batch_size: int = 4
    sequence: int = 16
    seed: int = 42
    dim: int = 64
    layers: int = 2
    heads: int = 4
    kv_heads: int = 2
    gradient_clip: float = 1.0
    device: str = "cpu"
    record_every: int = 1

    def validate(self):
        if self.steps < 1 or self.batch_size < 1 or self.sequence < 2:
            raise ValueError("Invalid steps, batch size or sequence")
        if self.lr <= 0 or not math.isfinite(self.lr):
            raise ValueError("Learning rate must be positive and finite")
        if self.dim % self.heads or self.heads % self.kv_heads:
            raise ValueError("Incompatible model dimensions")
        if (self.dim // self.heads) % 2:
            raise ValueError("RoPE requires even head dimension")
        if self.gradient_clip <= 0 or self.record_every < 1:
            raise ValueError("Invalid clipping or recording interval")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Unrecognized device")


@dataclasses.dataclass
class ProbeStep:
    step: int
    train_before: float
    train_after: float
    validation: float
    gradient_norm_before_clip: float
    parameter_delta_l2: float
    prediction_correct_fraction: float
    train_tokens: int

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def split_bytes(
    train: bytes,
    validation: bytes,
    sequence: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(train) <= sequence + 1 or len(validation) <= sequence + 1:
        raise ValueError("Both datasets need at least sequence + 2 bytes")
    if train == validation:
        raise ValueError("Training and validation byte sequences are identical")
    if not train.strip() or not validation.strip():
        raise ValueError("Blank training or validation data")
    return (
        torch.tensor(list(train), dtype=torch.long),
        torch.tensor(list(validation), dtype=torch.long),
    )


def dataset_fingerprint(train: bytes, validation: bytes) -> dict:
    return {
        "training_sha256": hashlib.sha256(train).hexdigest(),
        "validation_sha256": hashlib.sha256(validation).hexdigest(),
        "training_bytes": len(train),
        "validation_bytes": len(validation),
        "identical": train == validation,
    }


def make_batch(
    source: torch.Tensor,
    sequence: int,
    batch_size: int,
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source.ndim != 1 or len(source) <= sequence + 1:
        raise ValueError("Source too small for batch")
    start = torch.randint(
        0, len(source) - sequence,
        (batch_size,),
        generator=generator,
    )
    offsets = torch.arange(sequence)
    x = source[start[:, None] + offsets]
    y = source[start[:, None] + offsets + 1]
    return x.to(device), y.to(device)


def capture_parameters(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def parameter_delta(before: dict[str, torch.Tensor], model) -> float:
    squared = 0.0
    now = dict(model.named_parameters())
    if set(before) != {name for name, p in now.items() if p.requires_grad}:
        raise ValueError("Learned parameter names changed")
    for name, original in before.items():
        current = now[name].detach().float().cpu()
        squared += float((current - original).square().sum())
    return math.sqrt(squared)


def prediction_fraction(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.shape[:2] != targets.shape:
        raise ValueError("Prediction/target shapes differ")
    guesses = logits.argmax(dim=-1)
    return float((guesses == targets).float().mean())


@torch.inference_mode()
def validation_loss(
    model,
    source: torch.Tensor,
    *,
    sequence: int,
    batch_size: int,
    generator: torch.Generator,
    device: str,
    repeats: int = 4,
) -> float:
    model.eval()
    samples = []
    for _ in range(repeats):
        x, y = make_batch(source, sequence, batch_size, generator, device)
        _, loss = model(x, y)
        samples.append(float(loss))
    return sum(samples) / len(samples)


def build_model(config: ProbeConfig):
    config.validate()
    torch.manual_seed(config.seed)
    model_config = Config(
        dim=config.dim,
        layers=config.layers,
        heads=config.heads,
        kv_heads=config.kv_heads,
        context=config.sequence,
    )
    return SBrain(model_config).to(config.device)


def step_once(
    model,
    optimizer,
    train_batch: tuple[torch.Tensor, torch.Tensor],
    valid_source: torch.Tensor,
    *,
    config: ProbeConfig,
    valid_generator: torch.Generator,
    index: int,
) -> ProbeStep:
    model.train()
    x, y = train_batch
    optimizer.zero_grad(set_to_none=True)
    before = capture_parameters(model)
    logits, loss = model(x, y)
    start_loss = float(loss.detach())
    correct = prediction_fraction(logits.detach(), y)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.gradient_clip
    )
    if not torch.isfinite(grad_norm):
        raise RuntimeError("Non-finite gradient; refusing update")
    optimizer.step()
    changed = parameter_delta(before, model)
    if not math.isfinite(changed) or changed <= 0:
        raise RuntimeError("No measurable trainable parameter update")
    model.eval()
    with torch.inference_mode():
        _, after = model(x, y)
    measured_validation = validation_loss(
        model, valid_source,
        sequence=config.sequence,
        batch_size=config.batch_size,
        generator=valid_generator,
        device=config.device,
    )
    return ProbeStep(
        step=index,
        train_before=start_loss,
        train_after=float(after),
        validation=measured_validation,
        gradient_norm_before_clip=float(grad_norm),
        parameter_delta_l2=changed,
        prediction_correct_fraction=correct,
        train_tokens=y.numel(),
    )


def run_probe(
    train: bytes,
    validation: bytes,
    config: ProbeConfig | None = None,
) -> dict:
    config = config or ProbeConfig()
    config.validate()
    data = dataset_fingerprint(train, validation)
    training, validation_tokens = split_bytes(
        train, validation, config.sequence
    )
    model = build_model(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=0.01
    )
    training_rng = torch.Generator().manual_seed(config.seed + 1)
    validation_rng = torch.Generator().manual_seed(config.seed + 2)
    baseline = validation_loss(
        model, validation_tokens,
        sequence=config.sequence,
        batch_size=config.batch_size,
        generator=torch.Generator().manual_seed(config.seed + 3),
        device=config.device,
    )
    records = []
    for step in range(1, config.steps + 1):
        x, y = make_batch(
            training, config.sequence, config.batch_size,
            training_rng, config.device
        )
        item = step_once(
            model, optimizer, (x, y), validation_tokens,
            config=config, valid_generator=validation_rng,
            index=step
        )
        if step == 1 or step == config.steps or step % config.record_every == 0:
            records.append(item.as_dict())
    ending = validation_loss(
        model, validation_tokens,
        sequence=config.sequence,
        batch_size=config.batch_size,
        generator=torch.Generator().manual_seed(config.seed + 3),
        device=config.device,
    )
    verified_updates = all(
        item["parameter_delta_l2"] > 0 for item in records
    )
    return {
        "config": dataclasses.asdict(config),
        "dataset": data,
        "baseline_validation_loss": baseline,
        "final_validation_loss": ending,
        "validation_loss_delta": ending - baseline,
        "recorded_updates": len(records),
        "all_recorded_updates_changed_weights": verified_updates,
        "measurements": records,
        "assessment": {
            "learned_on_validation": ending < baseline,
            "independent_training": True,
            "general_intelligence_proven": False,
            "description": (
                "Loss change under one held-out byte sample; "
                "not a validated general-purpose assistant benchmark"
            ),
        },
    }


def summarize_probe(report: dict) -> dict:
    records = report["measurements"]
    return {
        "recorded_steps": len(records),
        "mean_gradient_norm": sum(
            item["gradient_norm_before_clip"] for item in records
        ) / len(records),
        "mean_weight_change_l2": sum(
            item["parameter_delta_l2"] for item in records
        ) / len(records),
        "validation_loss_delta": report["validation_loss_delta"],
        "independently_learned": report["all_recorded_updates_changed_weights"],
        "validation_improved": report["assessment"]["learned_on_validation"],
    }


def write_report(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Measure actual predictions, errors and weight updates"
    )
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--sequence", type=int, default=16)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    config = ProbeConfig(
        steps=args.steps, sequence=args.sequence,
        dim=args.dim, layers=args.layers, device=args.device,
    )
    report = run_probe(
        args.train.read_bytes(), args.validation.read_bytes(), config
    )
    if args.output is not None:
        write_report(args.output, report)
    print(json.dumps(summarize_probe(report), indent=2))


if __name__ == "__main__":
    main()
