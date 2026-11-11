"""S-Brain checkpoint promotion with hard quality and provenance gates.

Choosing the lowest validation loss from a list of checkpoint folders is not
sufficient evidence of generalization. This module requires matching dataset
fingerprints, verified safetensors hashes, a held-out report produced without
Ollama, plus configurable loss/accuracy/latency gates.

Promotion creates a *manifest reference*, not a destructive overwrite of
weights or a live application switch. Operator approval remains explicit.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable

from .checkpoint import load_manifest, sha256_file


@dataclasses.dataclass(frozen=True)
class SelectionPolicy:
    min_loss_reduction: float = 0.01
    max_accuracy_regression: float = 0.02
    max_latency_increase_fraction: float = 0.50
    minimum_holdout_records: int = 10
    minimum_holdout_tokens: int = 100
    require_same_dataset: bool = True
    require_same_architecture: bool = True
    require_changed_weights: bool = True
    require_independent_evaluation: bool = True
    require_operator_approval: bool = True

    def validate(self):
        if any(x < 0 or not math.isfinite(x) for x in (
            self.min_loss_reduction, self.max_accuracy_regression,
            self.max_latency_increase_fraction,
        )):
            raise ValueError("Invalid checkpoint gate threshold")
        if self.minimum_holdout_records < 1 or self.minimum_holdout_tokens < 1:
            raise ValueError("Holdout quality minimum must be positive")


@dataclasses.dataclass(frozen=True)
class CheckpointCandidate:
    path: Path
    weight_sha256: str
    dataset_fingerprint: str
    architecture_fingerprint: str
    training_step: int
    validation_loss: float | None

    @classmethod
    def from_path(cls, path: Path):
        manifest = load_manifest(path, verify_digest=True)
        architecture = hashlib.sha256(
            json.dumps(manifest.model_config, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return cls(
            path=path,
            weight_sha256=manifest.weights_sha256,
            dataset_fingerprint=manifest.dataset_fingerprint,
            architecture_fingerprint=architecture,
            training_step=manifest.step,
            validation_loss=manifest.validation_loss,
        )

    def public_metadata(self):
        return {
            "path": str(self.path),
            "weight_sha256": self.weight_sha256,
            "dataset_fingerprint": self.dataset_fingerprint,
            "architecture_fingerprint": self.architecture_fingerprint,
            "training_step": self.training_step,
            "validation_loss": self.validation_loss,
        }


@dataclasses.dataclass(frozen=True)
class SelectionEvidence:
    checkpoint_path: str
    evaluation_sha256: str
    holdout_fingerprint: str
    accuracy: float
    cross_entropy: float
    records: int
    tokens: int
    p95_latency_ms: float | None = None
    teacher_used_at_evaluation: bool = False

    def validate(self):
        if not self.checkpoint_path or not self.holdout_fingerprint:
            raise ValueError("Checkpoint and holdout reference required")
        if len(self.evaluation_sha256) != 64:
            raise ValueError("Evaluation content hash required")
        if not 0 <= self.accuracy <= 1 or not math.isfinite(self.accuracy):
            raise ValueError("Invalid held-out accuracy")
        if self.cross_entropy < 0 or not math.isfinite(self.cross_entropy):
            raise ValueError("Invalid cross-entropy")
        if self.records < 1 or self.tokens < 1:
            raise ValueError("No evaluation examples")
        if self.p95_latency_ms is not None and (
            self.p95_latency_ms < 0 or not math.isfinite(self.p95_latency_ms)
        ):
            raise ValueError("Invalid latency measurement")


@dataclasses.dataclass
class SelectionDecision:
    baseline: CheckpointCandidate
    candidate: CheckpointCandidate
    baseline_evidence: SelectionEvidence
    candidate_evidence: SelectionEvidence
    checks: dict[str, bool]
    deltas: dict[str, float | None]
    approval_required: bool

    @property
    def accepted(self):
        return all(self.checks.values())

    def report(self):
        return {
            "accepted_by_quality_gates": self.accepted,
            "promotion_requires_operator_approval": self.approval_required,
            "checks": self.checks,
            "deltas": self.deltas,
            "baseline": self.baseline.public_metadata(),
            "candidate": self.candidate.public_metadata(),
            "evaluation": {
                "baseline_hash": self.baseline_evidence.evaluation_sha256,
                "candidate_hash": self.candidate_evidence.evaluation_sha256,
                "holdout_fingerprint": self.candidate_evidence.holdout_fingerprint,
            },
            "note": "Passing these metrics is not evidence of frontier-level AI",
        }


def evidence_from_report(
    checkpoint: Path,
    path: Path,
) -> SelectionEvidence:
    raw = path.read_bytes()
    report = json.loads(raw.decode("utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Evaluation report must be object")
    result = SelectionEvidence(
        checkpoint_path=str(checkpoint),
        evaluation_sha256=hashlib.sha256(raw).hexdigest(),
        holdout_fingerprint=str(report["fingerprint"]),
        accuracy=float(report["accuracy"]),
        cross_entropy=float(report["cross_entropy"]),
        records=int(report["records"]),
        tokens=int(report["tokens"]),
        p95_latency_ms=report.get("p95_latency_ms"),
        teacher_used_at_evaluation=bool(report.get("teacher_used_at_evaluation", False)),
    )
    result.validate()
    return result


def select_candidate(
    baseline: CheckpointCandidate,
    candidate: CheckpointCandidate,
    before: SelectionEvidence,
    after: SelectionEvidence,
    *,
    policy: SelectionPolicy | None = None,
) -> SelectionDecision:
    policy = policy or SelectionPolicy()
    policy.validate()
    before.validate()
    after.validate()
    if before.checkpoint_path != str(baseline.path):
        raise ValueError("Baseline evidence references different checkpoint")
    if after.checkpoint_path != str(candidate.path):
        raise ValueError("Candidate evidence references different checkpoint")
    same_holdout = before.holdout_fingerprint == after.holdout_fingerprint
    reduction = before.cross_entropy - after.cross_entropy
    accuracy_delta = after.accuracy - before.accuracy
    latency_delta = None
    if before.p95_latency_ms is not None and after.p95_latency_ms is not None:
        latency_delta = after.p95_latency_ms - before.p95_latency_ms
    checks = {
        "valid_weight_hashes": all(len(x.weight_sha256) == 64 for x in (baseline, candidate)),
        "same_holdout": same_holdout,
        "same_dataset": (
            baseline.dataset_fingerprint == candidate.dataset_fingerprint
            or not policy.require_same_dataset
        ),
        "same_architecture": (
            baseline.architecture_fingerprint == candidate.architecture_fingerprint
            or not policy.require_same_architecture
        ),
        "weights_changed": (
            baseline.weight_sha256 != candidate.weight_sha256
            or not policy.require_changed_weights
        ),
        "loss_improved": reduction >= policy.min_loss_reduction,
        "accuracy_not_regressed": accuracy_delta >= -policy.max_accuracy_regression,
        "holdout_records": min(before.records, after.records) >= policy.minimum_holdout_records,
        "holdout_tokens": min(before.tokens, after.tokens) >= policy.minimum_holdout_tokens,
        "independent_evaluation": (
            not (before.teacher_used_at_evaluation or after.teacher_used_at_evaluation)
            or not policy.require_independent_evaluation
        ),
        "checkpoint_progress": candidate.training_step > baseline.training_step,
        "latency": (
            latency_delta is None
            or before.p95_latency_ms is None
            or after.p95_latency_ms <= (
                before.p95_latency_ms * (1 + policy.max_latency_increase_fraction)
            )
        ),
    }
    return SelectionDecision(
        baseline, candidate, before, after, checks,
        {
            "loss_reduction": reduction,
            "accuracy_gain": accuracy_delta,
            "p95_latency_delta_ms": latency_delta,
        },
        policy.require_operator_approval,
    )


def rank_checkpoints(
    candidates: Iterable[CheckpointCandidate],
    evidence: dict[str, SelectionEvidence],
) -> list[dict]:
    scored = []
    for item in candidates:
        record = evidence.get(str(item.path))
        if record is None:
            continue
        record.validate()
        scored.append({
            "checkpoint": item.public_metadata(),
            "cross_entropy": record.cross_entropy,
            "accuracy": record.accuracy,
            "records": record.records,
            "tokens": record.tokens,
        })
    return sorted(
        scored,
        key=lambda item: (
            item["cross_entropy"], -item["accuracy"],
            -item["checkpoint"]["training_step"],
        ),
    )


def atomic_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=".s-brain-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def write_selection(path: Path, decision: SelectionDecision) -> None:
    atomic_write(path, json.dumps(decision.report(), indent=2, allow_nan=False))


def prepare_promotion(
    decision: SelectionDecision,
    output: Path,
    *,
    operator_approved: bool = False,
) -> dict:
    """Write an inert checkpoint pointer only after every required gate."""
    if not decision.accepted:
        raise ValueError("Checkpoint did not pass quality gates")
    if decision.approval_required and not operator_approved:
        raise PermissionError("Explicit operator approval required")
    payload = {
        "status": "selected_not_deployed",
        "weights_path": str(decision.candidate.path),
        "weights_sha256": decision.candidate.weight_sha256,
        "dataset_fingerprint": decision.candidate.dataset_fingerprint,
        "evaluation_sha256": decision.candidate_evidence.evaluation_sha256,
        "operator_approved": operator_approved,
        "note": "No server hot reload or deployment is performed",
    }
    atomic_write(output, json.dumps(payload, indent=2))
    return payload


def evaluate_promotion_status(pointer: Path) -> dict:
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    if payload.get("status") != "selected_not_deployed":
        raise ValueError("Unknown pointer status")
    candidate = CheckpointCandidate.from_path(Path(payload["weights_path"]))
    return {
        "valid": candidate.weight_sha256 == payload["weights_sha256"],
        "selected": payload["weights_path"],
        "deployed": False,
        "operator_approved": bool(payload.get("operator_approved")),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Review independent S-Brain checkpoint metrics")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline = CheckpointCandidate.from_path(args.baseline)
    candidate = CheckpointCandidate.from_path(args.candidate)
    decision = select_candidate(
        baseline, candidate,
        evidence_from_report(args.baseline, args.before),
        evidence_from_report(args.candidate, args.after),
    )
    write_selection(args.report, decision)
    print(json.dumps(decision.report(), indent=2))
    return 0 if decision.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
