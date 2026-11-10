"""Immutable model-comparison evidence packet and reproducibility audit.

A selection claim must be traceable to matching training configuration,
dataset versions, checkpoint weights, evaluation fingerprints and paired
seed runs. This module collects an audit packet without moving model weights,
running external network calls or enabling S's live chat backend.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path

from .checkpoint import load_manifest
from .curriculum_experiments import validate_experiment_report
from .dataset_versions import DatasetRegistry


@dataclasses.dataclass(frozen=True)
class EvidencePolicy:
    minimum_seeds: int = 2
    require_same_dataset: bool = True
    require_same_initialization: bool = True
    require_all_weights_updated: bool = True
    require_test_isolated: bool = True
    forbid_real_world_claims_without_blind_test: bool = True

    def validate(self):
        if self.minimum_seeds < 1:
            raise ValueError("minimum_seeds must be positive")


@dataclasses.dataclass
class EvidencePacket:
    experiment_sha256: str
    dataset_release_sha256: str
    checkpoint_digests: dict[str, str]
    checks: dict[str, bool]
    outcomes: dict
    source_paths: dict[str, str]
    notes: list[str]

    @property
    def accepted(self) -> bool:
        return all(self.checks.values())

    def as_dict(self):
        return {
            **dataclasses.asdict(self),
            "evidence_accepted": self.accepted,
            "production_ready": False,
            "gemini_equivalence": False,
        }


def digest_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda:file.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def validate_pair(row: dict, policy: EvidencePolicy) -> dict[str, bool]:
    uniform=row["uniform"]
    curriculum=row["curriculum"]
    return {
        "seed_match": uniform["seed"]==curriculum["seed"]==row["seed"],
        "matching_dataset": (
            uniform["fingerprint"]==curriculum["fingerprint"]
            or not policy.require_same_dataset
        ),
        "matching_initialization": (
            abs(uniform["baseline_loss"]-curriculum["baseline_loss"])<1e-5
            or not policy.require_same_initialization
        ),
        "weight_updates": (
            uniform["gradient_updates_verified"]
            and curriculum["gradient_updates_verified"]
            or not policy.require_all_weights_updated
        ),
        "finite_losses": all(
            math.isfinite(value)
            for arm in (uniform,curriculum)
            for value in (arm["baseline_loss"],arm["best_loss"],arm["final_loss"])
        ),
    }


def checkpoint_digests_for_pair(row: dict) -> dict[str,str]:
    values={}
    for arm in ("uniform","curriculum"):
        path=Path(row[arm]["checkpoint_path"])
        manifest=load_manifest(path,verify_digest=True)
        if manifest.seed!=row["seed"]:
            raise ValueError("Checkpoint seed differs from experiment")
        if manifest.dataset_fingerprint!=row[arm]["fingerprint"]:
            raise ValueError("Checkpoint dataset differs from experiment")
        values[f'{row["seed"]}:{arm}']=manifest.weights_sha256
    return values


def generate_evidence_packet(
    experiment_file: Path,
    dataset_registry: DatasetRegistry,
    *,
    dataset_name: str,
    dataset_version: str,
    policy: EvidencePolicy | None = None,
    blind_report: Path | None = None,
) -> EvidencePacket:
    policy=policy or EvidencePolicy()
    policy.validate()
    report=json.loads(experiment_file.read_text("utf-8"))
    validate_experiment_report(report)
    release=dataset_registry.load(
        dataset_name,dataset_version,verify_files=True
    )
    paired=report["pairs"]
    checks={
        "minimum_seed_repeats":len(paired)>=policy.minimum_seeds,
        "unique_seed_ids":len({item["seed"] for item in paired})==len(paired),
        "validation_only_selection":report["selection_scope"]=="validation_only",
        "test_isolation":(
            not report["test_used_for_selection"]
            or not policy.require_test_isolated
        ),
        "dataset_train":(
            release.train.fingerprint==report["integrity"]["train_fingerprint"]
        ),
        "dataset_validation":(
            release.validation.fingerprint==report["integrity"]["holdout_fingerprint"]
        ),
    }
    digests={}
    for item in paired:
        fields=validate_pair(item,policy)
        for name,passed in fields.items():
            checks[f"seed-{item['seed']}:{name}"]=passed
        digests.update(checkpoint_digests_for_pair(item))
    notes=["All training claims require independent replication."]
    if blind_report is not None:
        blind=json.loads(blind_report.read_text("utf-8"))
        checks["blind_fingerprint"]=(
            blind["benchmark"]["test_fingerprint"]==release.test.fingerprint
        )
        checks["blind_test_not_used_for_selection"]=not blind.get(
            "used_to_select_checkpoint",True
        )
        notes.append("Blind test is descriptive only; do not tune on these results.")
    elif policy.forbid_real_world_claims_without_blind_test:
        notes.append("No blind test report: do not claim generalization to real-world tasks.")
    packet=EvidencePacket(
        experiment_sha256=digest_file(experiment_file),
        dataset_release_sha256=release.release_fingerprint,
        checkpoint_digests=digests,checks=checks,
        outcomes={
            "paired_seed_count":len(paired),
            "mean_curriculum_advantage":report["summary"]["mean_curriculum_advantage"],
            "blind_test_supplied":blind_report is not None,
        },
        source_paths={
            "experiment":str(experiment_file),
            "dataset":str(dataset_registry.path_for(dataset_name,dataset_version)),
            "blind_report":str(blind_report) if blind_report else "",
        },
        notes=notes,
    )
    return packet


def compare_packets(left: EvidencePacket,right: EvidencePacket):
    return {
        "same_experiment":left.experiment_sha256==right.experiment_sha256,
        "same_dataset":left.dataset_release_sha256==right.dataset_release_sha256,
        "same_checkpoints":left.checkpoint_digests==right.checkpoint_digests,
        "left_accepted":left.accepted,
        "right_accepted":right.accepted,
    }


def write_packet(path: Path, packet: EvidencePacket):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError("Evidence packet is immutable; choose a new path")
    path.write_text(
        json.dumps(packet.as_dict(),indent=2,allow_nan=False),
        encoding="utf-8",
    )


def main(argv=None):
    p=argparse.ArgumentParser(description="Verify S-Brain learning experiment evidence")
    p.add_argument("--experiment",type=Path,required=True)
    p.add_argument("--registry",type=Path,required=True)
    p.add_argument("--name",required=True)
    p.add_argument("--version",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--blind-report",type=Path)
    a=p.parse_args(argv)
    packet=generate_evidence_packet(
        a.experiment,DatasetRegistry(a.registry),
        dataset_name=a.name,dataset_version=a.version,
        blind_report=a.blind_report,
    )
    write_packet(a.output,packet)
    print(json.dumps({
        "accepted":packet.accepted,"checks":packet.checks
    },indent=2))
    return 0 if packet.accepted else 2


if __name__=="__main__":
    raise SystemExit(main())
