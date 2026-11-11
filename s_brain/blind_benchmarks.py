"""Blind benchmark protocol for independent S-Brain language-model evidence.

This module scores held-out byte-completion examples, records frozen test-set
fingerprints and checks model checkpoints against predefined acceptance
thresholds. It never calls Ollama or touches the live Continuum server.

The benchmark uses teacher-forced next-byte log loss and per-byte accuracy.
These are meaningful diagnostics of learned next-token prediction, NOT a
human preference test, factual knowledge test or broad reasoning benchmark.

A benchmark can only be called 'blind' if the test examples were authored
and sealed separately from training and *not* used in model selection.
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

from .supervised_data import (
    SupervisedRecord, import_jsonl, fingerprint_dataset,
    detect_train_test_leakage,
)
from .checkpoint import load_manifest, load_weights
from .evaluate_learning import EvalSettings, evaluate_checkpoint


@dataclasses.dataclass(frozen=True)
class BenchmarkProtocol:
    name: str
    version: str
    test_fingerprint: str
    record_count: int
    minimum_accuracy: float = 0.0
    maximum_cross_entropy: float = 10.0
    minimum_categories: int = 1
    minimum_tokens: int = 1
    require_reviewer: bool = True
    category_floors: dict[str, float] = dataclasses.field(default_factory=dict)
    explicit_no_selection_usage: bool = True

    def validate(self):
        if not self.name or not self.version:
            raise ValueError("Benchmark name and version required")
        if len(self.test_fingerprint)!=64:
            raise ValueError("Frozen test fingerprint must be SHA-256")
        if self.record_count<1 or self.minimum_categories<1 or self.minimum_tokens<1:
            raise ValueError("Invalid benchmark size thresholds")
        if not 0<=self.minimum_accuracy<=1:
            raise ValueError("Invalid required accuracy")
        if self.maximum_cross_entropy<=0 or not math.isfinite(self.maximum_cross_entropy):
            raise ValueError("Invalid maximum cross-entropy")
        if any(not 0<=floor<=1 for floor in self.category_floors.values()):
            raise ValueError("Category accuracy floor must be within [0,1]")
        if not self.explicit_no_selection_usage:
            raise ValueError("Blind test cannot be used for checkpoint selection")


@dataclasses.dataclass(frozen=True)
class BenchmarkSeal:
    protocol: BenchmarkProtocol
    authored_by: str
    reviewed_by: str
    license_description: str
    creation_note: str
    source_file_sha256: str

    def validate(self):
        self.protocol.validate()
        if not self.authored_by or not self.license_description:
            raise ValueError("Benchmark authorship and permission required")
        if self.protocol.require_reviewer and not self.reviewed_by:
            raise ValueError("Independent reviewer required")
        if len(self.source_file_sha256)!=64:
            raise ValueError("Source bytes hash must be recorded")

    def as_dict(self):
        return dataclasses.asdict(self)


def file_digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def seal_benchmark(
    test_path: Path, *,
    name: str, version: str,
    author: str, reviewer: str,
    license_description: str,
    minimum_accuracy: float = 0.0,
    maximum_cross_entropy: float = 10.0,
    category_floors: dict[str,float] | None = None,
) -> BenchmarkSeal:
    records=import_jsonl(test_path)
    if not records:
        raise ValueError("Cannot seal empty test data")
    if any(row.split!="test" for row in records):
        raise ValueError("Only blind test records can be sealed")
    if len({row.identifier for row in records})!=len(records):
        raise ValueError("Duplicate test record identifiers")
    if detect_train_test_leakage(records):
        raise ValueError("Same question appears in multiple test splits")
    protocol=BenchmarkProtocol(
        name=name,version=version,
        test_fingerprint=fingerprint_dataset(records),
        record_count=len(records),
        minimum_accuracy=minimum_accuracy,
        maximum_cross_entropy=maximum_cross_entropy,
        category_floors=category_floors or {},
    )
    seal=BenchmarkSeal(
        protocol=protocol,authored_by=author,
        reviewed_by=reviewer,license_description=license_description,
        creation_note="Frozen before model scoring",
        source_file_sha256=file_digest(test_path),
    )
    seal.validate()
    return seal


def seal_to_file(path: Path,seal: BenchmarkSeal):
    seal.validate()
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError("Benchmark seal already exists")
    _atomic_json(path,seal.as_dict())


def load_seal(path: Path)->BenchmarkSeal:
    raw=json.loads(path.read_text(encoding="utf-8"))
    protocol=BenchmarkProtocol(**raw["protocol"])
    seal=BenchmarkSeal(
        protocol=protocol,
        **{name:value for name,value in raw.items() if name!="protocol"}
    )
    seal.validate()
    return seal


def _atomic_json(path:Path,obj:dict):
    fd,name=tempfile.mkstemp(dir=path.parent,prefix=".benchmark-")
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def verify_sealed_test(
    seal: BenchmarkSeal,test_path: Path,
    *,
    train_records: list[SupervisedRecord] | None = None,
    validation_records: list[SupervisedRecord] | None = None,
) -> dict:
    seal.validate()
    actual=file_digest(test_path)
    records=import_jsonl(test_path)
    checks={
        "source_file_unchanged":actual==seal.source_file_sha256,
        "test_fingerprint_unchanged":(
            fingerprint_dataset(records)==seal.protocol.test_fingerprint
        ),
        "test_record_count":len(records)==seal.protocol.record_count,
        "only_test_split":all(row.split=="test" for row in records),
    }
    if train_records is not None or validation_records is not None:
        source=(train_records or [])+(validation_records or [])
        from .dataset_versions import audit_release,DatasetReleasePolicy
        audit=audit_release(
            train_records or [],validation_records or [],records,
            DatasetReleasePolicy(
                minimum_train=0,minimum_validation=0,minimum_test=1,
            ),
        )
        checks["no_training_overlap"]=audit["checks"]["no_exact_question_leakage"]
        checks["no_shared_source_hashes"]=audit["checks"]["unique_source_hashes"]
    return {
        "valid":all(checks.values()),
        "checks":checks,
        "frozen_fingerprint":seal.protocol.test_fingerprint,
        "note":"Lexical and source-hash checks do not exclude all semantic leakage",
    }


def acceptance_checks(metrics:dict,protocol:BenchmarkProtocol)->dict:
    protocol.validate()
    categories=metrics.get("categories",{})
    checks={
        "record_count":metrics["records"]>=protocol.record_count,
        "minimum_tokens":metrics["tokens"]>=protocol.minimum_tokens,
        "category_count":len(categories)>=protocol.minimum_categories,
        "accuracy":metrics["accuracy"]>=protocol.minimum_accuracy,
        "cross_entropy":metrics["cross_entropy"]<=protocol.maximum_cross_entropy,
    }
    for name,floor in protocol.category_floors.items():
        checks[f"category:{name}"]=(
            name in categories and categories[name]["accuracy"]>=floor
        )
    return checks


def evaluate_blind_checkpoint(
    checkpoint:Path,test_path:Path,seal:BenchmarkSeal,
    *,
    device:str="cpu",
    train_records:list[SupervisedRecord]|None=None,
    validation_records:list[SupervisedRecord]|None=None,
) -> dict:
    validation=verify_sealed_test(
        seal,test_path,
        train_records=train_records,validation_records=validation_records,
    )
    if not validation["valid"]:
        raise ValueError("Benchmark seal or split integrity failed")
    model,manifest=load_weights(checkpoint,device)
    records=import_jsonl(test_path)
    metrics=evaluate_checkpoint(
        model,records,
        EvalSettings(
            context=model.cfg.context,
            repetitions=50,
            max_error_examples=0,
        ),
    )
    checks=acceptance_checks(metrics,seal.protocol)
    return {
        "benchmark":{
            "name":seal.protocol.name,
            "version":seal.protocol.version,
            "test_fingerprint":seal.protocol.test_fingerprint,
        },
        "checkpoint":{
            "step":manifest.step,
            "weight_sha256":manifest.weights_sha256,
            "dataset_fingerprint":manifest.dataset_fingerprint,
        },
        "metrics":metrics,
        "acceptance_checks":checks,
        "accepted":all(checks.values()),
        "seal_verified":True,
        "used_to_select_checkpoint":False,
        "independent_of_teacher":True,
        "claim":"Held-out next-byte metric only; not a general reasoning assessment",
    }


def compare_blind_reports(before:dict,after:dict)->dict:
    if before["benchmark"]!=after["benchmark"]:
        raise ValueError("Blind benchmark versions or fingerprints differ")
    original,candidate=before["metrics"],after["metrics"]
    return {
        "same_benchmark":True,
        "accuracy_gain":candidate["accuracy"]-original["accuracy"],
        "loss_reduction":original["cross_entropy"]-candidate["cross_entropy"],
        "candidate_passed":after["accepted"],
        "descriptive_only":True,
        "warning":"Repeated inspection of blind test creates selection leakage",
    }


def write_report(path:Path,report:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    _atomic_json(path,report)


def main(argv=None):
    p=argparse.ArgumentParser(description="Evaluate frozen blind S-Brain byte benchmark")
    sub=p.add_subparsers(dest="action",required=True)
    seal_cmd=sub.add_parser("seal")
    seal_cmd.add_argument("--test",type=Path,required=True)
    seal_cmd.add_argument("--output",type=Path,required=True)
    seal_cmd.add_argument("--name",required=True)
    seal_cmd.add_argument("--version",required=True)
    seal_cmd.add_argument("--author",required=True)
    seal_cmd.add_argument("--reviewer",required=True)
    seal_cmd.add_argument("--license",required=True)
    evaluate_cmd=sub.add_parser("evaluate")
    evaluate_cmd.add_argument("--test",type=Path,required=True)
    evaluate_cmd.add_argument("--seal",type=Path,required=True)
    evaluate_cmd.add_argument("--checkpoint",type=Path,required=True)
    evaluate_cmd.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv)
    if args.action=="seal":
        value=seal_benchmark(
            args.test,name=args.name,version=args.version,
            author=args.author,reviewer=args.reviewer,
            license_description=args.license,
        )
        seal_to_file(args.output,value)
        print(json.dumps(value.as_dict(),indent=2))
    else:
        report=evaluate_blind_checkpoint(
            args.checkpoint,args.test,load_seal(args.seal),
        )
        write_report(args.output,report)
        print(json.dumps({
            "accepted":report["accepted"],
            "metrics":{
                name:report["metrics"][name]
                for name in ("accuracy","cross_entropy","perplexity")
            },
        },indent=2))
    return 0


if __name__=="__main__":
    main()
