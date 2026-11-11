"""Content-addressed, append-only dataset version registry for S-Brain.

Dataset releases contain immutable hashes of approved supervised records and
the review/provenance metadata supplied by their authors. The registry writes
JSON metadata, never raw private examples. It does not establish legal rights
by itself, cannot detect semantic leakage, and never modifies old releases.
A release is valid only when its data hashes match the original local files.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Iterable

from .supervised_data import (
    SupervisedRecord, import_jsonl, fingerprint_dataset,
    detect_train_test_leakage, record_fingerprint,
)


@dataclasses.dataclass(frozen=True)
class DatasetReleasePolicy:
    minimum_train: int = 1
    minimum_validation: int = 1
    minimum_test: int = 1
    require_all_splits: bool = True
    require_unique_ids: bool = True
    require_unique_source_hashes: bool = True
    require_permissions: bool = True
    forbid_test_training_overlap: bool = True

    def validate(self):
        for key in ("minimum_train","minimum_validation","minimum_test"):
            if getattr(self,key) < 0:
                raise ValueError("Minimum counts cannot be negative")


@dataclasses.dataclass(frozen=True)
class DatasetPart:
    split: str
    record_count: int
    target_tokens: int
    input_tokens: int
    fingerprint: str
    file_sha256: str
    categories: dict[str, int]
    origin_path: str

    def validate(self):
        if self.split not in ("train","validation","test"):
            raise ValueError("Unrecognized split")
        if self.record_count < 0 or self.target_tokens < 0:
            raise ValueError("Invalid dataset counters")
        if len(self.fingerprint) != 64 or len(self.file_sha256) != 64:
            raise ValueError("Invalid dataset fingerprint")


@dataclasses.dataclass(frozen=True)
class DatasetRelease:
    name: str
    version: str
    created_at_epoch: int
    author: str
    license_description: str
    review_protocol: str
    train: DatasetPart
    validation: DatasetPart
    test: DatasetPart
    release_fingerprint: str
    schema: int = 1

    def parts(self):
        return (self.train,self.validation,self.test)

    def as_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data,dict):
            raise ValueError("Dataset release must be an object")
        fields = {key: DatasetPart(**data[key]) for key in ("train","validation","test")}
        release = cls(
            **{key:value for key,value in data.items()
               if key not in ("train","validation","test")},
            **fields,
        )
        release.validate()
        return release

    def validate(self):
        if self.schema != 1:
            raise ValueError("Unsupported dataset schema")
        if not self.name or not self.version or not self.author:
            raise ValueError("Missing dataset identity")
        if "/" in self.name or "/" in self.version or ".." in self.name or ".." in self.version:
            raise ValueError("Unsafe dataset release ID")
        if not self.license_description or not self.review_protocol:
            raise ValueError("Source/license review required")
        for name,item in zip(("train","validation","test"),self.parts()):
            item.validate()
            if item.split != name:
                raise ValueError("Split identity mismatch")
        if self.release_fingerprint != release_digest(
            self.name,self.version,self.author,self.license_description,
            self.review_protocol,self.parts()
        ):
            raise ValueError("Release content hash mismatch")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def release_digest(name,version,author,license_description,review_protocol,parts):
    h = hashlib.sha256()
    meta = {
        "name":name,"version":version,"author":author,
        "license_description":license_description,"review_protocol":review_protocol,
        "parts":[{
            "split":p.split,"count":p.record_count,
            "file_sha256":p.file_sha256,"fingerprint":p.fingerprint,
        } for p in parts],
    }
    h.update(json.dumps(meta,sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def create_part(split: str, path: Path, records: list[SupervisedRecord]) -> DatasetPart:
    from collections import Counter
    if any(record.split != split for record in records):
        raise ValueError("Record loaded into incorrect split")
    return DatasetPart(
        split=split,record_count=len(records),
        target_tokens=sum(len(item.completion_bytes) for item in records),
        input_tokens=sum(item.total_tokens for item in records),
        fingerprint=fingerprint_dataset(records),
        file_sha256=sha256_file(path),
        categories=dict(Counter(row.category for row in records)),
        origin_path=str(path),
    )


def audit_release(
    train: list[SupervisedRecord],
    validation: list[SupervisedRecord],
    test: list[SupervisedRecord],
    policy: DatasetReleasePolicy | None = None,
) -> dict:
    policy = policy or DatasetReleasePolicy()
    policy.validate()
    splits={"train":train,"validation":validation,"test":test}
    ids=[row.identifier for values in splits.values() for row in values]
    sources=[row.source_hash for values in splits.values() for row in values]
    all_rows=train+validation+test
    overlap=detect_train_test_leakage(all_rows)
    checks={
        "train_count":len(train)>=policy.minimum_train,
        "validation_count":len(validation)>=policy.minimum_validation,
        "test_count":len(test)>=policy.minimum_test,
        "split_integrity":all(
            row.split==split for split,rows in splits.items() for row in rows
        ),
        "unique_ids":not policy.require_unique_ids or len(set(ids))==len(ids),
        "unique_source_hashes":(
            not policy.require_unique_source_hashes
            or len(set(sources))==len(sources)
        ),
        "source_permission":not policy.require_permissions or all(
            row.license and row.license.lower() not in (
                "unknown","unverified","none"
            ) for row in all_rows
        ),
        "no_exact_question_leakage":(
            not policy.forbid_test_training_overlap or not overlap
        ),
    }
    return {
        "accepted":all(checks.values()),
        "checks":checks,
        "record_counts":{name:len(values) for name,values in splits.items()},
        "question_leakage_ids":overlap,
        "limitations":"Does not detect all semantic leakage or verify legal permission",
    }


def create_release(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    *,
    name: str,
    version: str,
    author: str,
    license_description: str,
    review_protocol: str,
    policy: DatasetReleasePolicy | None = None,
    timestamp: int | None = None,
) -> DatasetRelease:
    sources = {
        "train":(train_path,import_jsonl(train_path)),
        "validation":(validation_path,import_jsonl(validation_path)),
        "test":(test_path,import_jsonl(test_path)),
    }
    audit = audit_release(*(sources[key][1] for key in (
        "train","validation","test"
    )),policy)
    if not audit["accepted"]:
        raise ValueError("Dataset release failed gates: " + str(audit["checks"]))
    parts=tuple(create_part(name_key,*sources[name_key]) for name_key in (
        "train","validation","test"
    ))
    release=DatasetRelease(
        name=name,version=version,created_at_epoch=(
            int(time.time()) if timestamp is None else timestamp
        ),
        author=author,license_description=license_description,
        review_protocol=review_protocol,
        train=parts[0],validation=parts[1],test=parts[2],
        release_fingerprint=release_digest(
            name,version,author,license_description,review_protocol,parts
        ),
    )
    release.validate()
    return release


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".s-dataset-",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as stream:
            json.dump(value,stream,indent=2,sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


class DatasetRegistry:
    """Version registry; each release ID is write-once."""

    def __init__(self,root: Path):
        self.root=root

    def path_for(self,name: str,version: str) -> Path:
        if not name or not version or "/" in name or "/" in version:
            raise ValueError("Invalid version identifier")
        if ".." in name or ".." in version or chr(92) in name or chr(92) in version:
            raise ValueError("Invalid version path")
        return self.root/name/(version+".json")

    def register(self,release: DatasetRelease):
        release.validate()
        path=self.path_for(release.name,release.version)
        if path.exists():
            raise FileExistsError("Version already registered")
        atomic_json(path,release.as_dict())
        return path

    def load(self,name: str,version: str,*,verify_files:bool=False):
        item=DatasetRelease.from_dict(
            json.loads(self.path_for(name,version).read_text("utf-8"))
        )
        if verify_files:
            for part in item.parts():
                file=Path(part.origin_path)
                if not file.exists() or sha256_file(file)!=part.file_sha256:
                    raise ValueError("Release source file changed or missing")
                rows=import_jsonl(file)
                if fingerprint_dataset(rows)!=part.fingerprint:
                    raise ValueError("Release content differs from manifest")
        return item

    def versions(self,name: str):
        folder=self.root/name
        if not folder.exists():
            return []
        return sorted(path.stem for path in folder.glob("*.json"))

    def compare(self,name: str,left: str,right: str):
        a,b=self.load(name,left),self.load(name,right)
        return {
            "same_release":a.release_fingerprint==b.release_fingerprint,
            "versions":(a.version,b.version),
            "parts":{
                x.split:{
                    "record_delta":y.record_count-x.record_count,
                    "target_token_delta":y.target_tokens-x.target_tokens,
                    "content_changed":x.fingerprint!=y.fingerprint,
                } for x,y in zip(a.parts(),b.parts())
            },
        }


def main(argv=None):
    p=argparse.ArgumentParser(description="Register an immutable supervised dataset version")
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--test",type=Path,required=True)
    p.add_argument("--registry",type=Path,required=True)
    p.add_argument("--name",required=True)
    p.add_argument("--version",required=True)
    p.add_argument("--author",required=True)
    p.add_argument("--license",required=True)
    p.add_argument("--review",required=True)
    a=p.parse_args(argv)
    release=create_release(
        a.train,a.validation,a.test,name=a.name,version=a.version,
        author=a.author,license_description=a.license,review_protocol=a.review,
    )
    destination=DatasetRegistry(a.registry).register(release)
    print(json.dumps({"path":str(destination),"release":release.release_fingerprint}))
    return 0


if __name__=="__main__":
    main()
