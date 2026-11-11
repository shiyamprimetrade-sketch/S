"""Byte-level supervised learning records for S-Brain.

Turn reviewed teacher examples into an explicit completion objective:
given a prompt prefix, predict answer bytes; prompt bytes are context only,
not supervised target labels. The loss mask uses -100 (PyTorch ignore_index).
This is response text supervision, NOT access to teacher probability logits.
The byte tokenizer here matches SBrain vocab_size=256 exactly.

All data remains local to the operator; no training artifacts are committed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import torch

from .teacher_data import TeacherExample, eligible_training_examples


@dataclasses.dataclass(frozen=True)
class FormatPolicy:
    question_prefix: str = "Question: "
    answer_prefix: str = "\nAnswer: "
    terminator: str = "\n<END>\n"
    max_question_bytes: int = 1024
    max_answer_bytes: int = 4096
    minimum_answer_bytes: int = 2
    allow_truncation: bool = False

    def validate(self):
        if any(not value for value in (
            self.question_prefix,self.answer_prefix,self.terminator
        )):
            raise ValueError("Prompt markers must be nonempty")
        if self.max_question_bytes < 1 or self.max_answer_bytes < 1:
            raise ValueError("Invalid byte limits")
        if not 1 <= self.minimum_answer_bytes <= self.max_answer_bytes:
            raise ValueError("Invalid minimum answer size")
        if self.answer_prefix in self.question_prefix:
            raise ValueError("Ambiguous prompt delimiters")


@dataclasses.dataclass(frozen=True)
class SupervisedRecord:
    identifier: str
    category: str
    source_hash: str
    split: str
    prompt_bytes: tuple[int, ...]
    completion_bytes: tuple[int, ...]
    license: str

    @property
    def total_tokens(self):
        return len(self.prompt_bytes) + len(self.completion_bytes)

    def validate(self):
        if self.split not in ("train","validation","test"):
            raise ValueError("Invalid split")
        if not self.identifier or not self.source_hash or not self.license:
            raise ValueError("Missing provenance")
        if not self.prompt_bytes or not self.completion_bytes:
            raise ValueError("Empty training example")
        if not all(0 <= value <= 255 for value in (
            self.prompt_bytes + self.completion_bytes
        )):
            raise ValueError("Token exceeds byte vocabulary")


def formatted_prompt(question: str, policy: FormatPolicy) -> str:
    policy.validate()
    if not isinstance(question,str) or not question.strip():
        raise ValueError("Question required")
    return policy.question_prefix + question.strip() + policy.answer_prefix


def encode_record(
    identifier: str,
    question: str,
    answer: str,
    *,
    category: str,
    source_hash: str,
    permission: str,
    split: str = "train",
    policy: FormatPolicy | None = None,
) -> SupervisedRecord:
    policy = policy or FormatPolicy()
    policy.validate()
    prompt = formatted_prompt(question,policy).encode("utf-8")
    response = (answer.strip()+policy.terminator).encode("utf-8")
    question_size = len(question.encode("utf-8"))
    answer_size = len(answer.strip().encode("utf-8"))
    if question_size > policy.max_question_bytes:
        if not policy.allow_truncation:
            raise ValueError("Question exceeds maximum")
        prompt = prompt[-(policy.max_question_bytes+len(
            policy.answer_prefix.encode("utf-8")
        )):]
    if answer_size < policy.minimum_answer_bytes:
        raise ValueError("Answer too short")
    if answer_size > policy.max_answer_bytes:
        if not policy.allow_truncation:
            raise ValueError("Answer exceeds maximum")
        response = response[:policy.max_answer_bytes]
    record = SupervisedRecord(
        identifier=identifier,category=category,source_hash=source_hash,
        split=split,prompt_bytes=tuple(prompt),
        completion_bytes=tuple(response),license=permission,
    )
    record.validate()
    return record


def from_teacher_examples(
    examples: Iterable[TeacherExample],
    *,
    policy: FormatPolicy | None = None,
) -> list[SupervisedRecord]:
    """Only approved train examples are admitted. No unseen test leakage."""
    approved = eligible_training_examples(list(examples))
    return [
        encode_record(
            identifier=item.identifier,question=item.question,answer=item.answer,
            category=item.category,source_hash=item.answer_sha256,
            permission=item.permission,split="train",policy=policy,
        )
        for item in approved
    ]


def record_fingerprint(record: SupervisedRecord) -> str:
    record.validate()
    digest = hashlib.sha256()
    digest.update(bytes(record.prompt_bytes))
    digest.update(b"\0")
    digest.update(bytes(record.completion_bytes))
    return digest.hexdigest()


def detect_train_test_leakage(records: Iterable[SupervisedRecord]) -> list[str]:
    split_by_prompt = {}
    violations = []
    for item in records:
        question_hash = hashlib.sha256(bytes(item.prompt_bytes)).hexdigest()
        previous = split_by_prompt.setdefault(question_hash,item.split)
        if previous != item.split:
            violations.append(item.identifier)
    return violations


def audit_records(records: Iterable[SupervisedRecord]) -> dict:
    values = list(records)
    hashes = [record_fingerprint(item) for item in values]
    return {
        "records":len(values),
        "unique":len(set(hashes)),
        "duplicate_fraction":1-len(set(hashes))/max(1,len(values)),
        "split_counts":dict(Counter(item.split for item in values)),
        "category_counts":dict(Counter(item.category for item in values)),
        "target_tokens":sum(len(item.completion_bytes) for item in values),
        "input_tokens":sum(item.total_tokens for item in values),
        "split_leakage":detect_train_test_leakage(values),
    }


def export_jsonl(path:Path,records:Iterable[SupervisedRecord]) -> int:
    path.parent.mkdir(parents=True,exist_ok=True)
    count = 0
    with path.open("w",encoding="utf-8") as handle:
        for item in records:
            item.validate()
            obj = dataclasses.asdict(item)
            obj["prompt_bytes"] = list(item.prompt_bytes)
            obj["completion_bytes"] = list(item.completion_bytes)
            obj["fingerprint"] = record_fingerprint(item)
            handle.write(json.dumps(obj,ensure_ascii=False)+"\n")
            count += 1
    return count


def import_jsonl(path:Path) -> list[SupervisedRecord]:
    records=[]
    seen=set()
    with path.open("r",encoding="utf-8") as file:
        for number,line in enumerate(file,1):
            if not line.strip():
                continue
            try:
                value=json.loads(line)
                fingerprint=value.pop("fingerprint")
                value["prompt_bytes"]=tuple(value["prompt_bytes"])
                value["completion_bytes"]=tuple(value["completion_bytes"])
                record=SupervisedRecord(**value)
                record.validate()
                if record_fingerprint(record)!=fingerprint:
                    raise ValueError("Training record digest mismatch")
                if record.identifier in seen:
                    raise ValueError("Duplicate record identifier")
            except (TypeError,ValueError,KeyError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            seen.add(record.identifier)
            records.append(record)
    return records


@dataclasses.dataclass(frozen=True)
class TrainingWindow:
    token_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    supervised_tokens: int
    identifier: str

    def validate(self, context:int):
        if len(self.token_ids)!=context or len(self.target_ids)!=context:
            raise ValueError("Window must have fixed context")
        if self.supervised_tokens<1:
            raise ValueError("Window needs at least one supervised label")
        if self.supervised_tokens!=sum(i!=-100 for i in self.target_ids):
            raise ValueError("Supervised token count mismatch")


def make_windows(
    record:SupervisedRecord,context:int,
    *,
    stride:int|None=None,
    pad_id:int=0,
) -> list[TrainingWindow]:
    """Construct causal next-byte windows with only completion bytes labeled.

    Every answer byte appears as a target at least once. Windows overlap for
    context but loss is computed only on the designated supervised targets.
    Padding inputs are ignored by targets and are never included after the
    active sequence; this avoids shifting active RoPE positions.
    """
    record.validate()
    if context<4:
        raise ValueError("Context must be >= 4")
    if not 0<=pad_id<=255:
        raise ValueError("Invalid padding byte")
    if stride is None:
        stride=max(1,context//2)
    if not 1<=stride<context:
        raise ValueError("Invalid window stride")
    joined=record.prompt_bytes+record.completion_bytes
    response_start=len(record.prompt_bytes)
    # Predict token i from tokens preceding i. Use only prefix-complete
    # right-aligned windows, so first trained answer token sees full prefix
    # when possible.
    window_positions=list(range(response_start,len(joined),stride))
    if not window_positions or window_positions[-1]!=len(joined)-1:
        window_positions.append(len(joined)-1)
    results=[]
    covered=set()
    for last_target in window_positions:
        # Input span has up to context tokens ending at last_target-1.
        # Target span begins at input+1, preserving one-token shift.
        end=min(len(joined)-1,last_target+stride-1)
        begin=max(0,end-context)
        input_ids=list(joined[begin:end])
        target_ids=list(joined[begin+1:end+1])
        if not input_ids:
            continue
        labels=[]
        for target_index,value in enumerate(target_ids,start=begin+1):
            if target_index>=response_start:
                labels.append(value)
                covered.add(target_index)
            else:
                labels.append(-100)
        if not any(x!=-100 for x in labels):
            continue
        # Pad at the left side. Since RoPE is relative, shifting all active
        # positions equally preserves relative attention within active span,
        # but padded keys would be attended to. To avoid that, do NOT train
        # on shorter windows in this fixed-context sampler.
        if len(input_ids)<context:
            continue
        item=TrainingWindow(
            tuple(input_ids),tuple(labels),
            sum(x!=-100 for x in labels),record.identifier,
        )
        item.validate(context)
        results.append(item)
    return results


def dense_completion_windows(
    record:SupervisedRecord,
    context:int,
) -> list[TrainingWindow]:
    """Generate one target per window for maximally clear loss accounting.

    This is slower than grouped supervised tokens but avoids double-counting
    labels in overlapping windows. The prefix must be at least context bytes,
    otherwise there is no full context to feed the original Transformer.
    """
    values=record.prompt_bytes+record.completion_bytes
    start=len(record.prompt_bytes)
    results=[]
    for target_index in range(max(start,context),len(values)):
        inputs=tuple(values[target_index-context:target_index])
        labels=(-100,)*(context-1)+(values[target_index],)
        window=TrainingWindow(inputs,labels,1,record.identifier)
        window.validate(context)
        results.append(window)
    return results


def build_windows(
    records:Iterable[SupervisedRecord],
    context:int,
    *,
    method:str="dense",
) -> list[TrainingWindow]:
    all_records=list(records)
    leakage=detect_train_test_leakage(all_records)
    if leakage:
        raise ValueError("A prompt appears in multiple dataset splits")
    results=[]
    for item in all_records:
        windows=(
            dense_completion_windows(item,context)
            if method=="dense" else make_windows(item,context)
        )
        results.extend(windows)
    return results


class WindowSampler:
    def __init__(self,windows:list[TrainingWindow],context:int,seed:int=42):
        if not windows:
            raise ValueError("Cannot sample an empty dataset")
        for window in windows:
            window.validate(context)
        self.windows=windows
        self.context=context
        self.random=random.Random(seed)

    def sample(self,batch_size:int,device:str="cpu"):
        if batch_size<1:
            raise ValueError("Invalid batch size")
        selected=self.random.choices(self.windows,k=batch_size)
        x=torch.tensor([row.token_ids for row in selected],dtype=torch.long,device=device)
        y=torch.tensor([row.target_ids for row in selected],dtype=torch.long,device=device)
        supervised=sum(row.supervised_tokens for row in selected)
        return x,y,supervised


def masked_cross_entropy(logits:torch.Tensor,targets:torch.Tensor)->torch.Tensor:
    if logits.shape[:2]!=targets.shape:
        raise ValueError("Logits/targets shape mismatch")
    if logits.size(-1)!=256:
        raise ValueError("S-Brain byte vocabulary expected")
    if not bool((targets!=-100).any()):
        raise ValueError("All labels are masked")
    return torch.nn.functional.cross_entropy(
        logits.float().reshape(-1,logits.shape[-1]),
        targets.reshape(-1),ignore_index=-100
    )


def fingerprint_dataset(records:list[SupervisedRecord])->str:
    digest=hashlib.sha256()
    for row in sorted(records,key=lambda x:x.identifier):
        digest.update(row.identifier.encode("utf-8"))
        digest.update(record_fingerprint(row).encode("ascii"))
        digest.update(row.split.encode("ascii"))
    return digest.hexdigest()


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description="Create supervised bytes from reviewed Ollama answers")
    parser.add_argument("--approved-examples",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    from .teacher_data import load_examples
    rows=from_teacher_examples(load_examples(args.approved_examples))
    if not rows:
        parser.error("No human-reviewed and approved training examples")
    export_jsonl(args.output,rows)
    print(json.dumps(audit_records(rows),indent=2))


if __name__=="__main__":
    main()
