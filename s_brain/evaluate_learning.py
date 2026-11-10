"""Independent S-Brain checkpoint comparison and error analysis.

Evaluation uses held-out human-authored responses and S's own logits.
It never calls Ollama. Metrics include mean answer-only cross entropy,
perplexity, teacher-forced top-1 token accuracy, accuracy by category,
confidence intervals and checkpoint-to-checkpoint deltas.

Important: teacher-forced accuracy is not free-form answer correctness.
A small dataset's apparent gain cannot prove frontier capability.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

import torch

from .checkpoint import load_weights, load_manifest
from .supervised_data import (
    SupervisedRecord, TrainingWindow, build_windows, import_jsonl,
    fingerprint_dataset, detect_train_test_leakage,
)


@dataclasses.dataclass(frozen=True)
class EvalSettings:
    context: int = 64
    batch_size: int = 8
    seed: int = 42
    repetitions: int = 200
    confidence: float = 0.95
    max_error_examples: int = 20

    def validate(self):
        if self.context < 4 or self.batch_size < 1:
            raise ValueError("Invalid context or batch size")
        if not 0 < self.confidence < 1:
            raise ValueError("Invalid confidence level")
        if self.repetitions < 50 or self.max_error_examples < 0:
            raise ValueError("Invalid bootstrap or error example settings")


@dataclasses.dataclass(frozen=True)
class TokenError:
    record_id: str
    position: int
    expected_byte: int
    predicted_byte: int
    expected_probability: float
    nll: float

    def redacted(self):
        return {
            "record_hash": hashlib.sha256(self.record_id.encode("utf-8")).hexdigest(),
            "position": self.position,
            "expected_byte": self.expected_byte,
            "predicted_byte": self.predicted_byte,
            "expected_probability": self.expected_probability,
            "negative_log_likelihood": self.nll,
        }


@dataclasses.dataclass
class RecordScore:
    identifier: str
    category: str
    split: str
    tokens: int
    correct_tokens: int
    total_nll: float
    errors: list[TokenError]

    @property
    def accuracy(self):
        return self.correct_tokens / self.tokens if self.tokens else 0.0

    @property
    def cross_entropy(self):
        return self.total_nll / self.tokens if self.tokens else float("inf")

    def as_summary(self):
        return {
            "record_hash": hashlib.sha256(self.identifier.encode("utf-8")).hexdigest(),
            "category": self.category,
            "split": self.split,
            "tokens": self.tokens,
            "accuracy": self.accuracy,
            "cross_entropy": self.cross_entropy,
        }


def check_holdout(records:list[SupervisedRecord]):
    if not records:
        raise ValueError("Holdout must contain examples")
    if any(record.split not in ("validation","test") for record in records):
        raise ValueError("Holdout must not include training records")
    if len({item.identifier for item in records}) != len(records):
        raise ValueError("Duplicate holdout identifiers")
    if detect_train_test_leakage(records):
        raise ValueError("Holdout duplicates across splits")
    return fingerprint_dataset(records)


@torch.inference_mode()
def score_record(model,record:SupervisedRecord,context:int,device:str):
    """Score each byte exactly once using dense, masked completion windows."""
    model.eval()
    windows=build_windows([record],context,method="dense")
    if not windows:
        raise ValueError(f"No trainable windows for {record.identifier}")
    errors=[]
    count=0
    correct=0
    nll=0.0
    for index,window in enumerate(windows):
        input_ids=torch.tensor([window.token_ids],dtype=torch.long,device=device)
        logits,_=model(input_ids)
        scores=logits[0,-1,:].float()
        target=window.target_ids[-1]
        log_probs=torch.log_softmax(scores,dim=-1)
        loss=-float(log_probs[target])
        guess=int(scores.argmax())
        probability=float(log_probs[target].exp())
        correct+=int(guess==target)
        count+=1
        nll+=loss
        if guess!=target:
            errors.append(TokenError(
                record_id=record.identifier,
                position=index,
                expected_byte=target,
                predicted_byte=guess,
                expected_probability=probability,
                nll=loss,
            ))
    return RecordScore(
        record.identifier,record.category,record.split,
        count,correct,nll,errors,
    )


def aggregate_scores(scores:list[RecordScore],max_errors:int=20):
    if not scores:
        raise ValueError("No scored examples")
    total=sum(item.tokens for item in scores)
    if total<=0:
        raise ValueError("No scored tokens")
    accuracy=sum(item.correct_tokens for item in scores)/total
    loss=sum(item.total_nll for item in scores)/total
    groups=collections.defaultdict(list)
    for item in scores:
        groups[item.category].append(item)
    categories={}
    for name,items in sorted(groups.items()):
        token_count=sum(item.tokens for item in items)
        categories[name]={
            "records":len(items),
            "tokens":token_count,
            "accuracy":sum(item.correct_tokens for item in items)/token_count,
            "cross_entropy":sum(item.total_nll for item in items)/token_count,
        }
    error_samples=sorted(
        (error for item in scores for error in item.errors),
        key=lambda error:error.nll,
        reverse=True,
    )[:max_errors]
    return {
        "records":len(scores),
        "tokens":total,
        "accuracy":accuracy,
        "cross_entropy":loss,
        "perplexity":math.exp(loss) if loss<700 else float("inf"),
        "categories":categories,
        "errors":[error.redacted() for error in error_samples],
    }


def bootstrap_metric(
    scores:list[RecordScore],
    metric:str,
    *,
    repetitions:int=200,
    seed:int=42,
    confidence:float=0.95,
):
    if not scores or metric not in ("accuracy","cross_entropy"):
        raise ValueError("Invalid bootstrap selection")
    if repetitions<50 or not 0<confidence<1:
        raise ValueError("Invalid bootstrap settings")
    rng=random.Random(seed)
    values=[]
    for _ in range(repetitions):
        sample=rng.choices(scores,k=len(scores))
        report=aggregate_scores(sample,max_errors=0)
        values.append(report[metric])
    values.sort()
    alpha=1-confidence
    left=min(len(values)-1,int((alpha/2)*len(values)))
    right=min(len(values)-1,int((1-alpha/2)*len(values)))
    return {
        "metric":metric,
        "confidence":confidence,
        "lower":values[left],
        "upper":values[right],
        "resamples":repetitions,
        "sampling_unit":"record",
    }


def evaluate_checkpoint(
    model,
    records:list[SupervisedRecord],
    settings:EvalSettings|None=None,
)->dict:
    settings=settings or EvalSettings(context=model.cfg.context)
    settings.validate()
    check_holdout(records)
    if settings.context!=model.cfg.context:
        raise ValueError("Evaluation context differs from architecture")
    device=str(next(model.parameters()).device)
    scored=[score_record(model,row,settings.context,device) for row in records]
    result=aggregate_scores(scored,max_errors=settings.max_error_examples)
    result["fingerprint"]=fingerprint_dataset(records)
    result["accuracy_confidence_interval"]=bootstrap_metric(
        scored,"accuracy",repetitions=settings.repetitions,
        seed=settings.seed,confidence=settings.confidence,
    )
    result["loss_confidence_interval"]=bootstrap_metric(
        scored,"cross_entropy",repetitions=settings.repetitions,
        seed=settings.seed,confidence=settings.confidence,
    )
    result["note"]="Teacher-forced byte prediction, not an instruction accuracy benchmark"
    return result


def paired_evaluate(
    baseline_model,
    candidate_model,
    records:list[SupervisedRecord],
    settings:EvalSettings|None=None,
)->dict:
    if dataclasses.asdict(baseline_model.cfg)!=dataclasses.asdict(candidate_model.cfg):
        raise ValueError("Architecture mismatch")
    settings=settings or EvalSettings(context=baseline_model.cfg.context)
    original=evaluate_checkpoint(baseline_model,records,settings)
    improved=evaluate_checkpoint(candidate_model,records,settings)
    return {
        "dataset_fingerprint":original["fingerprint"],
        "baseline":original,
        "candidate":improved,
        "loss_reduction":original["cross_entropy"]-improved["cross_entropy"],
        "accuracy_gain":improved["accuracy"]-original["accuracy"],
        "improved_loss":improved["cross_entropy"]<original["cross_entropy"],
        "proven_general_intelligence":False,
    }


def compare_saved_checkpoints(
    baseline:Path,candidate:Path,holdout:Path,
    *,
    device:str="cpu",settings:EvalSettings|None=None,
):
    original,first=load_weights(baseline,device)
    second,last=load_weights(candidate,device)
    records=import_jsonl(holdout)
    result=paired_evaluate(original,second,records,settings)
    result["weights_changed"]=first.weights_sha256!=last.weights_sha256
    result["baseline_step"]=first.step
    result["candidate_step"]=last.step
    result["same_training_dataset"]=(
        first.dataset_fingerprint==last.dataset_fingerprint
    )
    return result


def pass_improvement_gate(
    report:dict,
    *,
    minimum_loss_reduction:float=0.01,
    max_accuracy_drop:float=0.01,
):
    if minimum_loss_reduction<0 or max_accuracy_drop<0:
        raise ValueError("Improvement gate cannot use negative thresholds")
    return {
        "passed":(
            report["loss_reduction"]>=minimum_loss_reduction
            and report["accuracy_gain"]>=-max_accuracy_drop
            and report.get("weights_changed",True)
        ),
        "loss_reduction":report["loss_reduction"],
        "accuracy_gain":report["accuracy_gain"],
        "threshold":minimum_loss_reduction,
        "note":"Gates confirm selected metrics only; no frontier-equivalence assertion",
    }


def save_report(path:Path,report:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")


def main(argv=None):
    parser=argparse.ArgumentParser(description="Blind checkpoint evaluation, no Ollama calls")
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--holdout",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--minimum-loss-reduction",type=float,default=0.01)
    args=parser.parse_args(argv)
    report=compare_saved_checkpoints(
        args.baseline,args.candidate,args.holdout,device=args.device
    )
    report["gate"]=pass_improvement_gate(
        report,minimum_loss_reduction=args.minimum_loss_reduction
    )
    save_report(args.output,report)
    print(json.dumps({
        "loss_reduction":report["loss_reduction"],
        "accuracy_gain":report["accuracy_gain"],
        "gate":report["gate"],
    },indent=2))
    return 0 if report["gate"]["passed"] else 2


if __name__=="__main__":
    raise SystemExit(main())
