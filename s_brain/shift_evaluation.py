"""S-Learning Cycle 8: distribution-shift and calibrated evaluation.

Independent next-byte evidence must distinguish an in-distribution split from
new domains, altered wording and harder examples. All reports are descriptive
until scored on unseen, independently reviewed tests. This harness reads
SBrain checkpoints and supervised records, never calls Ollama, and rejects
question overlap or shared source hashes across evaluation groups.

Confidence intervals resample at RECORD level rather than pretending byte
positions are independent observations. Thresholds are declared before
scoring, while repeated examination of a sealed test invalidates blindness.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Iterable

from .checkpoint import load_weights
from .evaluate_learning import RecordScore,score_record,aggregate_scores
from .supervised_data import (
    SupervisedRecord,detect_train_test_leakage,fingerprint_dataset,
    import_jsonl,
)


@dataclasses.dataclass(frozen=True)
class ShiftProtocol:
    max_accuracy_drop: float = 0.15
    max_relative_loss_increase: float = 0.50
    minimum_records_per_group: int = 2
    confidence: float = 0.95
    bootstrap_repeats: int = 200
    seed: int = 2718
    require_disjoint_prompts: bool = True
    require_disjoint_sources: bool = True

    def validate(self):
        if any(not 0<=value<=1 for value in (
            self.max_accuracy_drop,self.confidence
        )) or self.confidence in (0,1):
            raise ValueError("Invalid fraction settings")
        if self.max_relative_loss_increase<0 or not math.isfinite(
            self.max_relative_loss_increase
        ):
            raise ValueError("Invalid relative loss bound")
        if self.minimum_records_per_group<1 or self.bootstrap_repeats<50:
            raise ValueError("Insufficient evaluation samples")


@dataclasses.dataclass(frozen=True)
class ShiftGroup:
    name: str
    explanation: str
    records: tuple[SupervisedRecord, ...]

    def validate(self):
        if not self.name or not self.explanation:
            raise ValueError("Group name and declared shift required")
        if not self.records:
            raise ValueError("Group must contain examples")
        if any(row.split not in ("validation","test") for row in self.records):
            raise ValueError("Shift evaluation cannot contain training examples")
        if len({row.identifier for row in self.records})!=len(self.records):
            raise ValueError("Duplicate example ID within shift group")

    @property
    def fingerprint(self):
        return fingerprint_dataset(list(self.records))


@dataclasses.dataclass
class ShiftOutcome:
    group: str
    records: int
    token_count: int
    accuracy: float
    cross_entropy: float
    perplexity: float
    category_details: dict
    interval: dict
    fingerprint: str

    def as_dict(self):
        return dataclasses.asdict(self)


def validate_group_isolation(
    base:ShiftGroup,others:list[ShiftGroup],
    policy:ShiftProtocol,
)->dict:
    base.validate()
    policy.validate()
    seen_names={base.name}
    all_groups=[base]+others
    for item in others:
        item.validate()
        if item.name in seen_names:
            raise ValueError("Repeated shift group name")
        seen_names.add(item.name)
    ids={}
    source_hashes={}
    prompts={}
    issues=[]
    for group in all_groups:
        for record in group.records:
            if record.identifier in ids:
                issues.append(f"duplicate_id:{record.identifier}")
            ids[record.identifier]=group.name
            if policy.require_disjoint_sources:
                if record.source_hash in source_hashes:
                    issues.append(f"shared_source:{record.identifier}")
                source_hashes[record.source_hash]=group.name
            if policy.require_disjoint_prompts:
                prompt=hashlib.sha256(bytes(record.prompt_bytes)).hexdigest()
                if prompt in prompts:
                    issues.append(f"shared_prompt:{record.identifier}")
                prompts[prompt]=group.name
    return {
        "isolated":not issues,
        "issues":issues,
        "groups":len(all_groups),
        "records":sum(len(item.records) for item in all_groups),
        "fingerprints":{
            item.name:item.fingerprint for item in all_groups
        },
    }


def weighted_record_interval(
    scores:list[RecordScore],metric:str,policy:ShiftProtocol,
)->dict:
    if metric not in ("accuracy","cross_entropy") or not scores:
        raise ValueError("Unsupported bootstrap metric")
    policy.validate()
    randomizer=random.Random(policy.seed)
    estimates=[]
    for _ in range(policy.bootstrap_repeats):
        batch=randomizer.choices(scores,k=len(scores))
        summary=aggregate_scores(batch,max_errors=0)
        estimates.append(summary[metric])
    estimates.sort()
    alpha=1-policy.confidence
    lo=int(alpha/2*(len(estimates)-1))
    hi=int((1-alpha/2)*(len(estimates)-1))
    return {
        "metric":metric,
        "estimate":aggregate_scores(scores,max_errors=0)[metric],
        "lower":estimates[lo],"upper":estimates[hi],
        "confidence":policy.confidence,
        "sampling_unit":"whole_records",
        "bootstrap_repeats":policy.bootstrap_repeats,
    }


def score_group(model,group:ShiftGroup,policy:ShiftProtocol)->ShiftOutcome:
    group.validate()
    policy.validate()
    if len(group.records)<policy.minimum_records_per_group:
        raise ValueError("Shift group too small for declared policy")
    device=str(next(model.parameters()).device)
    scores=[
        score_record(model,row,model.cfg.context,device)
        for row in group.records
    ]
    aggregated=aggregate_scores(scores,max_errors=0)
    return ShiftOutcome(
        group=group.name,records=aggregated["records"],
        token_count=aggregated["tokens"],
        accuracy=aggregated["accuracy"],
        cross_entropy=aggregated["cross_entropy"],
        perplexity=aggregated["perplexity"],
        category_details=aggregated["categories"],
        interval={
            "accuracy":weighted_record_interval(scores,"accuracy",policy),
            "cross_entropy":weighted_record_interval(scores,"cross_entropy",policy),
        },
        fingerprint=group.fingerprint,
    )


def evaluate_shift(
    model,base:ShiftGroup,
    shifted:list[ShiftGroup],*,
    policy:ShiftProtocol|None=None,
)->dict:
    policy=policy or ShiftProtocol()
    audit=validate_group_isolation(base,shifted,policy)
    if not audit["isolated"]:
        raise ValueError("Split leakage: "+",".join(audit["issues"]))
    base_score=score_group(model,base,policy)
    shifted_scores=[score_group(model,item,policy) for item in shifted]
    comparisons={}
    for item in shifted_scores:
        accuracy_drop=base_score.accuracy-item.accuracy
        loss_increase=(
            (item.cross_entropy-base_score.cross_entropy)
            / max(1e-12,base_score.cross_entropy)
        )
        comparisons[item.group]={
            "accuracy_drop":accuracy_drop,
            "relative_loss_increase":loss_increase,
            "accuracy_gate":accuracy_drop<=policy.max_accuracy_drop,
            "loss_gate":loss_increase<=policy.max_relative_loss_increase,
            "robust":(
                accuracy_drop<=policy.max_accuracy_drop
                and loss_increase<=policy.max_relative_loss_increase
            ),
        }
    return {
        "base":base_score.as_dict(),
        "shifts":[x.as_dict() for x in shifted_scores],
        "comparisons":comparisons,
        "isolation":audit,
        "policy":dataclasses.asdict(policy),
        "robust_to_all_tested_shifts":all(x["robust"] for x in comparisons.values()),
        "note":"Teacher-forced byte scores measure language modeling, not reasoning",
    }


def compare_model_shift_reports(before:dict,after:dict)->dict:
    if before["isolation"]["fingerprints"]!=after["isolation"]["fingerprints"]:
        raise ValueError("Different test sets cannot be compared")
    if before["policy"]!=after["policy"]:
        raise ValueError("Evaluation protocols differ")
    pairs={}
    for name in before["comparisons"]:
        if name not in after["comparisons"]:
            raise ValueError("Missing comparison shift")
        original=before["comparisons"][name]
        candidate=after["comparisons"][name]
        pairs[name]={
            "accuracy_drop_reduction":(
                original["accuracy_drop"]-candidate["accuracy_drop"]
            ),
            "relative_loss_increase_reduction":(
                original["relative_loss_increase"]
                -candidate["relative_loss_increase"]
            ),
            "robust_before":original["robust"],
            "robust_after":candidate["robust"],
        }
    return {
        "same_protocol":True,
        "comparisons":pairs,
        "newly_robust_shifts":sum(
            x["robust_after"] and not x["robust_before"] for x in pairs.values()
        ),
    }


def category_regressions(
    before:ShiftOutcome,after:ShiftOutcome,
    accuracy_tolerance:float=0.01,
)->dict:
    if before.fingerprint!=after.fingerprint:
        raise ValueError("Category reports use different evaluation records")
    original=before.category_details
    candidate=after.category_details
    if set(original)!=set(candidate):
        raise ValueError("Reported categories differ")
    results={}
    for category in sorted(original):
        delta=candidate[category]["accuracy"]-original[category]["accuracy"]
        results[category]={
            "accuracy_gain":delta,
            "loss_reduction":(
                original[category]["cross_entropy"]
                -candidate[category]["cross_entropy"]
            ),
            "passes":delta>=-accuracy_tolerance,
        }
    return results


def write_report(path:Path,report:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError("Do not overwrite an evaluation report")
    path.write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8"
    )


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description="Evaluate held-out distribution shift")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--base",type=Path,required=True)
    p.add_argument("--shift",type=Path,action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv)
    model,manifest=load_weights(args.checkpoint,"cpu")
    base=ShiftGroup("reference","In-distribution holdout",
                    tuple(import_jsonl(args.base)))
    shifts=[
        ShiftGroup(f"shift-{i}","Operator-declared distribution change",
                   tuple(import_jsonl(path)))
        for i,path in enumerate(args.shift)
    ]
    report=evaluate_shift(model,base,shifts)
    report["model_weights_sha256"]=manifest.weights_sha256
    write_report(args.output,report)
    print(json.dumps({
        "robust_to_all_tested_shifts":report["robust_to_all_tested_shifts"],
        "comparisons":report["comparisons"],
    },indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
