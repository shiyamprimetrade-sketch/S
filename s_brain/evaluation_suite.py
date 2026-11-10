"""Evaluation integrity, leakage diagnostics, stratification and paired comparisons.

Important: independent evaluation can detect regressions, but synthetic tasks
and text-completion scores are not sufficient to establish general reasoning.
All reports contain data provenance and explicit statistical assumptions.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Callable, Iterable

from .evaluate import BenchmarkCase, CaseResult, run_cases, score_answer

WORD = re.compile(r"\w+",re.UNICODE)


@dataclasses.dataclass(frozen=True)
class EvaluationPolicy:
    min_cases: int = 20
    max_duplicate_fraction: float = 0.0
    min_categories: int = 2
    min_category_cases: int = 3
    confidence_samples: int = 1000
    seed: int = 42
    fail_on_overlap: bool = True

    def validate(self):
        if self.min_cases < 1 or self.min_categories < 1 or self.min_category_cases < 1:
            raise ValueError("Minimum case counts must be positive")
        if not 0 <= self.max_duplicate_fraction < 1:
            raise ValueError("Duplicate fraction must be in [0,1)")
        if self.confidence_samples < 50:
            raise ValueError("At least 50 bootstrap iterations needed")


@dataclasses.dataclass(frozen=True)
class DataProvenance:
    dataset_name: str
    version: str
    author: str
    licensing: str
    evaluation_method: str
    source_fingerprint: str
    written_independently: bool
    contamination_reviewed: bool = False

    def validate(self):
        for key in ("dataset_name","version","author","licensing",
                    "evaluation_method","source_fingerprint"):
            if not getattr(self,key):
                raise ValueError(f"Missing provenance: {key}")
        if not self.written_independently:
            raise ValueError("Blind benchmark must be authored independently")

    def display(self):
        return dataclasses.asdict(self)


def fingerprint(value:str)->str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(text:str)->str:
    return " ".join(WORD.findall(text.casefold()))


def word_ngrams(text:str,n:int=3)->set[tuple[str,...]]:
    if n<1:
        raise ValueError("n must be positive")
    words=WORD.findall(text.casefold())
    return {tuple(words[i:i+n]) for i in range(max(0,len(words)-n+1))}


def jaccard(left:set,right:set)->float:
    union=len(left|right)
    return len(left&right)/union if union else 1.0


def token_overlap(left:str,right:str,n:int=3)->float:
    return jaccard(word_ngrams(left,n),word_ngrams(right,n))


def shingle_signature(text:str,window:int=3)->str:
    shingles=sorted(" ".join(item) for item in word_ngrams(text,window))
    return fingerprint("\n".join(shingles))


def duplicate_pairs(cases:Iterable[BenchmarkCase])->list[tuple[str,str]]:
    by_prompt={}
    duplicates=[]
    for item in cases:
        normalized=normalize_text(item.prompt)
        if normalized in by_prompt:
            duplicates.append((by_prompt[normalized],item.identifier))
        else:
            by_prompt[normalized]=item.identifier
    return duplicates


def contamination_pairs(
    cases:Iterable[BenchmarkCase],
    training_records:Iterable[str],
    threshold:float=0.8,
    ngram:int=3,
    max_pairs:int=100,
)->list[dict]:
    if not 0<=threshold<=1 or max_pairs<1:
        raise ValueError("Invalid contamination threshold or limit")
    corpus=[(fingerprint(text),word_ngrams(text,ngram)) for text in training_records]
    matches=[]
    for item in cases:
        query=word_ngrams(item.prompt,ngram)
        for source_id,source_ngrams in corpus:
            score=jaccard(query,source_ngrams)
            if score>=threshold:
                matches.append({
                    "case_id":item.identifier,
                    "training_digest":source_id,
                    "ngram_jaccard":score,
                })
                if len(matches)>=max_pairs:
                    return matches
    return matches


def inspect_benchmark(
    cases:list[BenchmarkCase],
    policy:EvaluationPolicy,
    training_texts:Iterable[str]=(),
)->dict:
    policy.validate()
    categories=collections.Counter(item.category for item in cases)
    duplicates=duplicate_pairs(cases)
    pair_count=len(duplicates)
    total=max(1,len(cases))
    contamination=contamination_pairs(cases,training_texts)
    issues=[]
    if len(cases)<policy.min_cases:
        issues.append("insufficient_cases")
    if len(categories)<policy.min_categories:
        issues.append("insufficient_categories")
    if any(n<policy.min_category_cases for n in categories.values()):
        issues.append("underrepresented_categories")
    if pair_count/total>policy.max_duplicate_fraction:
        issues.append("duplicate_prompts")
    if contamination and policy.fail_on_overlap:
        issues.append("potential_contamination")
    return {
        "accepted":not issues,
        "issues":issues,
        "case_count":len(cases),
        "by_category":dict(categories),
        "duplicate_pairs":duplicates,
        "near_overlap_count":len(contamination),
        "near_overlap_examples":contamination[:20],
        "fingerprint":fingerprint(
            json.dumps([dataclasses.asdict(x) for x in cases],sort_keys=True)
        ),
        "policy":dataclasses.asdict(policy),
    }


def category_accuracy(rows:Iterable[CaseResult])->dict:
    groups=collections.defaultdict(list)
    for item in rows:
        groups[item.category].append(int(item.passed))
    return {
        name:{
            "samples":len(values),
            "accuracy":statistics.mean(values),
            "correct":sum(values),
        }
        for name,values in sorted(groups.items())
    }


def paired_accuracy_delta(baseline:list[CaseResult],candidate:list[CaseResult])->dict:
    left={item.identifier:item for item in baseline}
    right={item.identifier:item for item in candidate}
    if set(left)!=set(right) or not left:
        raise ValueError("Paired evaluations require identical case IDs")
    ids=sorted(left)
    changes=[
        int(right[key].passed)-int(left[key].passed)
        for key in ids
    ]
    wins=sum(value==1 for value in changes)
    losses=sum(value==-1 for value in changes)
    return {
        "cases":len(ids),
        "baseline_accuracy":sum(int(left[k].passed) for k in ids)/len(ids),
        "candidate_accuracy":sum(int(right[k].passed) for k in ids)/len(ids),
        "accuracy_delta":statistics.mean(changes),
        "wins":wins,
        "losses":losses,
        "ties":len(ids)-wins-losses,
    }


def bootstrap_paired_delta(
    baseline:list[CaseResult],
    candidate:list[CaseResult],
    seed:int=42,
    iterations:int=1000,
    alpha:float=0.05,
)->dict:
    if not 0<alpha<1 or iterations<50:
        raise ValueError("Invalid bootstrap settings")
    left={item.identifier:item for item in baseline}
    right={item.identifier:item for item in candidate}
    if not left or set(left)!=set(right):
        raise ValueError("Case identifiers differ")
    differences=[int(right[key].passed)-int(left[key].passed) for key in sorted(left)]
    rng=random.Random(seed)
    samples=sorted(
        statistics.mean(rng.choices(differences,k=len(differences)))
        for _ in range(iterations)
    )
    lower=samples[int((alpha/2)*(iterations-1))]
    upper=samples[int((1-alpha/2)*(iterations-1))]
    return {
        "estimate":statistics.mean(differences),
        "lower":lower,
        "upper":upper,
        "alpha":alpha,
        "iterations":iterations,
        "sampling_unit":"paired case",
        "note":"Bootstrap interval assumes cases represent the target task population",
    }


def performance_gate(
    baseline:dict,
    candidate:dict,
    *,
    max_accuracy_drop:float=0.01,
    max_latency_regression:float=0.25,
    max_failure_rate:float=0.01,
)->dict:
    required=("accuracy","p95_latency_ms","failure_rate")
    for name in required:
        if name not in baseline or name not in candidate:
            raise ValueError(f"Missing metric: {name}")
    checks={
        "accuracy":candidate["accuracy"]>=baseline["accuracy"]-max_accuracy_drop,
        "latency":candidate["p95_latency_ms"]<=baseline["p95_latency_ms"]*(1+max_latency_regression),
        "reliability":candidate["failure_rate"]<=max_failure_rate,
    }
    return {
        "passed":all(checks.values()),
        "checks":checks,
        "deltas":{name:candidate[name]-baseline[name] for name in required},
        "baseline":baseline,
        "candidate":candidate,
    }


def evaluate_with_provenance(
    cases:list[BenchmarkCase],
    generator:Callable[[str],str],
    provenance:DataProvenance,
    policy:EvaluationPolicy|None=None,
    training_texts:Iterable[str]=(),
    scoring:str="normalized",
)->dict:
    provenance.validate()
    policy=policy or EvaluationPolicy()
    inspection=inspect_benchmark(cases,policy,training_texts)
    if not inspection["accepted"]:
        raise ValueError("Evaluation rejected: "+",".join(inspection["issues"]))
    rows,summary=run_cases(cases,generator,strategy=scoring)
    return {
        "provenance":provenance.display(),
        "integrity":inspection,
        "summary":summary,
        "category_accuracy":category_accuracy(rows),
        "cases":[dataclasses.asdict(row) for row in rows],
    }


def write_evaluation_report(path:Path,report:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")


def read_evaluation_report(path:Path)->dict:
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict) or "summary" not in value:
        raise ValueError("Invalid evaluation report")
    return value


def summarize_reliability(
    cases:Iterable[CaseResult],
    failure_marker:str="[GENERATION_ERROR:",
)->dict:
    values=list(cases)
    failures=[item for item in values if item.answer.startswith(failure_marker)]
    return {
        "cases":len(values),
        "failures":len(failures),
        "failure_rate":len(failures)/len(values) if values else 0.0,
        "failed_ids":[item.identifier for item in failures],
    }
