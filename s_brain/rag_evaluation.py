"""Paired RAG answer-evaluation harness for S-Brain.

Pairs are evaluated using the same generator, seed and memory scope.
Question/expected-answer labels must come from independently reviewed
fixtures. The evaluator reports exact-match and normalized token F1; these
are observable benchmark statistics, not evidence that a tiny model has
general reasoning or that memory is always beneficial.

Unlike an experiment that uses the gold answer to create memory, this
module only reads already approved memory and never writes it.
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

from .rag_inference import run_ablation_pair,RetrievalOptions
from .memory_store import MemoryStore,tokens
from .grounded_answers import extract_markers


@dataclasses.dataclass(frozen=True)
class AnswerCase:
    identifier:str
    question:str
    expected_answer:str
    scope:str
    category:str
    expected_source_ids:tuple[str,...]=()
    permission:str="self-authored"
    split:str="test"

    def validate(self):
        if not self.identifier or not self.question.strip():
            raise ValueError("Question and ID required")
        if not self.expected_answer.strip() or not self.category:
            raise ValueError("Gold answer and category required")
        if not self.permission:
            raise ValueError("Gold answer provenance required")
        if self.split not in ("validation","test"):
            raise ValueError("Training examples cannot be scored as held out")
        if len(set(self.expected_source_ids))!=len(self.expected_source_ids):
            raise ValueError("Duplicate gold citation IDs")


@dataclasses.dataclass(frozen=True)
class AnswerEvalPolicy:
    max_cases:int=100
    minimum_cases:int=1
    seed:int=17
    max_new_tokens:int=32
    bootstrap_iterations:int=200
    minimum_exact_gain:float=0.0
    require_source_match:bool=False
    no_answer_leakage:bool=True

    def validate(self):
        if not 1<=self.minimum_cases<=self.max_cases:
            raise ValueError("Case size policy invalid")
        if not 1<=self.max_new_tokens<=512:
            raise ValueError("Decode length invalid")
        if self.bootstrap_iterations<50:
            raise ValueError("Bootstrap repetitions too small")
        if not -1<=self.minimum_exact_gain<=1:
            raise ValueError("Exact gain outside range")


@dataclasses.dataclass
class PairedAnswerScore:
    identifier:str
    category:str
    question_hash:str
    no_memory_exact:float
    memory_exact:float
    no_memory_f1:float
    memory_f1:float
    cited_precision:float
    evidence_retrieved:int
    answer_changed:bool
    no_memory_response_hash:str
    memory_response_hash:str

    @property
    def exact_gain(self):
        return self.memory_exact-self.no_memory_exact

    @property
    def f1_gain(self):
        return self.memory_f1-self.no_memory_f1

    def report(self):
        return {
            **dataclasses.asdict(self),
            "exact_gain":self.exact_gain,
            "f1_gain":self.f1_gain,
        }


def normalize_answer(text:str)->str:
    text=re.sub(r"\[mem:[0-9a-fA-F-]{36}\]","",text)
    return " ".join(re.findall(r"\w+",text.casefold(),re.UNICODE))


def exact_match(prediction:str,gold:str)->float:
    return float(normalize_answer(prediction)==normalize_answer(gold))


def token_f1(prediction:str,gold:str)->float:
    left=tokens(normalize_answer(prediction))
    right=tokens(normalize_answer(gold))
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection=sum((collections.Counter(left)&collections.Counter(right)).values())
    if not intersection:
        return 0.0
    precision=intersection/len(left)
    recall=intersection/len(right)
    return 2*precision*recall/(precision+recall)


def citation_precision(prediction:str,gold_ids:tuple[str,...])->float:
    cited=set(extract_markers(prediction))
    expected=set(gold_ids)
    if not cited:
        return 1.0 if not expected else 0.0
    return len(cited&expected)/len(cited)


def sha(text:str)->str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def score_ablation_case(
    case:AnswerCase,ablation:dict,
)->PairedAnswerScore:
    case.validate()
    if sha(case.question)!=ablation["question_sha256"]:
        raise ValueError("Paired output belongs to another question")
    original=ablation["without_memory"]
    treatment=ablation["with_memory"]
    if original["question_sha256"]!=treatment["question_sha256"]:
        raise ValueError("Mismatched ablation questions")
    no_memory=original["answer"]
    with_memory=treatment["answer"]
    return PairedAnswerScore(
        identifier=case.identifier,category=case.category,
        question_hash=sha(case.question),
        no_memory_exact=exact_match(no_memory,case.expected_answer),
        memory_exact=exact_match(with_memory,case.expected_answer),
        no_memory_f1=token_f1(no_memory,case.expected_answer),
        memory_f1=token_f1(with_memory,case.expected_answer),
        cited_precision=citation_precision(with_memory,case.expected_source_ids),
        evidence_retrieved=treatment["evidence"]["included"],
        answer_changed=no_memory!=with_memory,
        no_memory_response_hash=sha(no_memory),
        memory_response_hash=sha(with_memory),
    )


def paired_confidence_interval(
    differences:list[float],
    *,
    seed:int=17,iterations:int=200,confidence:float=0.95,
)->dict:
    if not differences or iterations<50 or not 0<confidence<1:
        raise ValueError("Invalid bootstrap interval")
    rng=random.Random(seed)
    values=sorted(
        statistics.mean(rng.choices(differences,k=len(differences)))
        for _ in range(iterations)
    )
    alpha=1-confidence
    low=int(alpha/2*(iterations-1))
    high=int((1-alpha/2)*(iterations-1))
    return {
        "mean":statistics.mean(differences),
        "lower":values[low],
        "upper":values[high],
        "method":"paired_record_bootstrap",
        "repetitions":iterations,
    }


def aggregate_paired_scores(
    scores:list[PairedAnswerScore],policy:AnswerEvalPolicy,
)->dict:
    policy.validate()
    if len(scores)<policy.minimum_cases:
        raise ValueError("Insufficient independently labeled cases")
    exact=[item.exact_gain for item in scores]
    f1=[item.f1_gain for item in scores]
    categories=collections.defaultdict(list)
    for row in scores:
        categories[row.category].append(row)
    category_results={}
    for category,items in sorted(categories.items()):
        category_results[category]={
            "cases":len(items),
            "exact_gain":statistics.mean(x.exact_gain for x in items),
            "f1_gain":statistics.mean(x.f1_gain for x in items),
            "memory_exact":statistics.mean(x.memory_exact for x in items),
            "without_memory_exact":statistics.mean(x.no_memory_exact for x in items),
        }
    return {
        "cases":len(scores),
        "memory_exact":statistics.mean(x.memory_exact for x in scores),
        "without_memory_exact":statistics.mean(x.no_memory_exact for x in scores),
        "memory_f1":statistics.mean(x.memory_f1 for x in scores),
        "without_memory_f1":statistics.mean(x.no_memory_f1 for x in scores),
        "mean_exact_gain":statistics.mean(exact),
        "mean_f1_gain":statistics.mean(f1),
        "exact_confidence_interval":paired_confidence_interval(
            exact,seed=policy.seed,iterations=policy.bootstrap_iterations
        ),
        "f1_confidence_interval":paired_confidence_interval(
            f1,seed=policy.seed,iterations=policy.bootstrap_iterations
        ),
        "categories":category_results,
        "better_cases":sum(value>0 for value in exact),
        "worse_cases":sum(value<0 for value in exact),
        "tied_cases":sum(value==0 for value in exact),
        "passes_declared_exact_gain":(
            statistics.mean(exact)>=policy.minimum_exact_gain
        ),
        "model_quality_proven":False,
        "note":"Task-specific exact-match/F1; blind test integrity depends on independent labeling",
    }


def run_answer_evaluation(
    model,store:MemoryStore,cases:list[AnswerCase],
    *,
    generator:Callable|None=None,
    policy:AnswerEvalPolicy|None=None,
)->dict:
    policy=policy or AnswerEvalPolicy()
    policy.validate()
    if not policy.minimum_cases<=len(cases)<=policy.max_cases:
        raise ValueError("Case count outside policy")
    ids=[case.identifier for case in cases]
    if len(set(ids))!=len(ids):
        raise ValueError("Duplicate answer benchmark IDs")
    scores=[]
    for case in cases:
        case.validate()
        result=run_ablation_pair(
            model,case.question,store,
            scope=case.scope,seed=policy.seed,
            max_new_tokens=policy.max_new_tokens,
            generator=generator,
        )
        scores.append(score_ablation_case(case,result))
    summary=aggregate_paired_scores(scores,policy)
    return {
        "policy":dataclasses.asdict(policy),
        "summary":summary,
        "case_results":[x.report() for x in scores],
        "question_set_sha256":sha(json.dumps(
            [dataclasses.asdict(x) for x in cases],sort_keys=True
        )),
        "same_generator_and_seed":True,
        "gold_data_used_only_for_scoring":True,
    }


def compare_evaluations(before:dict,after:dict)->dict:
    if before["question_set_sha256"]!=after["question_set_sha256"]:
        raise ValueError("Cannot compare different labeled cases")
    if before["policy"]!=after["policy"]:
        raise ValueError("Evaluation policy mismatch")
    left=before["summary"]
    right=after["summary"]
    return {
        "memory_f1_change":right["memory_f1"]-left["memory_f1"],
        "memory_exact_change":right["memory_exact"]-left["memory_exact"],
        "ablation_gain_change":right["mean_exact_gain"]-left["mean_exact_gain"],
        "quality_claim":"descriptive_only",
    }


def read_cases(path:Path)->list[AnswerCase]:
    result=[]
    with path.open("r",encoding="utf-8") as stream:
        for number,line in enumerate(stream,1):
            if not line.strip():
                continue
            try:
                value=json.loads(line)
                value["expected_source_ids"]=tuple(value.get("expected_source_ids",()))
                item=AnswerCase(**value)
                item.validate()
            except (TypeError,ValueError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            result.append(item)
    return result


def write_cases(path:Path,cases:list[AnswerCase]):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError("Refusing to overwrite existing gold labels")
    with path.open("w",encoding="utf-8") as stream:
        for case in cases:
            case.validate()
            stream.write(json.dumps(dataclasses.asdict(case),ensure_ascii=False)+"\n")


def write_evaluation(path:Path,report:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError("Evaluation report already exists")
    path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
