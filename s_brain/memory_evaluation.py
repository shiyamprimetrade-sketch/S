"""S-Memory evaluation: retrieval relevance, ablation and citation integrity.

This harness evaluates memory search on manually labeled relevance IDs
within one project scope. It does not claim that RAG improves language-model
reasoning; model-side evaluation is separately required. A leakage probe
tests that private records in another scope are never exposed to searches.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import statistics
from pathlib import Path

from .memory_store import MemoryStore, RetrievalHit, memory_quality


@dataclasses.dataclass(frozen=True)
class RetrievalCase:
    identifier: str
    scope: str
    query: str
    expected_ids: tuple[str, ...]
    category: str = "general"

    def validate(self):
        if not self.identifier or not self.scope or not self.query.strip():
            raise ValueError("Retrieval case requires ID, scope and question")
        if len(set(self.expected_ids))!=len(self.expected_ids):
            raise ValueError("Duplicate expected memory IDs")


@dataclasses.dataclass(frozen=True)
class RetrievalPolicy:
    k: int = 5
    minimum_cases: int = 1
    minimum_recall: float = 0.0
    minimum_precision: float = 0.0
    forbid_unlabeled_results: bool = False

    def validate(self):
        if self.k<1 or self.minimum_cases<1:
            raise ValueError("Invalid retrieval policy")
        if not 0<=self.minimum_recall<=1 or not 0<=self.minimum_precision<=1:
            raise ValueError("Invalid quality thresholds")


@dataclasses.dataclass
class RetrievalCaseResult:
    identifier: str
    scope: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    citation_complete: bool
    evidence_hashes: list[str]

    def as_dict(self):
        return dataclasses.asdict(self)


def evaluate_case(store:MemoryStore,case:RetrievalCase,policy:RetrievalPolicy):
    case.validate()
    policy.validate()
    hits=store.retrieve(case.scope,case.query,limit=policy.k)
    ids=[hit.record.id for hit in hits]
    matches=set(case.expected_ids)&set(ids)
    precision=len(matches)/max(1,len(ids))
    recall=len(matches)/max(1,len(case.expected_ids))
    citation_complete=all(
        len(hit.record.content_sha256)==64
        and len(hit.record.source_sha256)==64
        and hit.record.scope==case.scope
        for hit in hits
    )
    return RetrievalCaseResult(
        identifier=case.identifier,scope=case.scope,
        retrieved_ids=ids,relevant_ids=list(case.expected_ids),
        precision_at_k=precision,recall_at_k=recall,
        citation_complete=citation_complete,
        evidence_hashes=[hit.record.content_sha256 for hit in hits],
    )


def evaluate_retrieval(
    store:MemoryStore,cases:list[RetrievalCase],
    policy:RetrievalPolicy|None=None,
)->dict:
    policy=policy or RetrievalPolicy()
    policy.validate()
    if len(cases)<policy.minimum_cases:
        raise ValueError("Not enough labeled queries")
    if len({x.identifier for x in cases})!=len(cases):
        raise ValueError("Duplicate retrieval case identifiers")
    results=[evaluate_case(store,item,policy) for item in cases]
    precision=statistics.mean(x.precision_at_k for x in results)
    recall=statistics.mean(x.recall_at_k for x in results)
    citations=all(x.citation_complete for x in results)
    checks={
        "precision":precision>=policy.minimum_precision,
        "recall":recall>=policy.minimum_recall,
        "citations":citations,
    }
    if policy.forbid_unlabeled_results:
        checks["no_unlabeled_results"]=all(
            set(row.retrieved_ids)<=set(row.relevant_ids)
            for row in results
        )
    return {
        "cases":len(results),
        "mean_precision_at_k":precision,
        "mean_recall_at_k":recall,
        "citations_complete":citations,
        "checks":checks,
        "accepted":all(checks.values()),
        "policy":dataclasses.asdict(policy),
        "results":[x.as_dict() for x in results],
        "note":"Labeled lexical retrieval, not proof of correct LLM answers",
    }


def cross_scope_probe(
    store:MemoryStore,allowed_scope:str,other_scope:str,
    query:str,
)->dict:
    if allowed_scope==other_scope:
        raise ValueError("Probe needs distinct scopes")
    private_ids={row.id for row in store.list_scope(other_scope)}
    results=store.retrieve(allowed_scope,query,limit=20)
    exposed=[hit.record.id for hit in results if hit.record.id in private_ids]
    return {
        "isolated":not exposed,
        "attempted_scope":allowed_scope,
        "private_scope_hash":hashlib.sha256(other_scope.encode()).hexdigest(),
        "private_records_exposed":len(exposed),
        "no_raw_private_content":True,
    }


def compare_retrieval(before:dict,after:dict)->dict:
    if before["policy"]!=after["policy"]:
        raise ValueError("Comparisons require identical retrieval protocols")
    if {x["identifier"] for x in before["results"]}!={
        x["identifier"] for x in after["results"]
    }:
        raise ValueError("Retrieval case IDs do not match")
    return {
        "precision_gain":after["mean_precision_at_k"]-before["mean_precision_at_k"],
        "recall_gain":after["mean_recall_at_k"]-before["mean_recall_at_k"],
        "citation_coverage_maintained":after["citations_complete"],
        "tasks":before["cases"],
    }


def labeled_query_digest(cases:list[RetrievalCase])->str:
    content=[
        dataclasses.asdict(case) for case in sorted(cases,key=lambda x:x.identifier)
    ]
    return hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()


def write_report(path:Path,report:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError("Evaluation report already exists")
    path.write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8"
    )
