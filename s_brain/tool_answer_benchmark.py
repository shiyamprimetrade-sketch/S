"""Tool-assisted answer evaluation with independently verified tool results.

A tool computation is deterministic. That does not imply the S language model
has learned to select or use it correctly. This harness evaluates actual host-
approved tool selection, result fidelity and task-answer correctness separately.
It never infers permission from generated text. No state-mutating tools, network
calls or arbitrary commands may enter this benchmark.

The evaluator operates on intentionally frozen cases. Using a test gold value
to construct an input is legitimate only for self-authored deterministic
fixtures; it is forbidden to pass the gold target into generation.
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
from typing import Callable,Iterable

from .tools_control import ToolController,ToolPolicy
from .tool_orchestration import (
    ToolIntent,ToolAuthorization,ToolOrchestrator,
    ExecutedTool,ToolTrace,tool_success_metrics,redact_trace,
)
from .rag_evaluation import normalize_answer,exact_match,token_f1


@dataclasses.dataclass(frozen=True)
class ToolAnswerCase:
    identifier:str
    question:str
    scope:str
    tool_name:str
    arguments:dict
    expected_output:dict
    expected_answer:str
    category:str
    permission:str="self-authored"
    split:str="test"

    def validate(self):
        if not self.identifier or not self.question.strip():
            raise ValueError("Tool case ID and question required")
        if self.tool_name not in ("calculator","text_stats","json_extract"):
            raise ValueError("Benchmark must use read-only tools")
        if not self.scope or not self.category:
            raise ValueError("Scope and category required")
        if not isinstance(self.arguments,dict) or not isinstance(self.expected_output,dict):
            raise ValueError("Tool input/output must be JSON objects")
        if not self.expected_answer.strip() or not self.permission:
            raise ValueError("Expected answer and provenance required")
        if self.split!="test":
            raise ValueError("Benchmark requires held-out test cases")


@dataclasses.dataclass(frozen=True)
class ToolAnswerPolicy:
    max_cases:int=100
    min_cases:int=1
    require_per_case_approval:bool=True
    minimum_execution_rate:float=0.0
    minimum_output_fidelity:float=0.0
    minimum_answer_accuracy:float=0.0

    def validate(self):
        if not 1<=self.min_cases<=self.max_cases:
            raise ValueError("Invalid case bounds")
        for x in (
            self.minimum_execution_rate,self.minimum_output_fidelity,
            self.minimum_answer_accuracy,
        ):
            if not 0<=x<=1:
                raise ValueError("Quality requirements must be fractions")


@dataclasses.dataclass(frozen=True)
class ToolAnswerResult:
    case_hash:str
    category:str
    executed:bool
    correct_tool:bool
    output_matches:bool
    answer_exact:float
    answer_f1:float
    tool_audit_id:str|None
    answer_sha256:str

    def public(self):
        return dataclasses.asdict(self)


def canonical_json(value)->str:
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      ensure_ascii=False,allow_nan=False)


def value_matches(actual,expected)->bool:
    return canonical_json(actual)==canonical_json(expected)


def case_digest(case:ToolAnswerCase)->str:
    return hashlib.sha256(canonical_json(dataclasses.asdict(case)).encode()).hexdigest()


def fixture_controller(scope:str)->ToolController:
    controller=ToolController(
        scope=scope,
        policy=ToolPolicy(
            max_calls=1,max_batch=1,
            permitted_tools=("calculator","text_stats","json_extract"),
            allow_memory_write=False,
        ),
    )
    controller.enable()
    return controller


def execute_case(
    case:ToolAnswerCase,
    *,
    controller:ToolController|None=None,
    authorize:bool=True,
)->tuple[ToolTrace,ExecutedTool|None]:
    case.validate()
    controller=controller or fixture_controller(case.scope)
    intent=ToolIntent(
        identifier="case-"+case.identifier,name=case.tool_name,
        arguments=case.arguments,scope=case.scope,
        requested_by="application",
    )
    approval=ToolAuthorization(
        tool_name=intent.name,scope=intent.scope,
        intent_id=intent.identifier,approved_by="benchmark-host",
        enabled=True,
    ) if authorize else None
    trace=ToolOrchestrator(controller,max_intents=1).run(
        [intent],
        approvals={intent.identifier:approval} if approval else {},
    )
    return trace,trace.executed[0] if trace.executed else None


def score_tool_answer(
    case:ToolAnswerCase,
    execution:ExecutedTool|None,
    answer:str,
)->ToolAnswerResult:
    case.validate()
    if not isinstance(answer,str):
        raise ValueError("Generated answer must be text")
    return ToolAnswerResult(
        case_hash=case_digest(case),category=case.category,
        executed=execution is not None,
        correct_tool=execution is not None and execution.tool==case.tool_name,
        output_matches=bool(
            execution and execution.success
            and value_matches(execution.result,case.expected_output)
        ),
        answer_exact=exact_match(answer,case.expected_answer),
        answer_f1=token_f1(answer,case.expected_answer),
        tool_audit_id=execution.audit_id if execution else None,
        answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
    )


def evaluate_tool_answers(
    cases:list[ToolAnswerCase],
    *,
    answerer:Callable[[ToolAnswerCase,ExecutedTool|None],str],
    policy:ToolAnswerPolicy|None=None,
    authorize:bool=True,
)->dict:
    policy=policy or ToolAnswerPolicy()
    policy.validate()
    if not policy.min_cases<=len(cases)<=policy.max_cases:
        raise ValueError("Invalid benchmark case count")
    if len({case.identifier for case in cases})!=len(cases):
        raise ValueError("Duplicate case IDs")
    results=[]
    traces=[]
    for case in cases:
        case.validate()
        trace,execution=execute_case(case,authorize=authorize)
        # Only the computed tool output, never expected_answer/output,
        # should be passed to the answerer. The fixture API provides the
        # metadata case for simple use; callers must independently ensure
        # generated answers do not read reference answers.
        response=answerer(case,execution)
        results.append(score_tool_answer(case,execution,response))
        traces.append(redact_trace(trace))
    aggregate=aggregate_tool_answers(results,policy)
    return {
        "policy":dataclasses.asdict(policy),
        "benchmark_fingerprint":hashlib.sha256(
            canonical_json([dataclasses.asdict(x) for x in cases]).encode()
        ).hexdigest(),
        "summary":aggregate,
        "case_results":[row.public() for row in results],
        "tool_traces":traces,
        "read_only_tools":True,
        "model_answer_ability_proven":False,
        "warning":"Passing deterministic tool execution does not prove model tool selection",
    }


def aggregate_tool_answers(
    scores:list[ToolAnswerResult],policy:ToolAnswerPolicy,
)->dict:
    if not scores:
        raise ValueError("No scored tool results")
    execution=sum(item.executed for item in scores)/len(scores)
    output_match=sum(item.output_matches for item in scores)/len(scores)
    answer_accuracy=statistics.mean(item.answer_exact for item in scores)
    answer_f1=statistics.mean(item.answer_f1 for item in scores)
    checks={
        "execution_rate":execution>=policy.minimum_execution_rate,
        "output_fidelity":output_match>=policy.minimum_output_fidelity,
        "answer_accuracy":answer_accuracy>=policy.minimum_answer_accuracy,
    }
    groups=collections.defaultdict(list)
    for item in scores:
        groups[item.category].append(item)
    return {
        "cases":len(scores),
        "execution_rate":execution,
        "output_fidelity":output_match,
        "answer_exact":answer_accuracy,
        "answer_f1":answer_f1,
        "categories":{
            name:{
                "cases":len(values),
                "tool_success":statistics.mean(x.output_matches for x in values),
                "answer_accuracy":statistics.mean(x.answer_exact for x in values),
            }
            for name,values in sorted(groups.items())
        },
        "quality_checks":checks,
        "accepted":all(checks.values()),
    }


def compare_tool_reports(before:dict,after:dict)->dict:
    if before["benchmark_fingerprint"]!=after["benchmark_fingerprint"]:
        raise ValueError("Tool benchmarks differ")
    left,right=before["summary"],after["summary"]
    return {
        "tool_fidelity_gain":right["output_fidelity"]-left["output_fidelity"],
        "answer_accuracy_gain":right["answer_exact"]-left["answer_exact"],
        "answer_f1_gain":right["answer_f1"]-left["answer_f1"],
        "same_case_set":True,
    }


def read_tool_cases(path:Path)->list[ToolAnswerCase]:
    cases=[]
    for number,line in enumerate(path.read_text("utf-8").splitlines(),1):
        if not line.strip():
            continue
        try:
            case=ToolAnswerCase(**json.loads(line))
            case.validate()
        except (TypeError,ValueError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        cases.append(case)
    return cases


def write_tool_cases(path:Path,cases:list[ToolAnswerCase]):
    if path.exists():
        raise FileExistsError("Do not overwrite gold tool cases")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as stream:
        for case in cases:
            case.validate()
            stream.write(canonical_json(dataclasses.asdict(case))+"\n")


def write_tool_report(path:Path,report:dict):
    if path.exists():
        raise FileExistsError("Do not overwrite prior measured results")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
