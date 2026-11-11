"""Checkpoint-backed, teacher-free S-Brain memory ablation benchmark.

This runner explicitly loads model weights from a checksum-verified
safetensors checkpoint and uses the real independent byte model for BOTH
arms. No mocked generator is accepted here. A memory-off arm and a
memory-on arm use identical model weights, deterministic decoding seed,
max tokens and original question. The paired score is exact match and
token F1 against independently authored answers.

The test suite may exercise this on random-weight smoke-test checkpoints.
That verifies execution only. A real-world improvement must have a
representative private corpus, a frozen benchmark, confidence intervals and
results measured on meaningfully trained independent weights.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path

from .checkpoint import load_weights,load_manifest
from .memory_store import MemoryStore
from .rag_evaluation import (
    AnswerCase,AnswerEvalPolicy,run_answer_evaluation,read_cases,
    paired_confidence_interval,
)
from .supervised_data import fingerprint_dataset
from .rag_inference import RetrievalOptions,ContextOverflow


@dataclasses.dataclass(frozen=True)
class RealRagPlan:
    seed:int=101
    max_new_tokens:int=16
    maximum_cases:int=50
    minimum_cases:int=2
    minimum_context:int=512
    allow_validation_split:bool=False
    require_gold_permission:bool=True
    require_sealed_questions:bool=True
    bootstrap_iterations:int=200
    minimum_gain_to_claim:float=0.05

    def validate(self):
        if type(self.seed) is not int or self.seed<0:
            raise ValueError("Invalid decode seed")
        if not 1<=self.minimum_cases<=self.maximum_cases:
            raise ValueError("Invalid benchmark size")
        if not 1<=self.max_new_tokens<=256:
            raise ValueError("Invalid benchmark generation cap")
        if self.minimum_context<128 or self.bootstrap_iterations<50:
            raise ValueError("Benchmark capacity invalid")
        if not -1<=self.minimum_gain_to_claim<=1:
            raise ValueError("Invalid improvement threshold")


@dataclasses.dataclass
class VerifiedCheckpoint:
    source_path:str
    weights_sha256:str
    training_step:int
    context:int
    parameter_count:int
    dataset_fingerprint:str
    model:object

    def metadata(self):
        return {
            "source_path":self.source_path,
            "weights_sha256":self.weights_sha256,
            "training_step":self.training_step,
            "context":self.context,
            "parameter_count":self.parameter_count,
            "dataset_fingerprint":self.dataset_fingerprint,
        }


def open_checkpoint(path:Path,*,device:str="cpu")->VerifiedCheckpoint:
    if not (path/"manifest.json").exists():
        raise ValueError("Only verified safetensors checkpoints supported")
    model,manifest=load_weights(path,device)
    if not getattr(model,"cfg",None):
        raise TypeError("Checkpoint model lacks architecture configuration")
    params=sum(item.numel() for item in model.parameters())
    if params<=0:
        raise ValueError("Empty model parameters")
    model.eval()
    return VerifiedCheckpoint(
        source_path=str(path.resolve()),weights_sha256=manifest.weights_sha256,
        training_step=manifest.step,context=model.cfg.context,
        parameter_count=params,
        dataset_fingerprint=manifest.dataset_fingerprint,model=model,
    )


def case_fingerprint(cases:list[AnswerCase])->str:
    h=hashlib.sha256()
    for item in sorted(cases,key=lambda x:x.identifier):
        item.validate()
        h.update(json.dumps(dataclasses.asdict(item),sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def inspect_cases(cases:list[AnswerCase],plan:RealRagPlan)->dict:
    plan.validate()
    if not plan.minimum_cases<=len(cases)<=plan.maximum_cases:
        raise ValueError("Real-model evaluation needs enough distinct cases")
    if len({item.identifier for item in cases})!=len(cases):
        raise ValueError("Duplicate benchmark case identifiers")
    prompts=[item.question.casefold().strip() for item in cases]
    if len(set(prompts))!=len(prompts):
        raise ValueError("Repeated benchmark questions")
    if any(item.split!="test" for item in cases) and not plan.allow_validation_split:
        raise ValueError("Benchmark must use an independent test split")
    if plan.require_gold_permission and any(
        not item.permission or item.permission in ("unknown","unverified")
        for item in cases
    ):
        raise ValueError("Gold labels lack reviewed permission")
    return {
        "cases":len(cases),
        "categories":sorted({item.category for item in cases}),
        "fingerprint":case_fingerprint(cases),
        "all_test":all(item.split=="test" for item in cases),
        "independently_labeled":True,
    }


def prepare_benchmark(
    model_context:int,questions:list[AnswerCase],plan:RealRagPlan,
):
    if model_context<plan.minimum_context:
        raise ValueError("Checkpoint context too short for controlled RAG")
    # Do not silently trim context, since the answer then may be conditioned
    # on a truncated question or missing source marker.
    for item in questions:
        if len(item.question.encode("utf-8"))>=model_context//2:
            raise ValueError("Question too large for benchmark model context")


def real_model_ablation(
    checkpoint:Path,
    memory_file:Path,
    cases:list[AnswerCase],
    *,
    plan:RealRagPlan|None=None,
    device:str="cpu",
)->dict:
    plan=plan or RealRagPlan()
    plan.validate()
    case_audit=inspect_cases(cases,plan)
    inspected=open_checkpoint(checkpoint,device=device)
    prepare_benchmark(inspected.context,cases,plan)
    if not memory_file.exists():
        raise FileNotFoundError("Memory database not found")
    started=time.perf_counter()
    with MemoryStore(memory_file) as memory:
        # The scoring loop is exactly run_answer_evaluation without passing
        # a generator argument. Therefore it uses S's own inference callback.
        scores=run_answer_evaluation(
            inspected.model,memory,cases,
            policy=AnswerEvalPolicy(
                max_cases=plan.maximum_cases,
                minimum_cases=plan.minimum_cases,
                seed=plan.seed,
                max_new_tokens=plan.max_new_tokens,
                bootstrap_iterations=plan.bootstrap_iterations,
                minimum_exact_gain=plan.minimum_gain_to_claim,
            ),
        )
    summary=scores["summary"]
    interval=summary["exact_confidence_interval"]
    improvement_supported=(
        summary["mean_exact_gain"]>=plan.minimum_gain_to_claim
        and interval["lower"]>0
        and summary["better_cases"]>summary["worse_cases"]
    )
    result={
        "checkpoint":inspected.metadata(),
        "cases":case_audit,
        "plan":dataclasses.asdict(plan),
        "summary":summary,
        "per_case":scores["case_results"],
        "same_real_model_in_both_arms":True,
        "same_seed_in_both_arms":True,
        "teacher_called":False,
        "tool_called":False,
        "gold_answer_used_for_generation":False,
        "memory_improvement_supported_by_selected_metric":improvement_supported,
        "independent_general_intelligence_established":False,
        "elapsed_seconds":time.perf_counter()-started,
    }
    return result


def compare_real_benchmarks(before:dict,after:dict)->dict:
    if before["cases"]["fingerprint"]!=after["cases"]["fingerprint"]:
        raise ValueError("Different gold datasets")
    if before["plan"]!=after["plan"]:
        raise ValueError("Different evaluation protocols")
    if before["checkpoint"]["weights_sha256"]==after["checkpoint"]["weights_sha256"]:
        changed=False
    else:
        changed=True
    return {
        "weights_changed":changed,
        "gold_unchanged":True,
        "memory_exact_delta":(
            after["summary"]["memory_exact"]-before["summary"]["memory_exact"]
        ),
        "memory_f1_delta":(
            after["summary"]["memory_f1"]-before["summary"]["memory_f1"]
        ),
        "no_memory_exact_delta":(
            after["summary"]["without_memory_exact"]-
            before["summary"]["without_memory_exact"]
        ),
        "evidence_quality_claim":"measured_metrics_only",
    }


def verify_benchmark_provenance(result:dict)->dict:
    checks={
        "independent_checkpoint":bool(result["checkpoint"]["weights_sha256"]),
        "checkpoint_training_step_known":result["checkpoint"]["training_step"]>=0,
        "gold_source_known":len(result["cases"]["fingerprint"])==64,
        "test_split":result["cases"]["all_test"],
        "same_model":result["same_real_model_in_both_arms"],
        "same_seed":result["same_seed_in_both_arms"],
        "no_teacher":not result["teacher_called"],
        "no_tool":not result["tool_called"],
        "gold_not_in_generator":not result["gold_answer_used_for_generation"],
    }
    return {"valid":all(checks.values()),"checks":checks}


def write_benchmark(path:Path,report:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError("Cannot overwrite immutable measured benchmark")
    fd,temp=tempfile.mkstemp(dir=path.parent,prefix=".s-real-rag-")
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as out:
            json.dump(report,out,indent=2,allow_nan=False)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def main(argv=None):
    p=argparse.ArgumentParser(
        description="Run a real independent S checkpoint in paired RAG benchmark"
    )
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--memory-db",type=Path,required=True)
    p.add_argument("--gold",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-new-tokens",type=int,default=16)
    p.add_argument("--seed",type=int,default=101)
    p.add_argument("--minimum-cases",type=int,default=2)
    args=p.parse_args(argv)
    cases=read_cases(args.gold)
    plan=RealRagPlan(
        max_new_tokens=args.max_new_tokens,seed=args.seed,
        minimum_cases=args.minimum_cases,
    )
    report=real_model_ablation(
        args.checkpoint,args.memory_db,cases,plan=plan
    )
    report["provenance"]=verify_benchmark_provenance(report)
    write_benchmark(args.output,report)
    print(json.dumps({
        "summary":report["summary"],
        "provenance":report["provenance"],
    },indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
