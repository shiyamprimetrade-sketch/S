"""Cycle 3 batch throughput experiments and honest regression gates.

Benchmark reports use the same model, prompts, generation settings and host.
This is a benchmark harness, not evidence of a speedup until executed.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path

import torch

from .batching import BatchRequest,batch_generate,naive_reference,batch_summary
from .checkpoint import load_weights
from .evaluation_suite import performance_gate


@dataclasses.dataclass(frozen=True)
class BatchBenchmarkPlan:
    batch_sizes: tuple[int,...] = (1,2,4,8)
    prompt_bytes: int = 24
    output_tokens: int = 12
    warmups: int = 1
    repeats: int = 3
    seed: int = 42

    def validate(self):
        if not self.batch_sizes or any(size<1 or size>128 for size in self.batch_sizes):
            raise ValueError("Invalid batch sizes")
        if not 1<=self.prompt_bytes<=8192:
            raise ValueError("Invalid prompt size")
        if not 1<=self.output_tokens<=256:
            raise ValueError("Invalid output length")
        if self.warmups<0 or self.repeats<1:
            raise ValueError("Invalid benchmark repetitions")


def synchronize(device:str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def make_requests(plan:BatchBenchmarkPlan,count:int,
                  heterogeneous:bool=False)->list[BatchRequest]:
    """Seeded prompt samples. Heterogeneous mode varies prompt lengths."""
    plan.validate()
    rng=random.Random(plan.seed)
    values=[]
    for i in range(count):
        length=plan.prompt_bytes
        if heterogeneous:
            length=max(1,length-rng.randint(0,max(1,length//2)))
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        prompt="".join(rng.choice(alphabet) for _ in range(length))
        values.append(BatchRequest(
            identifier=f"sample-{i:04d}",
            prompt=prompt,
            max_new_tokens=plan.output_tokens,
            seed=plan.seed+i,
        ))
    return values


def latency_summary(values:list[float])->dict:
    if not values:
        raise ValueError("No observations")
    samples=sorted(values)
    return {
        "count":len(values),
        "mean_seconds":statistics.mean(values),
        "median_seconds":statistics.median(values),
        "min_seconds":min(values),
        "max_seconds":max(values),
        "p95_seconds":samples[math.ceil(0.95*len(samples))-1],
    }


def time_inference(model,requests,mode:str,
                   repeats:int,warmups:int,device:str)->dict:
    if mode not in ("batched","single"):
        raise ValueError("Unknown inference mode")
    execute=(
        lambda:batch_generate(model,requests,device=device)
        if mode=="batched"
        else naive_reference(model,requests,device=device)
    )
    durations=[]
    final=None
    for index in range(warmups+repeats):
        synchronize(device)
        started=time.perf_counter()
        results=execute()
        synchronize(device)
        duration=time.perf_counter()-started
        if index>=warmups:
            durations.append(duration)
            final=results
    total_tokens=sum(row.generated_tokens for row in final)
    return {
        "latency":latency_summary(durations),
        "generated_tokens":total_tokens,
        "tokens_per_second":total_tokens/statistics.mean(durations),
        "output_digest":hashlib.sha256(
            "\n".join(row.response for row in final).encode("utf-8")
        ).hexdigest(),
        "finish_reasons":[row.finish_reason for row in final],
        "total_cache_rebuilds":sum(row.cache_rebuilds for row in final),
    }


def run_benchmarks(model,plan:BatchBenchmarkPlan,
                   *,device:str|None=None,
                   heterogeneous:bool=False)->dict:
    plan.validate()
    device=device or str(next(model.parameters()).device)
    model.eval()
    comparisons=[]
    for batch_size in plan.batch_sizes:
        requests=make_requests(plan,batch_size,heterogeneous)
        mode_results={}
        for mode in ("single","batched"):
            mode_results[mode]=time_inference(
                model,requests,mode,plan.repeats,plan.warmups,device
            )
        numerator=mode_results["single"]["latency"]["median_seconds"]
        denominator=mode_results["batched"]["latency"]["median_seconds"]
        comparisons.append({
            "requests":batch_size,
            "same_outputs":(
                mode_results["single"]["output_digest"]
                ==mode_results["batched"]["output_digest"]
            ),
            "median_speed_ratio":numerator/denominator,
            "modes":mode_results,
        })
    return {
        "environment":{
            "python":platform.python_version(),
            "pytorch":torch.__version__,
            "platform":platform.platform(),
            "device":device,
        },
        "architecture":dataclasses.asdict(model.cfg),
        "plan":dataclasses.asdict(plan),
        "heterogeneous":heterogeneous,
        "comparisons":comparisons,
        "note":"Performance is hardware-dependent and not a reasoning score",
    }


def compare_batch_reports(before:dict,after:dict)->dict:
    for key in ("architecture","plan","heterogeneous"):
        if before[key]!=after[key]:
            raise ValueError(f"Benchmark mismatch: {key}")
    if before["environment"]["device"]!=after["environment"]["device"]:
        raise ValueError("Different devices cannot be directly compared")
    differences=[]
    for left,right in zip(before["comparisons"],after["comparisons"]):
        if left["requests"]!=right["requests"]:
            raise ValueError("Batch sizes differ")
        old=left["modes"]["batched"]["latency"]["median_seconds"]
        new=right["modes"]["batched"]["latency"]["median_seconds"]
        differences.append({
            "requests":left["requests"],
            "baseline_seconds":old,
            "candidate_seconds":new,
            "speed_ratio":old/new,
            "delta_seconds":new-old,
        })
    return {"compatible":True,"comparisons":differences}


def benchmark_passed(report:dict,require_parity:bool=True)->bool:
    if not report["comparisons"]:
        return False
    if require_parity and not all(item["same_outputs"] for item in report["comparisons"]):
        return False
    return True


def write_report(path:Path,value:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")


def main(argv=None):
    parser=argparse.ArgumentParser(description="Benchmark S-Brain batched inference")
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--batches",type=int,nargs="+",default=[1,2,4])
    parser.add_argument("--prompt-bytes",type=int,default=24)
    parser.add_argument("--tokens",type=int,default=12)
    parser.add_argument("--warmups",type=int,default=1)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--heterogeneous",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model,_=load_weights(args.checkpoint,device)
    plan=BatchBenchmarkPlan(
        batch_sizes=tuple(args.batches),
        prompt_bytes=args.prompt_bytes,
        output_tokens=args.tokens,
        warmups=args.warmups,repeats=args.repeats,
    )
    report=run_benchmarks(model,plan,device=device,heterogeneous=args.heterogeneous)
    if args.output:
        write_report(args.output,report)
    print(json.dumps(report,indent=2))
    if not benchmark_passed(report):
        raise SystemExit("Batched output parity check failed")


if __name__=="__main__":
    main()
