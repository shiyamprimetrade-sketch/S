"""Deterministic performance and numerical-parity benchmark harness.

Do not use these numbers as proof of intelligence. The harness reports wall
time, peak observed Python/CUDA allocations where available, cached-vs-
uncached numerical parity, and token throughput on the same model/host.
For stable results perform several warmup and measurement rounds.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from dataclasses import asdict,dataclass
from pathlib import Path

import torch

from .checkpoint import load_weights
from .kv_cache import compare_full_and_cached,cache_report,prefill,append
from .model import Config,SBrain
from .sampling import SamplingOptions
from .streaming import GenerationOptions,collect_events,stream_events


@dataclass
class BenchmarkPlan:
    prompt_bytes: int = 32
    new_tokens: int = 24
    warmup_rounds: int = 2
    measurement_rounds: int = 5
    temperature: float = 0.8
    seed: int = 101
    verify_parity: bool = True

    def validate(self,context: int):
        if not 1 <= self.prompt_bytes <= 16384:
            raise ValueError("prompt_bytes outside supported range")
        if not 1 <= self.new_tokens <= 512:
            raise ValueError("new_tokens outside supported range")
        if not 0 <= self.warmup_rounds <= 100:
            raise ValueError("warmup_rounds outside supported range")
        if not 1 <= self.measurement_rounds <= 1000:
            raise ValueError("measurement_rounds outside supported range")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if context < 2:
            raise ValueError("Context too small")


def synchronize(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def model_profile(model) -> dict:
    parameters = list(model.parameters())
    all_count = sum(value.numel() for value in parameters)
    trainable = sum(value.numel() for value in parameters if value.requires_grad)
    bytes_used = sum(value.numel()*value.element_size() for value in parameters)
    return {
        "parameters":all_count,
        "trainable_parameters":trainable,
        "parameter_bytes":bytes_used,
        "parameter_mib":bytes_used/(1024**2),
        "dtype":str(parameters[0].dtype) if parameters else None,
        "device":str(parameters[0].device) if parameters else None,
        "layers":model.cfg.layers,
        "heads":model.cfg.heads,
        "kv_heads":model.cfg.kv_heads,
        "context":model.cfg.context,
    }


def collect_distribution(samples: list[float]) -> dict:
    if not samples:
        raise ValueError("No samples recorded")
    values = sorted(samples)
    def pct(q):
        return values[min(len(values)-1,max(0,math.ceil(q*len(values))-1))]
    return {
        "count":len(values),
        "mean":statistics.mean(values),
        "median":statistics.median(values),
        "minimum":min(values),
        "maximum":max(values),
        "p90":pct(0.9),
        "p95":pct(0.95),
        "p99":pct(0.99),
        "stdev":statistics.pstdev(values),
    }


def benchmark_mode(model,prompt:str,plan:BenchmarkPlan,
                   cached:bool,device:str) -> dict:
    settings = SamplingOptions(
        max_new_tokens=plan.new_tokens,
        temperature=plan.temperature,
        seed=plan.seed,
        top_k=40,
    )
    options = GenerationOptions(sampling=settings,use_cache=cached)
    measurements = []
    all_outputs = []
    all_usage = []
    for round_idx in range(plan.warmup_rounds+plan.measurement_rounds):
        synchronize(device)
        start = time.perf_counter()
        result = collect_events(stream_events(
            model,prompt,options,device=device
        ))
        synchronize(device)
        elapsed = time.perf_counter()-start
        if round_idx < plan.warmup_rounds:
            continue
        measurements.append(elapsed)
        all_outputs.append(result["response"])
        all_usage.append(result["usage"])
    if len(set(all_outputs))>1:
        raise RuntimeError("Seeded outputs are not deterministic across rounds")
    usage = all_usage[-1]
    return {
        "use_cache":cached,
        "latency_seconds":collect_distribution(measurements),
        "tokens_generated":usage["generated_tokens"],
        "mean_tokens_per_second":(
            usage["generated_tokens"]/statistics.mean(measurements)
        ),
        "last_usage":usage,
        "output_sha256":__import__("hashlib").sha256(
            all_outputs[-1].encode("utf-8")
        ).hexdigest(),
    }


def benchmark(model,plan:BenchmarkPlan,device:str | None = None) -> dict:
    device = device or str(next(model.parameters()).device)
    plan.validate(model.cfg.context)
    prompt = "S" * plan.prompt_bytes
    model.eval()
    with torch.inference_mode():
        data = {}
        for cached in (False,True):
            name = "cached" if cached else "uncached"
            data[name] = benchmark_mode(model,prompt,plan,cached,device)
        ratio = (
            data["uncached"]["latency_seconds"]["median"] /
            data["cached"]["latency_seconds"]["median"]
        )
        report = {
            "environment":{
                "python":platform.python_version(),
                "torch":torch.__version__,
                "platform":platform.platform(),
                "cuda_available":torch.cuda.is_available(),
            },
            "model":model_profile(model),
            "plan":asdict(plan),
            "measurements":data,
            "median_uncached_over_cached_ratio":ratio,
            "interpretation":"A ratio >1 indicates cached mode was faster on this run, not universally.",
        }
        if plan.verify_parity:
            torch.manual_seed(plan.seed)
            # Cover decoding across context rollover.
            tokens = torch.randint(
                0,model.cfg.vocab_size,
                (1,min(model.cfg.context+4,plan.prompt_bytes+plan.new_tokens+4)),
                device=device
            )
            report["logit_parity"] = compare_full_and_cached(model,tokens)
        return report


def save_benchmark(path:Path,report:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8"
    )


def compare_benchmarks(before:dict,after:dict) -> dict:
    """Guard against falsely comparing different architectures or devices."""
    for key in ("parameters","context","device","dtype"):
        if before["model"][key]!=after["model"][key]:
            raise ValueError(f"Cannot compare differing model {key}")
    if before["plan"]!=after["plan"]:
        raise ValueError("Benchmark plans differ")
    result = {}
    for mode in ("cached","uncached"):
        old = before["measurements"][mode]["latency_seconds"]["median"]
        new = after["measurements"][mode]["latency_seconds"]["median"]
        result[mode] = {
            "baseline_seconds":old,
            "candidate_seconds":new,
            "speed_ratio":old/new,
            "delta_percent":100*(new/old-1),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="S-Brain cached inference benchmarks")
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--prompt-bytes",type=int,default=32)
    parser.add_argument("--tokens",type=int,default=24)
    parser.add_argument("--warmup",type=int,default=2)
    parser.add_argument("--rounds",type=int,default=5)
    parser.add_argument("--seed",type=int,default=101)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model,_ = load_weights(args.checkpoint,device)
    plan = BenchmarkPlan(
        prompt_bytes=args.prompt_bytes,
        new_tokens=args.tokens,
        warmup_rounds=args.warmup,
        measurement_rounds=args.rounds,
        seed=args.seed,
    )
    result = benchmark(model,plan,device)
    if args.output:
        save_benchmark(args.output,result)
    print(json.dumps(result,indent=2))
    if not result.get("logit_parity",{}).get("match",True):
        raise SystemExit("Numerical parity failed")


if __name__=="__main__":
    main()
