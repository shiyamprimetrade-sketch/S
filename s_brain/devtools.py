"""S-Brain development CLI: inspect, evaluate, sample and audit without editing S.

Usage:
 python -m s_brain.devtools inspect --checkpoint s_brain/checkpoints/advanced/step-00000001
 python -m s_brain.devtools sample --checkpoint ... --prompt "Hello"
 python -m s_brain.devtools evaluate --checkpoint ... --file approved_holdout.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def print_result(value: dict) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


def command_inspect(args) -> int:
    from .checkpoint import checkpoint_summary
    print_result(checkpoint_summary(args.checkpoint))
    return 0


def command_list(args) -> int:
    from .checkpoint import list_checkpoints, best_checkpoint
    rows = list_checkpoints(args.root)
    best = best_checkpoint(args.root)
    print_result({"checkpoints": rows, "best": str(best) if best else None})
    return 0


def command_compare(args) -> int:
    from .checkpoint import compare_checkpoints
    print_result(compare_checkpoints(args.baseline, args.candidate))
    return 0


def command_audit(args) -> int:
    from .dataset import audit_manifest
    report = audit_manifest(args.manifest)
    print_result(report)
    if report["duplicate_hashes"] > 0:
        return 2
    return 0


def command_sample(args) -> int:
    from .checkpoint import load_weights
    from .sampling import SamplingOptions, encode_bytes, decode_bytes, generate_tokens

    device = resolve_device(args.device)
    model, manifest = load_weights(args.checkpoint, device)
    options = SamplingOptions(
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        stop=tuple(args.stop or ()),
    )
    generated, metrics = generate_tokens(
        model, encode_bytes(args.prompt),
        options, device=device
    )
    print_result({
        "model": "s-brain:independent-experimental",
        "checkpoint_step": manifest.step,
        "prompt": args.prompt,
        "completion": decode_bytes(generated),
        "generation": metrics,
        "note": "This is a next-byte model, not an instruction-tuned assistant",
    })
    return 0


def command_evaluate(args) -> int:
    from .checkpoint import load_weights
    from .evaluate import evaluate_byte_file, save_report
    device = resolve_device(args.device)
    model, manifest = load_weights(args.checkpoint, device)
    report = evaluate_byte_file(
        model, args.file, max_bytes=args.max_bytes, device=device
    )
    report["checkpoint_step"] = manifest.step
    report["checkpoint_weights_sha256"] = manifest.weights_sha256
    if args.output:
        save_report(args.output, report)
    print_result(report)
    return 0


def command_qa(args) -> int:
    from .checkpoint import load_weights
    from .evaluate import load_cases, run_cases, save_report
    from .sampling import SamplingOptions, encode_bytes, decode_bytes, generate_tokens

    device = resolve_device(args.device)
    model, manifest = load_weights(args.checkpoint, device)
    settings = SamplingOptions(max_new_tokens=args.tokens,temperature=0,seed=42)

    def answer(prompt: str) -> str:
        ids, _ = generate_tokens(model, encode_bytes(prompt), settings, device)
        return decode_bytes(ids)

    rows, summary = run_cases(load_cases(args.cases), answer, strategy=args.strategy)
    summary["checkpoint_step"] = manifest.step
    summary["model"] = "s-brain:experimental"
    if args.output:
        save_report(args.output, summary)
    print_result(summary)
    return 0


def resolve_device(value: str) -> str:
    import torch
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA selected but unavailable")
    if value not in ("cuda", "cpu"):
        raise ValueError("Device must be auto, cuda or cpu")
    return value


def add_checkpoint_arg(parser):
    parser.add_argument("--checkpoint",type=Path,required=True)


def add_device_arg(parser):
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")


def add_generation_args(parser):
    parser.add_argument("--tokens",type=int,default=80)
    parser.add_argument("--temperature",type=float,default=0.8)
    parser.add_argument("--top-k",type=int,default=40)
    parser.add_argument("--top-p",type=float,default=0.95)
    parser.add_argument("--seed",type=int,default=42)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m s_brain.devtools",
        description="Developer commands for the independent S-Brain lab"
    )
    sub = parser.add_subparsers(dest="command",required=True)

    inspect = sub.add_parser("inspect",help="Verify checkpoint hashes and metadata")
    add_checkpoint_arg(inspect)
    inspect.set_defaults(handler=command_inspect)

    listing = sub.add_parser("list",help="List checkpoints and locate best validation loss")
    listing.add_argument("--root",type=Path,required=True)
    listing.set_defaults(handler=command_list)

    compare = sub.add_parser("compare",help="Compare checkpoints on the same dataset")
    compare.add_argument("--baseline",type=Path,required=True)
    compare.add_argument("--candidate",type=Path,required=True)
    compare.set_defaults(handler=command_compare)

    audit = sub.add_parser("audit",help="Audit corpus provenance, hashes and splits")
    audit.add_argument("--manifest",type=Path,required=True)
    audit.set_defaults(handler=command_audit)

    sample = sub.add_parser("sample",help="Generate text using only S-Brain weights")
    add_checkpoint_arg(sample)
    add_device_arg(sample)
    add_generation_args(sample)
    sample.add_argument("--prompt",required=True)
    sample.add_argument("--stop",action="append",default=[])
    sample.set_defaults(handler=command_sample)

    eval_parser = sub.add_parser("evaluate",help="Score next-token loss on held-out data")
    add_checkpoint_arg(eval_parser)
    add_device_arg(eval_parser)
    eval_parser.add_argument("--file",type=Path,required=True)
    eval_parser.add_argument("--max-bytes",type=int,default=25000)
    eval_parser.add_argument("--output",type=Path)
    eval_parser.set_defaults(handler=command_evaluate)

    qa = sub.add_parser("qa",help="Exact/normalized benchmark (not reasoning proof)")
    add_checkpoint_arg(qa)
    add_device_arg(qa)
    qa.add_argument("--cases",type=Path,required=True)
    qa.add_argument("--strategy",choices=("exact","normalized","contains"),default="normalized")
    qa.add_argument("--tokens",type=int,default=80)
    qa.add_argument("--output",type=Path)
    qa.set_defaults(handler=command_qa)

    return parser


def main(argv=None) -> int:
    args = make_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        print(f"S-Brain: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
