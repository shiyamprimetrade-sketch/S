"""Measured supervised fine-tuning of S-Brain's own neural weights.

The input format is human-reviewed, permitted teacher-response text, followed
by independently authored holdout records. Train and validation questions are
required to have disjoint prompt hashes. Each optimizer update is measured and
guarded against nonfinite loss/gradients. Best validation weights are written
as a self-describing checkpoint; the baseline is saved before any training.

S-Brain model weights ARE modified by AdamW backpropagation. Ollama is never
called here and is not part of inference. This is text-response distillation,
NOT logit-level KL distillation from a teacher neural network.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable

import torch

from .checkpoint import save_checkpoint
from .model import Config, SBrain
from .supervised_data import (
    SupervisedRecord, WindowSampler, build_windows, detect_train_test_leakage,
    fingerprint_dataset, import_jsonl, masked_cross_entropy,
)


@dataclasses.dataclass(frozen=True)
class SupervisedTrainConfig:
    steps: int = 100
    batch_size: int = 4
    context: int = 64
    dim: int = 128
    layers: int = 2
    heads: int = 4
    kv_heads: int = 2
    lr: float = 4e-4
    min_lr: float = 4e-5
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    gradient_accumulation: int = 2
    validate_every: int = 10
    validation_batches: int = 8
    seed: int = 42
    device: str = "auto"
    minimum_train_windows: int = 4
    minimum_holdout_windows: int = 4

    def validate(self):
        integers = [
            "steps","batch_size","context","dim","layers","heads",
            "kv_heads","gradient_accumulation","validate_every",
            "validation_batches","minimum_train_windows","minimum_holdout_windows",
        ]
        if any(getattr(self,name) < 1 for name in integers):
            raise ValueError("All count settings must be positive")
        if self.context < 8:
            raise ValueError("Context must be at least 8")
        if self.dim % self.heads or self.heads % self.kv_heads:
            raise ValueError("Incompatible head dimensions")
        if (self.dim//self.heads) % 2:
            raise ValueError("RoPE head dimension must be even")
        if not 0 < self.min_lr <= self.lr or not math.isfinite(self.lr):
            raise ValueError("Learning rate range invalid")
        if self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid regularization settings")
        if self.device not in ("cpu","cuda","auto"):
            raise ValueError("Unsupported training device")

    def model_config(self):
        return Config(
            vocab_size=256,dim=self.dim,layers=self.layers,
            heads=self.heads,kv_heads=self.kv_heads,context=self.context
        )


@dataclasses.dataclass
class TrainMeasurement:
    step: int
    train_loss: float
    validation_loss: float | None
    gradient_norm: float
    parameter_update_l2: float
    supervised_tokens: int
    elapsed_seconds: float
    lr: float
    best_validation_loss: float

    def as_dict(self):
        return dataclasses.asdict(self)


def model_weights(model):
    return {
        name: parameter.detach().clone()
        for name,parameter in model.named_parameters()
        if parameter.requires_grad
    }


def weight_update_l2(before,model):
    squared=0.0
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:
            delta=(parameter.detach().float()-before[name].float())
            squared+=float(delta.square().sum())
    return math.sqrt(squared)


def set_reproducibility(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested:str):
    if requested=="auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested=="cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return requested


def cosine_lr(step:int,total:int,high:float,low:float):
    if total < 1 or not 0 <= step <= total:
        raise ValueError("Invalid learning schedule position")
    fraction=step/total
    return low+0.5*(high-low)*(1+math.cos(math.pi*fraction))


def check_split_integrity(
    train_records:list[SupervisedRecord],
    holdout_records:list[SupervisedRecord],
):
    if not train_records or not holdout_records:
        raise ValueError("Need both training and independent validation data")
    if any(record.split!="train" for record in train_records):
        raise ValueError("Train dataset contains nontrain records")
    if any(record.split not in ("validation","test") for record in holdout_records):
        raise ValueError("Holdout dataset contains training records")
    combined=train_records+holdout_records
    overlap=detect_train_test_leakage(combined)
    if overlap:
        raise ValueError("Holdout question appears in training")
    content_hashes={record.source_hash for record in train_records}
    if any(record.source_hash in content_hashes for record in holdout_records):
        raise ValueError("Holdout answer content overlaps training source hashes")
    return {
        "train_records":len(train_records),
        "holdout_records":len(holdout_records),
        "train_fingerprint":fingerprint_dataset(train_records),
        "holdout_fingerprint":fingerprint_dataset(holdout_records),
        "question_overlap":0,
    }


@torch.inference_mode()
def eval_windows(model,windows,config:SupervisedTrainConfig,
                 *,seed:int=100)->dict:
    model.eval()
    sampler=WindowSampler(windows,config.context,seed=seed)
    losses=[]
    tokens=0
    device=str(next(model.parameters()).device)
    for _ in range(config.validation_batches):
        x,y,count=sampler.sample(config.batch_size,device=device)
        logits,_=model(x)
        loss=masked_cross_entropy(logits,y)
        losses.append(float(loss))
        tokens+=count
    mean=sum(losses)/len(losses)
    return {
        "cross_entropy":mean,
        "perplexity":math.exp(mean) if mean<700 else float("inf"),
        "supervised_tokens":tokens,
        "batches":config.validation_batches,
    }


def optimizer_step(model,optimizer,sampler:WindowSampler,
                   config:SupervisedTrainConfig,lr:float)->dict:
    model.train()
    for group in optimizer.param_groups:
        group["lr"]=lr
    optimizer.zero_grad(set_to_none=True)
    previous=model_weights(model)
    tokens=0
    mean_loss=0.0
    for _ in range(config.gradient_accumulation):
        x,y,count=sampler.sample(config.batch_size,device=str(next(model.parameters()).device))
        logits,_=model(x)
        loss=masked_cross_entropy(logits,y)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        (loss/config.gradient_accumulation).backward()
        mean_loss+=float(loss)/config.gradient_accumulation
        tokens+=count
    gradient=torch.nn.utils.clip_grad_norm_(
        model.parameters(),config.gradient_clip
    )
    if not torch.isfinite(gradient):
        raise RuntimeError("Nonfinite gradient; refusing weight update")
    optimizer.step()
    change=weight_update_l2(previous,model)
    if change<=0 or not math.isfinite(change):
        raise RuntimeError("Optimizer did not measurably change weights")
    return {
        "loss":mean_loss,
        "gradient_norm":float(gradient),
        "weight_change_l2":change,
        "supervised_tokens":tokens,
    }


def _write_json(path:Path,obj:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8")


def _append_jsonl(path:Path,obj:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as handle:
        handle.write(json.dumps(obj,allow_nan=False)+"\n")


def run_supervised_training(
    train_records:list[SupervisedRecord],
    holdout_records:list[SupervisedRecord],
    output:Path,
    *,
    config:SupervisedTrainConfig|None=None,
)->dict:
    config=config or SupervisedTrainConfig()
    config.validate()
    integrity=check_split_integrity(train_records,holdout_records)
    train_windows=build_windows(train_records,config.context,method="dense")
    validation_windows=build_windows(holdout_records,config.context,method="dense")
    if len(train_windows)<config.minimum_train_windows:
        raise ValueError("Not enough supervised train windows")
    if len(validation_windows)<config.minimum_holdout_windows:
        raise ValueError("Not enough supervised holdout windows")
    set_reproducibility(config.seed)
    device=resolve_device(config.device)
    model=SBrain(config.model_config()).to(device)
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=config.lr,weight_decay=config.weight_decay
    )
    sampler=WindowSampler(train_windows,config.context,seed=config.seed+1)
    baseline=eval_windows(model,validation_windows,config,seed=config.seed+7)
    output.mkdir(parents=True,exist_ok=True)
    first_dataset=integrity["train_fingerprint"]+":"+integrity["holdout_fingerprint"]
    dataset_hash=hashlib.sha256(first_dataset.encode("utf-8")).hexdigest()
    beginning=time.perf_counter()
    save_checkpoint(
        output/"baseline",model,model.cfg,
        step=0,train_tokens=0,train_loss=baseline["cross_entropy"],
        validation_loss=baseline["cross_entropy"],
        seed=config.seed,dataset_fingerprint=dataset_hash,
        extras={"phase":"baseline","objective":"answer-only cross-entropy"},
    )
    best=baseline["cross_entropy"]
    best_step=0
    measured_tokens=0
    records=[]
    for step in range(1,config.steps+1):
        lr=cosine_lr(step-1,config.steps,config.lr,config.min_lr)
        step_info=optimizer_step(model,optimizer,sampler,config,lr)
        measured_tokens+=step_info["supervised_tokens"]
        validation=None
        if step%config.validate_every==0 or step==config.steps:
            # Reset seed for every checkpoint to compare fixed validation samples.
            validation=eval_windows(
                model,validation_windows,config,seed=config.seed+7
            )
            if validation["cross_entropy"]<best:
                best=validation["cross_entropy"]
                best_step=step
                save_checkpoint(
                    output/"best",model,model.cfg,
                    step=step,train_tokens=measured_tokens,
                    train_loss=step_info["loss"],
                    validation_loss=best,
                    seed=config.seed,dataset_fingerprint=dataset_hash,
                    extras={"phase":"best","objective":"answer-only cross-entropy"},
                )
        item=TrainMeasurement(
            step=step,train_loss=step_info["loss"],
            validation_loss=None if validation is None else validation["cross_entropy"],
            gradient_norm=step_info["gradient_norm"],
            parameter_update_l2=step_info["weight_change_l2"],
            supervised_tokens=step_info["supervised_tokens"],
            elapsed_seconds=time.perf_counter()-beginning,lr=lr,
            best_validation_loss=best,
        )
        records.append(item.as_dict())
        _append_jsonl(output/"metrics.jsonl",item.as_dict())
    final=eval_windows(model,validation_windows,config,seed=config.seed+7)
    save_checkpoint(
        output/"final",model,model.cfg,
        step=config.steps,train_tokens=measured_tokens,
        train_loss=records[-1]["train_loss"],
        validation_loss=final["cross_entropy"],
        seed=config.seed,dataset_fingerprint=dataset_hash,
        extras={"phase":"final","objective":"answer-only cross-entropy"},
    )
    report={
        "configuration":dataclasses.asdict(config),
        "integrity":integrity,
        "dataset_fingerprint":dataset_hash,
        "train_windows":len(train_windows),
        "validation_windows":len(validation_windows),
        "baseline_validation_loss":baseline["cross_entropy"],
        "best_validation_loss":best,
        "best_step":best_step,
        "final_validation_loss":final["cross_entropy"],
        "improvement_over_baseline":baseline["cross_entropy"]-best,
        "optimizer_steps":config.steps,
        "measured_supervised_tokens":measured_tokens,
        "weights_changed_each_step":all(
            item["parameter_update_l2"]>0 for item in records
        ),
        "best_model_saved":best_step>0,
        "independent_inference":True,
        "teacher_required_at_inference":False,
        "quality_claim":"experimental; no general intelligence established",
    }
    _write_json(output/"report.json",report)
    return report


def load_training_pair(train:Path,holdout:Path):
    rows=import_jsonl(train)
    validation=import_jsonl(holdout)
    return rows,validation


def main(argv=None):
    parser=argparse.ArgumentParser(
        description="Train independent S weights with reviewed response supervision"
    )
    parser.add_argument("--train",type=Path,required=True)
    parser.add_argument("--holdout",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=100)
    parser.add_argument("--context",type=int,default=64)
    parser.add_argument("--dim",type=int,default=128)
    parser.add_argument("--layers",type=int,default=2)
    parser.add_argument("--batch",type=int,default=4)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    args=parser.parse_args(argv)
    train,validation=load_training_pair(args.train,args.holdout)
    config=SupervisedTrainConfig(
        steps=args.steps,context=args.context,dim=args.dim,
        layers=args.layers,batch_size=args.batch,device=args.device,
    )
    result=run_supervised_training(train,validation,args.output,config=config)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
