"""Exact-length batched prefill and incremental decoding for S-Brain.

Padding positional encodings and causal masks incorrectly changes predictions.
This first batch implementation groups prompts by their actual byte length so
all rows in each group have equal active positions. Groups can be processed
independently without a special padding mask, while sharing a single SDPA call.

The batch deliberately stores one KVState per length group. Context rollover
rebuilds the active window, as in the verified single-sequence cache.
"""
from __future__ import annotations

import collections
import dataclasses
import math
import threading
import time
from typing import Iterable, Iterator

import torch

from .kv_cache import KVState, append, prefill
from .sampling import SamplingOptions, generator_for, select_token
from .streaming import UTF8Accumulator, StopMatcher


@dataclasses.dataclass(frozen=True)
class BatchRequest:
    identifier: str
    prompt: str
    max_new_tokens: int = 32
    seed: int = 42
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    stop: tuple[str, ...] = ()

    def validate(self, context: int):
        if not self.identifier or len(self.identifier)>128:
            raise ValueError("Request identifier must have 1..128 characters")
        if not isinstance(self.prompt,str) or not self.prompt:
            raise ValueError("Prompt must contain text")
        if not self.prompt.encode("utf-8"):
            raise ValueError("Prompt must contain at least one byte")
        if not 1 <= self.max_new_tokens <= 4096:
            raise ValueError("Invalid token count")
        if self.temperature < 0:
            raise ValueError("Temperature must be nonnegative")
        self.sampling().validate()
        if context < 2:
            raise ValueError("Model context too short")

    def sampling(self):
        return SamplingOptions(
            max_new_tokens=self.max_new_tokens,
            seed=self.seed,temperature=self.temperature,
            top_k=self.top_k,top_p=self.top_p,stop=self.stop,
        )


@dataclasses.dataclass
class BatchResult:
    identifier: str
    response: str
    prompt_bytes: int
    generated_tokens: int
    finish_reason: str
    stop_sequence: str | None
    cache_rebuilds: int
    elapsed_seconds: float

    def as_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class _Slot:
    request: BatchRequest
    prompt_bytes: int
    output_tokens: int = 0
    cache_rebuilds: int = 0
    complete: bool = False
    reason: str = "length"
    decoder: UTF8Accumulator = dataclasses.field(default_factory=UTF8Accumulator)
    matcher: StopMatcher | None = None
    chunks: list[str] = dataclasses.field(default_factory=list)
    created: float = dataclasses.field(default_factory=time.perf_counter)

    def __post_init__(self):
        self.matcher = StopMatcher(self.request.stop)

    def consume(self, byte: int):
        self.output_tokens += 1
        decoded = self.decoder.push(byte)
        visible = self.matcher.push(decoded)
        if visible:
            self.chunks.append(visible)
        if self.matcher.matched is not None:
            self.complete = True
            self.reason = "stop"
        elif self.output_tokens >= self.request.max_new_tokens:
            self.complete = True
            self.reason = "length"

    def finish(self):
        if self.reason != "stop":
            remainder = self.decoder.finish()
            self.chunks.append(self.matcher.push(remainder))
            self.chunks.append(self.matcher.finish())
        return BatchResult(
            identifier=self.request.identifier,
            response="".join(self.chunks),
            prompt_bytes=self.prompt_bytes,
            generated_tokens=self.output_tokens,
            finish_reason=self.reason,
            stop_sequence=self.matcher.matched,
            cache_rebuilds=self.cache_rebuilds,
            elapsed_seconds=max(0,time.perf_counter()-self.created),
        )


def group_by_length(requests: Iterable[BatchRequest], context: int):
    """Group by *effective* byte context, retaining original order metadata."""
    grouped = collections.defaultdict(list)
    for index,request in enumerate(requests):
        request.validate(context)
        raw = request.prompt.encode("utf-8")
        grouped[min(context,len(raw))].append((index,request,raw[-context:]))
    return dict(sorted(grouped.items()))


def build_prompt_batch(rows,device):
    """Create [batch,time] byte IDs, with no padding or misleading positions."""
    if not rows:
        raise ValueError("No prompts supplied")
    lengths = {len(row[2]) for row in rows}
    if len(lengths)!=1:
        raise ValueError("Different lengths require separate groups")
    return torch.tensor(
        [list(row[2]) for row in rows],
        dtype=torch.long,device=device,
    )


def _select_individual(logits: torch.Tensor,history: torch.Tensor,
                       slot: _Slot, rng: torch.Generator):
    """Per-request generators keep batch ordering from changing random samples."""
    return select_token(
        logits,history,slot.request.sampling(),generator=rng
    )


@torch.inference_mode()
def infer_group(model,rows,device:str="cpu") -> dict[int,BatchResult]:
    """Batch prefill, then batch decoding until a request completes.

    Completed rows are removed by slicing cache tensors. Grouping enables GPU
    utilization without mixing prompt lengths; dynamic compaction avoids wasting
    computation on finished requests. The cache is reindexed consistently.
    """
    if not rows:
        return {}
    model.eval()
    prompt_batch = build_prompt_batch(rows,device)
    slots = [_Slot(request=row[1],prompt_bytes=len(row[2])) for row in rows]
    random_generators = [
        generator_for(device,slot.request.seed) for slot in slots
    ]
    logits,state = prefill(model,prompt_batch)
    active = list(range(len(rows)))
    answer: dict[int,BatchResult] = {}
    while active:
        if logits.size(0)!=len(active) or state.batch_size!=len(active):
            raise RuntimeError("Batch rows and cache diverged")
        token_values = []
        survivors = []
        survivor_ids = []
        for row_index,slot_id in enumerate(active):
            slot = slots[slot_id]
            last_logits = logits[row_index:row_index+1,-1,:]
            history = state.tokens[row_index:row_index+1,:]
            token = _select_individual(
                last_logits,history,slot,random_generators[slot_id]
            )
            value = int(token.item())
            slot.consume(value)
            if slot.complete:
                answer[rows[slot_id][0]] = slot.finish()
            else:
                survivors.append(slot_id)
                survivor_ids.append(row_index)
                token_values.append(value)
        if not survivors:
            break
        selected = torch.tensor(survivor_ids,device=device,dtype=torch.long)
        kept_layers = []
        for layer in state.layers:
            kept_layers.append(type(layer)(
                layer.key.index_select(0,selected),
                layer.value.index_select(0,selected),
            ))
        kept_tokens = state.tokens.index_select(0,selected)
        state = KVState(
            kept_layers,state.positions,state.context_limit,kept_tokens
        )
        input_tokens = torch.tensor(
            token_values,dtype=torch.long,device=device
        ).view(-1,1)
        was_full = state.positions>=state.context_limit
        logits,state = append(model,input_tokens,state)
        if was_full:
            for slot_id in survivors:
                slots[slot_id].cache_rebuilds += 1
        active = survivors
    return answer


@torch.inference_mode()
def batch_generate(model,requests: Iterable[BatchRequest],
                   *,device:str | None=None,
                   max_batch_size:int=16) -> list[BatchResult]:
    """Main batched API, returning results in input order.

    It is a real simultaneous inference implementation within same-length
    groups. Short/long prompts have separate attention calls to avoid unsafe
    padding. A bounded microbatch size prevents unbounded memory consumption.
    """
    all_requests = list(requests)
    if not all_requests:
        return []
    if not 1 <= max_batch_size <= 1024:
        raise ValueError("Invalid max_batch_size")
    names = [entry.identifier for entry in all_requests]
    if len(set(names))!=len(names):
        raise ValueError("Duplicate batch identifiers")
    device = device or str(next(model.parameters()).device)
    grouped = group_by_length(all_requests,model.cfg.context)
    results = {}
    for _,rows in grouped.items():
        for start in range(0,len(rows),max_batch_size):
            sub_batch = rows[start:start+max_batch_size]
            results.update(infer_group(model,sub_batch,device))
    if len(results)!=len(all_requests):
        raise RuntimeError("Some batch requests did not complete")
    return [results[index] for index in range(len(all_requests))]


def naive_reference(model,requests: Iterable[BatchRequest],
                    *,device:str | None=None) -> list[BatchResult]:
    """Independent single-sequence baseline to catch batching regressions."""
    from .streaming import GenerationOptions,collect_events,stream_events
    device = device or str(next(model.parameters()).device)
    results = []
    for request in requests:
        request.validate(model.cfg.context)
        started = time.perf_counter()
        answer = collect_events(stream_events(
            model,request.prompt,
            GenerationOptions(sampling=request.sampling(),use_cache=True),
            device=device,
        ))
        usage = answer["usage"]
        results.append(BatchResult(
            identifier=request.identifier,
            response=answer["response"],
            prompt_bytes=len(request.prompt.encode("utf-8")),
            generated_tokens=usage["generated_tokens"],
            finish_reason=usage["finish_reason"],
            stop_sequence=usage["stopped_by"],
            cache_rebuilds=usage["cache_rebuilds"],
            elapsed_seconds=time.perf_counter()-started,
        ))
    return results


def compare_batch_to_single(model,requests:list[BatchRequest],
                            *,device:str | None=None) -> dict:
    """Compare deterministic content, finish reasons and generated token counts."""
    batched = batch_generate(model,requests,device=device)
    reference = naive_reference(model,requests,device=device)
    mismatches = []
    for b,r in zip(batched,reference):
        if (b.response,b.finish_reason,b.generated_tokens)!=(
            r.response,r.finish_reason,r.generated_tokens
        ):
            mismatches.append({
                "identifier":b.identifier,
                "batched_response":b.response,
                "reference_response":r.response,
                "batched_tokens":b.generated_tokens,
                "reference_tokens":r.generated_tokens,
            })
    return {
        "match":not mismatches,
        "requests":len(requests),
        "mismatches":mismatches,
    }


def batch_summary(results:Iterable[BatchResult]) -> dict:
    values = list(results)
    if not values:
        return {"requests":0,"tokens":0}
    return {
        "requests":len(values),
        "tokens":sum(item.generated_tokens for item in values),
        "avg_seconds":sum(item.elapsed_seconds for item in values)/len(values),
        "stops":dict(collections.Counter(item.finish_reason for item in values)),
        "cache_rebuilds":sum(item.cache_rebuilds for item in values),
    }
