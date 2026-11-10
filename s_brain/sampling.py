"""Sampling, decoding and generation controls for the independent S Transformer.

Generation operates on byte IDs. It deliberately does not claim chat tuning:
the base model is next-byte prediction and must be evaluated before use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SamplingOptions:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: int | None = None
    stop: tuple[str, ...] = ()

    def validate(self) -> None:
        if not 1 <= self.max_new_tokens <= 4096:
            raise ValueError("max_new_tokens must be 1..4096")
        if self.temperature < 0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite and nonnegative")
        if self.top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0,1]")
        if not 0 <= self.min_p < 1:
            raise ValueError("min_p must be in [0,1)")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if not all(math.isfinite(x) for x in (
            self.repetition_penalty, self.frequency_penalty, self.presence_penalty
        )):
            raise ValueError("Penalty settings must be finite")
        if len(self.stop) > 16 or any(len(item) > 256 or not item for item in self.stop):
            raise ValueError("Invalid stop strings")


def penalize_logits(logits: torch.Tensor, history: torch.Tensor,
                    options: SamplingOptions) -> torch.Tensor:
    """Apply presence/frequency and multiplicative repetition penalties."""
    if history.ndim != 2 or logits.ndim != 2:
        raise ValueError("Expected [batch,time] history and [batch,vocab] logits")
    if history.shape[0] != logits.shape[0]:
        raise ValueError("Batch mismatch")
    result = logits.clone()
    for batch in range(history.size(0)):
        ids, counts = history[batch].unique(return_counts=True)
        scores = result[batch, ids]
        if options.repetition_penalty != 1:
            scores = torch.where(
                scores > 0,
                scores / options.repetition_penalty,
                scores * options.repetition_penalty,
            )
        if options.frequency_penalty:
            scores = scores - options.frequency_penalty * counts
        if options.presence_penalty:
            scores = scores - options.presence_penalty
        result[batch, ids] = scores
    return result


def filter_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    values = torch.topk(logits, top_k, dim=-1).values
    threshold = values[..., -1, None]
    return logits.masked_fill(logits < threshold, -float("inf"))


def filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1:
        return logits
    sorted_logits, indices = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    remove = cumulative - probs >= top_p
    remove[..., 0] = False
    filtered = sorted_logits.masked_fill(remove, -float("inf"))
    output = torch.full_like(logits, -float("inf"))
    return output.scatter(dim=-1, index=indices, src=filtered)


def filter_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    if min_p <= 0:
        return logits
    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True).values
    reject = probs < min_p * max_prob
    return logits.masked_fill(reject, -float("inf"))


def select_token(logits: torch.Tensor, history: torch.Tensor,
                 options: SamplingOptions,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    options.validate()
    if logits.size(0) != 1:
        raise ValueError("Streaming sampler supports batch size one")
    if options.temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    scores = penalize_logits(logits.float(), history, options)
    scores = scores / options.temperature
    scores = filter_top_k(scores, options.top_k)
    scores = filter_top_p(scores, options.top_p)
    scores = filter_min_p(scores, options.min_p)
    probabilities = F.softmax(scores, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("Invalid probability distribution")
    return torch.multinomial(probabilities, 1, generator=generator)


def encode_bytes(text: str) -> list[int]:
    if not isinstance(text, str):
        raise TypeError("Expected prompt string")
    return list(text.encode("utf-8"))


def decode_bytes(ids: list[int], errors: str = "replace") -> str:
    if any(not 0 <= value <= 255 for value in ids):
        raise ValueError("Non-byte token generated")
    return bytes(ids).decode("utf-8", errors=errors)


def truncate_stop(text: str, stop: tuple[str, ...]) -> tuple[str, str | None]:
    found = []
    for marker in stop:
        location = text.find(marker)
        if location >= 0:
            found.append((location, marker))
    if not found:
        return text, None
    location, marker = min(found, key=lambda item: (item[0], len(item[1])))
    return text[:location], marker


def generator_for(device: str, seed: int | None):
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


@torch.inference_mode()
def generate_tokens(model, prompt: list[int], options: SamplingOptions,
                    device: str = "cpu") -> tuple[list[int], dict]:
    """Sampling without a KV cache; context is recomputed each step."""
    options.validate()
    if not prompt:
        raise ValueError("Prompt is empty")
    model.eval()
    ids = torch.tensor([prompt[-model.cfg.context:]], dtype=torch.long, device=device)
    new_tokens: list[int] = []
    rng = generator_for(device, options.seed)
    stop_reason = "length"
    stop_value = None
    for _ in range(options.max_new_tokens):
        context = ids[:, -model.cfg.context:]
        logits, _ = model(context)
        next_id = select_token(logits[:, -1, :], ids, options, rng)
        value = next_id.item()
        new_tokens.append(value)
        ids = torch.cat((ids, next_id), dim=-1)
        if options.stop:
            text = decode_bytes(new_tokens)
            _, matched = truncate_stop(text, options.stop)
            if matched is not None:
                stop_reason = "stop"
                stop_value = matched
                break
    result = decode_bytes(new_tokens)
    result, _ = truncate_stop(result, options.stop)
    return list(result.encode("utf-8")), {
        "finish_reason": stop_reason,
        "stop_sequence": stop_value,
        "tokens_generated": len(new_tokens),
        "context_tokens": len(prompt[-model.cfg.context:]),
        "seed": options.seed,
    }


@torch.inference_mode()
def generate_stream(model, prompt: list[int], options: SamplingOptions,
                    device: str = "cpu") -> Iterator[int]:
    """Yield raw byte IDs; consumer owns UTF-8 buffering and stop handling."""
    options.validate()
    if not prompt:
        raise ValueError("Prompt is empty")
    model.eval()
    ids = torch.tensor([prompt[-model.cfg.context:]], dtype=torch.long, device=device)
    rng = generator_for(device, options.seed)
    emitted = bytearray()
    for _ in range(options.max_new_tokens):
        logits, _ = model(ids[:, -model.cfg.context:])
        token = select_token(logits[:, -1, :], ids, options, rng)
        value = token.item()
        emitted.append(value)
        ids = torch.cat((ids, token), dim=-1)
        yield value
        if options.stop and truncate_stop(emitted.decode("utf-8", errors="replace"), options.stop)[1]:
            break


def estimate_work(model_config, prompt_tokens: int, new_tokens: int) -> dict:
    """Rough prefill/decoding estimate, not a real throughput benchmark."""
    if prompt_tokens < 1 or new_tokens < 0:
        raise ValueError("Invalid token counts")
    dim = model_config.dim
    layers = model_config.layers
    # A rough 12*N FLOPs/token rule for decoder Transformers.
    parameters = 0
    parameters += model_config.vocab_size * dim
    parameters += layers * (4*dim*dim + 3*(8*dim//3)*dim)
    total = prompt_tokens + new_tokens
    return {
        "approx_parameter_count": parameters,
        "approx_forward_flops": 2 * parameters * total,
        "note": "Approximation excludes attention quadratics, launch overhead and cache behavior",
    }
