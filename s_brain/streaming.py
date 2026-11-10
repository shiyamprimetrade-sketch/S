"""Cached generation, Unicode-safe delta events, stop sequences and cancellation.

Network adapters consume typed events and decide whether to emit SSE, NDJSON
or an ordinary JSON response. This module never opens network connections.
A byte-level model may output invalid UTF-8; the decoder holds incomplete
sequences and replaces irrecoverably invalid data rather than sending malformed
JSON or slicing multibyte characters in half.
"""
from __future__ import annotations

import codecs
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterator

import torch

from .kv_cache import append, prefill, cache_report
from .sampling import SamplingOptions, generator_for, select_token


@dataclass(frozen=True)
class StreamEvent:
    type: str
    sequence: int
    text: str = ""
    token_id: int | None = None
    finish_reason: str | None = None
    usage: dict | None = None
    model: str = "s-brain:experimental"

    def as_dict(self) -> dict:
        result = asdict(self)
        return {key:value for key,value in result.items() if value is not None}


@dataclass
class GenerationStats:
    prompt_bytes: int = 0
    context_tokens: int = 0
    generated_tokens: int = 0
    visible_characters: int = 0
    started: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    ended: float | None = None
    finish_reason: str = "length"
    stopped_by: str | None = None
    cache_rebuilds: int = 0

    def finish(self, reason: str) -> None:
        self.finish_reason = reason
        self.ended = time.perf_counter()

    def report(self) -> dict:
        end = self.ended if self.ended is not None else time.perf_counter()
        elapsed = max(0,end-self.started)
        first = None if self.first_token_at is None else (
            self.first_token_at-self.started
        )
        return {
            "prompt_bytes":self.prompt_bytes,
            "context_tokens":self.context_tokens,
            "generated_tokens":self.generated_tokens,
            "visible_characters":self.visible_characters,
            "finish_reason":self.finish_reason,
            "stopped_by":self.stopped_by,
            "cache_rebuilds":self.cache_rebuilds,
            "elapsed_seconds":elapsed,
            "time_to_first_token_seconds":first,
            "tokens_per_second":self.generated_tokens/elapsed if elapsed>0 else 0.0,
        }


class UTF8Accumulator:
    """Incremental UTF-8 decoder, stable for fragmented Unicode bytes."""

    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._finished = False
        self._count = 0

    def push(self, value: int) -> str:
        if self._finished:
            raise RuntimeError("Decoder already flushed")
        if not 0 <= value <= 255:
            raise ValueError("Token must be a byte")
        self._count += 1
        return self._decoder.decode(bytes((value,)),final=False)

    def finish(self) -> str:
        if self._finished:
            return ""
        self._finished = True
        return self._decoder.decode(b"",final=True)

    @property
    def bytes_seen(self) -> int:
        return self._count


class StopMatcher:
    """Hold back potential stop prefixes to avoid leaking stop text to clients.

    A bounded suffix may be a prefix of a stop string. That suffix is not sent
    yet. Once it cannot complete a stop marker it is released safely.
    """

    def __init__(self, sequences: tuple[str, ...] = ()):
        if len(sequences)>16 or any(not item or len(item)>256 for item in sequences):
            raise ValueError("Unsupported stop sequences")
        self.sequences = sequences
        self.pending = ""
        self.matched: str | None = None
        self.finished = False
        self._max_len = max(map(len,sequences),default=0)

    def push(self, text: str) -> str:
        if self.finished:
            raise RuntimeError("Stop matcher already completed")
        if self.matched is not None:
            return ""
        self.pending += text
        if not self.sequences:
            released,self.pending = self.pending,""
            return released

        matches = []
        for marker in self.sequences:
            at = self.pending.find(marker)
            if at != -1:
                matches.append((at,marker))
        if matches:
            at,marker = min(matches,key=lambda item:(item[0],len(item[1])))
            released = self.pending[:at]
            self.pending = ""
            self.matched = marker
            return released

        # Retain longest suffix that can still be the beginning of a stop.
        keep = 0
        for marker in self.sequences:
            upto = min(len(marker)-1,len(self.pending))
            for length in range(1,upto+1):
                if self.pending.endswith(marker[:length]):
                    keep = max(keep,length)
        if keep:
            released = self.pending[:-keep]
            self.pending = self.pending[-keep:]
        else:
            released,self.pending = self.pending,""
        return released

    def finish(self) -> str:
        self.finished = True
        released = self.pending if self.matched is None else ""
        self.pending = ""
        return released


@dataclass
class GenerationOptions:
    sampling: SamplingOptions = field(default_factory=SamplingOptions)
    use_cache: bool = True
    return_token_ids: bool = False
    max_prompt_bytes: int = 16000

    def validate(self):
        self.sampling.validate()
        if self.max_prompt_bytes < 1:
            raise ValueError("max_prompt_bytes must be positive")


def prepare_prompt(prompt: str, context: int,
                   max_bytes: int) -> tuple[list[int],int]:
    if not isinstance(prompt,str) or not prompt:
        raise ValueError("Prompt must be nonempty text")
    raw = prompt.encode("utf-8")
    if len(raw)>max_bytes:
        raise ValueError("Prompt exceeds byte allowance")
    if context < 1:
        raise ValueError("Context must be positive")
    # The network always consumes bytes, not Unicode code points.
    return list(raw[-context:]),len(raw)


@torch.inference_mode()
def stream_events(
    model,
    prompt: str,
    options: GenerationOptions | None = None,
    *,
    device: str | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Iterator[StreamEvent]:
    """Yield start, delta and complete events from independent S-Brain weights.

    The loop maintains a cache only until the model context fills; it then
    rebuilds the active window for exact parity with the original model.
    No model output is executed as a command.
    """
    options = options or GenerationOptions()
    options.validate()
    model.eval()
    device = device or str(next(model.parameters()).device)
    token_values,raw_bytes = prepare_prompt(
        prompt,model.cfg.context,options.max_prompt_bytes
    )
    prompt_ids = torch.tensor([token_values],dtype=torch.long,device=device)
    stats = GenerationStats(prompt_bytes=raw_bytes,context_tokens=len(token_values))
    stats.started = time.perf_counter()
    tokenizer = UTF8Accumulator()
    stops = StopMatcher(options.sampling.stop)
    rng = generator_for(device,options.sampling.seed)
    emitted = 0
    sequence = 0

    yield StreamEvent(
        type="start",sequence=sequence,
        usage={"prompt_tokens":len(token_values),"cache":options.use_cache}
    )

    if options.use_cache:
        logits,state = prefill(model,prompt_ids)
    else:
        logits,_ = model(prompt_ids)
        history = prompt_ids
        state = None

    for i in range(options.sampling.max_new_tokens):
        if cancelled and cancelled():
            stats.finish("cancelled")
            break
        scores = logits[:,-1,:]
        history_for_penalties = state.tokens if state is not None else history
        token = select_token(scores,history_for_penalties,options.sampling,rng)
        integer = int(token.item())
        if stats.first_token_at is None:
            stats.first_token_at = time.perf_counter()
        stats.generated_tokens += 1
        raw_chunk = tokenizer.push(integer)
        visible = stops.push(raw_chunk)
        if visible:
            stats.visible_characters += len(visible)
            sequence += 1
            emitted += 1
            yield StreamEvent(
                type="delta",sequence=sequence,text=visible,
                token_id=integer if options.return_token_ids else None
            )
        if stops.matched:
            stats.stopped_by = stops.matched
            stats.finish("stop")
            break
        if i+1 >= options.sampling.max_new_tokens:
            stats.finish("length")
            break
        if state is not None:
            was_full = state.positions >= state.context_limit
            logits,state = append(model,token,state)
            if was_full:
                stats.cache_rebuilds += 1
        else:
            history = torch.cat((history,token),dim=1)
            logits,_ = model(history[:,-model.cfg.context:])
    else:
        stats.finish("length")

    if stats.finish_reason != "stop":
        final = tokenizer.finish()
        visible = stops.push(final)
        tail = stops.finish()
        if visible or tail:
            sequence += 1
            chunk = visible+tail
            stats.visible_characters += len(chunk)
            yield StreamEvent(type="delta",sequence=sequence,text=chunk)

    sequence += 1
    yield StreamEvent(
        type="complete",sequence=sequence,
        finish_reason=stats.finish_reason,
        usage=stats.report()
    )


def collect_events(events: Iterator[StreamEvent]) -> dict:
    """Collect a stream for ordinary non-streaming JSON API compatibility."""
    text_parts = []
    final = None
    first = None
    for event in events:
        if event.type == "start":
            first = event
        elif event.type == "delta":
            text_parts.append(event.text)
        elif event.type == "complete":
            final = event
    if final is None:
        raise RuntimeError("Stream ended without completion event")
    return {
        "model":final.model,
        "response":"".join(text_parts),
        "finish_reason":final.finish_reason,
        "usage":final.usage,
    }


def jsonl_event(event: StreamEvent) -> bytes:
    import json
    return (json.dumps(event.as_dict(),ensure_ascii=False,separators=(",",":"))+"\n").encode("utf-8")


def sse_event(event: StreamEvent) -> bytes:
    """SSE framing; JSON encoding ensures user content is not injected as headers."""
    import json
    payload = json.dumps(event.as_dict(),ensure_ascii=False,separators=(",",":"))
    return f"event: {event.type}\ndata: {payload}\n\n".encode("utf-8")


def stream_to_text(events: Iterator[StreamEvent]) -> str:
    return "".join(event.text for event in events if event.type=="delta")


def model_device(model) -> str:
    return str(next(model.parameters()).device)


def compare_stream_modes(model,prompt: str,seed: int = 7,
                         tokens: int = 24,temperature: float = 0.7) -> dict:
    """Verify cached and uncached generation choose identical token sequences."""
    sampling = SamplingOptions(
        max_new_tokens=tokens,temperature=temperature,seed=seed
    )
    cached = collect_events(stream_events(
        model,prompt,GenerationOptions(sampling=sampling,use_cache=True)
    ))
    uncached = collect_events(stream_events(
        model,prompt,GenerationOptions(sampling=sampling,use_cache=False)
    ))
    return {
        "match":cached["response"]==uncached["response"],
        "cached_response":cached["response"],
        "uncached_response":uncached["response"],
        "cached_usage":cached["usage"],
        "uncached_usage":uncached["usage"],
    }
