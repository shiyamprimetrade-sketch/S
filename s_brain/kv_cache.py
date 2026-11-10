"""Exact incremental decode for the S-Brain causal Transformer.

Cache holds unexpanded grouped K/V tensors for each layer. The cached path uses
the *same* projection and feed-forward weights as model.py. There is no second
learned model. Causal prefill uses the original forward attention semantics;
incremental one-token decoding attends to all previously cached keys.

When reaching max_context we rebuild the full window: RoPE positions are
relative to the active window in the original model, so simply dropping a KV
entry would change positions and break numerical equivalence.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import torch
from torch.nn import functional as F

from .model import rope


@dataclass
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor

    def __post_init__(self):
        if self.key.ndim != 4 or self.key.shape != self.value.shape:
            raise ValueError("K and V must have same [batch,kv_heads,time,dim] shape")

    @property
    def length(self) -> int:
        return self.key.size(-2)

    @property
    def batch(self) -> int:
        return self.key.size(0)

    @property
    def bytes(self) -> int:
        return self.key.numel()*self.key.element_size()+self.value.numel()*self.value.element_size()

    def detached(self):
        return LayerKV(self.key.detach(),self.value.detach())


@dataclass
class KVState:
    layers: list[LayerKV] = field(default_factory=list)
    positions: int = 0
    context_limit: int = 0
    tokens: torch.Tensor | None = None

    def validate(self, layer_count: int | None = None) -> None:
        if self.positions < 0 or self.context_limit <= 0:
            raise ValueError("Invalid cache lengths")
        if self.positions > self.context_limit:
            raise ValueError("Cache exceeds model context")
        if layer_count is not None and len(self.layers) != layer_count:
            raise ValueError("Cache layer count differs from model")
        if self.layers and any(item.length != self.positions for item in self.layers):
            raise ValueError("Inconsistent layer KV length")
        if self.tokens is not None:
            if self.tokens.ndim != 2 or self.tokens.size(1) != self.positions:
                raise ValueError("Cache tokens do not match positions")

    @property
    def bytes(self) -> int:
        return sum(layer.bytes for layer in self.layers)

    @property
    def batch_size(self) -> int:
        return self.layers[0].batch if self.layers else (
            self.tokens.shape[0] if self.tokens is not None else 0
        )

    def detach(self):
        return KVState(
            [layer.detached() for layer in self.layers],
            self.positions,self.context_limit,
            None if self.tokens is None else self.tokens.detach()
        )


def rope_at(x: torch.Tensor, start: int) -> torch.Tensor:
    """RoPE at absolute local cache positions, compatible with model.rope."""
    if start < 0:
        raise ValueError("Negative RoPE offset")
    if start == 0:
        return rope(x)
    if x.ndim != 4 or x.size(-1) % 2:
        raise ValueError("Expected [batch,heads,time,even_dim]")
    count, dim = x.size(-2), x.size(-1)
    freq = 1.0 / (
        10000 ** (torch.arange(0,dim,2,device=x.device).float()/dim)
    )
    angles = torch.arange(start,start+count,device=x.device).float()[:,None]*freq[None,:]
    cosine = angles.cos()[None,None,:,:]
    sine = angles.sin()[None,None,:,:]
    a,b = x.float()[...,::2], x.float()[...,1::2]
    return torch.stack((a*cosine-b*sine,a*sine+b*cosine),dim=-1).flatten(-2).to(x.dtype)


def projected_kv(block, x: torch.Tensor, start: int):
    batch,time,_ = x.shape
    keys = block.kp(x).view(batch,time,block.k,block.d).transpose(1,2)
    values = block.v(x).view(batch,time,block.k,block.d).transpose(1,2)
    return LayerKV(rope_at(keys,start),values)


def attention_cached(block, x: torch.Tensor, cache: LayerKV | None,
                     start: int, causal: bool) -> tuple[torch.Tensor,LayerKV]:
    """Return attention output and unexpanded new KV state."""
    batch,time,dim = x.shape
    queries = block.q(x).view(batch,time,block.h,block.d).transpose(1,2)
    queries = rope_at(queries,start)
    current = projected_kv(block,x,start)
    if cache is None:
        all_kv = current
    else:
        if cache.batch != batch or cache.length != start:
            raise ValueError("Cache batch/offset mismatch")
        if cache.key.device != current.key.device:
            raise ValueError("Cache and input devices differ")
        all_kv = LayerKV(
            torch.cat((cache.key,current.key),dim=-2),
            torch.cat((cache.value,current.value),dim=-2),
        )
    expansion = block.h // block.k
    k = all_kv.key.repeat_interleave(expansion,dim=1)
    v = all_kv.value.repeat_interleave(expansion,dim=1)
    # A single incremental query is at the end of all keys; no causal mask is
    # necessary, whereas multi-token prefill must be causal.
    if cache is not None and time != 1:
        raise ValueError("Incremental mode accepts one token at a time")
    result = F.scaled_dot_product_attention(
        queries,k,v,is_causal=causal
    )
    result = result.transpose(1,2).contiguous().view(batch,time,dim)
    return block.o(result),all_kv


def block_cached(block, x: torch.Tensor, cache: LayerKV | None,
                 start: int) -> tuple[torch.Tensor,LayerKV]:
    y = block.n1(x)
    output, layer_cache = attention_cached(
        block,y,cache,start,causal=cache is None
    )
    x = x + output
    y = block.n2(x)
    x = x + block.down(F.silu(block.g(y))*block.u(y))
    return x,layer_cache


@torch.inference_mode()
def prefill(model, tokens: torch.Tensor) -> tuple[torch.Tensor, KVState]:
    """Process 1..context tokens and cache all layer keys and values."""
    if tokens.ndim != 2 or not 1 <= tokens.size(1) <= model.cfg.context:
        raise ValueError("Tokens must be [batch,time] inside context")
    model.eval()
    x = model.embed(tokens)
    layers = []
    for block in model.blocks:
        x, cache = block_cached(block,x,None,0)
        layers.append(cache)
    logits = model.lm_head(model.norm(x))
    state = KVState(layers,tokens.size(1),model.cfg.context,tokens.clone())
    state.validate(len(model.blocks))
    return logits,state


@torch.inference_mode()
def append(model, token: torch.Tensor,
           state: KVState) -> tuple[torch.Tensor,KVState]:
    """Append one token. On full context, rebuild the active window exactly."""
    if token.ndim != 2 or token.size(1) != 1:
        raise ValueError("Append requires [batch,1] input")
    state.validate(len(model.blocks))
    if token.size(0) != state.batch_size:
        raise ValueError("Input batch size differs from cache")
    if state.tokens is None:
        raise ValueError("Cache must retain token history")
    if token.device != state.tokens.device:
        raise ValueError("Cache/input device mismatch")
    if state.positions >= state.context_limit:
        tokens = torch.cat((state.tokens[:,1:],token),dim=1)
        return prefill(model,tokens)
    x = model.embed(token)
    new_layers = []
    for index,block in enumerate(model.blocks):
        x,next_layer = block_cached(block,x,state.layers[index],state.positions)
        new_layers.append(next_layer)
    logits = model.lm_head(model.norm(x))
    tokens = torch.cat((state.tokens,token),dim=1)
    result = KVState(new_layers,state.positions+1,state.context_limit,tokens)
    result.validate(len(model.blocks))
    return logits,result


@torch.inference_mode()
def extend(model, tokens: torch.Tensor,
           state: KVState) -> tuple[torch.Tensor, KVState]:
    """Append one or more tokens without incorrectly using a triangular mask."""
    if tokens.ndim != 2 or tokens.size(1) < 1:
        raise ValueError("Expected [batch,time>0]")
    logits = None
    for index in range(tokens.size(1)):
        logits,state = append(model,tokens[:,index:index+1],state)
    return logits,state


@torch.inference_mode()
def compare_full_and_cached(model, ids: torch.Tensor,
                            atol: float = 1e-5, rtol: float = 1e-5) -> dict:
    """Numerical parity test, including the context rollover boundary."""
    if ids.ndim != 2 or ids.size(1) < 1:
        raise ValueError("Expected nonempty token tensor")
    model.eval()
    delta = 0.0
    cached,state = prefill(model,ids[:,:1])
    reference,_ = model(ids[:,:1])
    delta = max(delta,(cached[:,-1,:]-reference[:,-1,:]).abs().max().item())
    for position in range(1,ids.size(1)):
        cached,state = append(model,ids[:,position:position+1],state)
        reference,_ = model(ids[:,max(0,position+1-model.cfg.context):position+1])
        err = (cached[:,-1,:]-reference[:,-1,:]).abs().max().item()
        delta = max(delta,err)
        if not torch.allclose(cached[:,-1,:],reference[:,-1,:],atol=atol,rtol=rtol):
            return {"match":False,"max_abs_error":delta,"first_bad_position":position}
    return {"match":True,"max_abs_error":delta,"tokens":ids.size(1),"cache_bytes":state.bytes}


@torch.inference_mode()
def next_token_logits(model, prompt: torch.Tensor,
                      state: KVState | None = None):
    """Get last-token logits, either prefill or append a single token."""
    if state is None:
        logits,state = prefill(model,prompt)
    else:
        if prompt.ndim != 2 or prompt.size(1) != 1:
            raise ValueError("Cached continuation must supply one token")
        logits,state = append(model,prompt,state)
    return logits[:,-1,:],state


def memory_estimate(config, batch: int, tokens: int, dtype_bytes: int = 4) -> int:
    """Predict bytes used by unexpanded K and V tensors; excludes scratch."""
    if batch < 1 or tokens < 0 or tokens > config.context or dtype_bytes < 1:
        raise ValueError("Invalid cache memory inputs")
    head_dim = config.dim//config.heads
    return batch*config.layers*config.kv_heads*tokens*head_dim*2*dtype_bytes


def cache_report(state: KVState) -> dict:
    state.validate()
    return {
        "layers":len(state.layers),
        "cached_tokens":state.positions,
        "context_limit":state.context_limit,
        "batch_size":state.batch_size,
        "bytes":state.bytes,
        "mebibytes":state.bytes/(1024**2),
        "dtype":str(state.layers[0].key.dtype) if state.layers else None,
        "device":str(state.layers[0].key.device) if state.layers else None,
    }


def rebuild_if_modified(model, state: KVState, new_prefix: torch.Tensor):
    """Explicitly invalidate cache when prompt edits invalidate prefix reuse."""
    if state.tokens is not None and torch.equal(state.tokens,new_prefix):
        return state
    _,recomputed = prefill(model,new_prefix[:,-model.cfg.context:])
    return recomputed
