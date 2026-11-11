"""Bounded, explicit cache accounting and eviction policy for S-Brain.

This module manages snapshots produced by kv_cache.prefill/append. It does not
implement unsafe cross-request KV sharing: requests with different prompts
may be aliased only after verifying the *entire* active token prefix. Cache
entries are immutable references to detached tensors. Eviction is LRU with
per-entry TTL and byte quotas; it is always acceptable to miss and recompute.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import threading
import time
from typing import Callable

import torch

from .kv_cache import KVState, cache_report, prefill


@dataclasses.dataclass(frozen=True)
class CachePolicy:
    max_entries: int = 16
    max_bytes: int = 256*1024*1024
    ttl_seconds: float = 120.0
    max_context: int = 65536
    allow_prefix_reuse: bool = False

    def validate(self) -> None:
        if not 1<=self.max_entries<=100000:
            raise ValueError("Invalid max_entries")
        if not 0<self.max_bytes<=2**50:
            raise ValueError("Invalid byte limit")
        if not 0<self.ttl_seconds<=86400:
            raise ValueError("Invalid TTL")
        if not 1<=self.max_context<=2**20:
            raise ValueError("Invalid context limit")


@dataclasses.dataclass(frozen=True)
class CacheIdentity:
    model_fingerprint: str
    device: str
    dtype: str
    token_digest: str
    token_count: int

    @classmethod
    def for_tokens(cls,model_fingerprint,values:torch.Tensor):
        if values.ndim!=2 or values.size(0)!=1:
            raise ValueError("Cache keys currently require one sequence")
        if values.dtype!=torch.long:
            raise ValueError("Expected torch.long token IDs")
        contiguous = values.detach().to("cpu").contiguous()
        # Stable across CUDA and CPU devices; exact byte comparison is repeated
        # after lookup to guard against accidental identity mismatches.
        data = contiguous.numpy().tobytes()
        digest = hashlib.sha256(data).hexdigest()
        return cls(
            model_fingerprint=str(model_fingerprint),
            device=str(values.device),
            dtype=str(values.dtype),
            token_digest=digest,
            token_count=int(values.size(1)),
        )


@dataclasses.dataclass
class CacheEntry:
    identity: CacheIdentity
    state: KVState
    inserted: float
    accessed: float
    hits: int = 0
    creator: str = "local"

    @property
    def bytes(self):
        token_bytes = (
            self.state.tokens.numel()*self.state.tokens.element_size()
            if self.state.tokens is not None else 0
        )
        return self.state.bytes+token_bytes

    def age(self,now):
        return max(0,now-self.inserted)

    def idle(self,now):
        return max(0,now-self.accessed)


class CachePool:
    """Thread-safe memory quota for detached single-sequence KV entries.

    GPU tensors may remain allocated briefly after deletion while PyTorch's
    caching allocator retains memory; the quota is logical tensor payload size.
    """

    def __init__(self,policy:CachePolicy | None=None,
                 clock:Callable[[],float] | None=None):
        self.policy = policy or CachePolicy()
        self.policy.validate()
        self.clock = clock or time.monotonic
        self._guard = threading.RLock()
        self._entries: collections.OrderedDict[CacheIdentity,CacheEntry] = collections.OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = collections.Counter()

    def _remove(self,key,reason):
        item = self._entries.pop(key,None)
        if item is not None:
            self._bytes -= item.bytes
            self._evictions[reason] += 1

    def _expire(self):
        now = self.clock()
        stale = [
            key for key,item in self._entries.items()
            if item.age(now)>self.policy.ttl_seconds
        ]
        for key in stale:
            self._remove(key,"expired")

    def _enforce(self):
        while self._entries and (
            len(self._entries)>self.policy.max_entries
            or self._bytes>self.policy.max_bytes
        ):
            oldest = next(iter(self._entries))
            reason = "entry_quota" if len(self._entries)>self.policy.max_entries else "byte_quota"
            self._remove(oldest,reason)

    def insert(self,identity:CacheIdentity,state:KVState,
               creator:str="local") -> bool:
        state.validate()
        if state.tokens is None:
            raise ValueError("Cannot cache state without exact token history")
        if state.batch_size!=1 or state.positions!=identity.token_count:
            raise ValueError("Cache identity/state length mismatch")
        # Fingerprints prevent reusing caches across different model weights.
        current = CacheIdentity.for_tokens(identity.model_fingerprint,state.tokens)
        if current!=identity:
            raise ValueError("Cache identity mismatch")
        detached = state.detach()
        now = self.clock()
        entry = CacheEntry(identity,detached,now,now,creator=creator)
        if entry.bytes>self.policy.max_bytes:
            return False
        with self._guard:
            self._expire()
            if identity in self._entries:
                self._remove(identity,"replaced")
            self._entries[identity] = entry
            self._bytes += entry.bytes
            self._enforce()
            return identity in self._entries

    def get(self,identity:CacheIdentity,
            expected_tokens:torch.Tensor | None=None) -> KVState | None:
        with self._guard:
            self._expire()
            entry = self._entries.get(identity)
            if entry is None:
                self._misses += 1
                return None
            if expected_tokens is not None:
                # Verify full token values, not just a truncated hash.
                if not torch.equal(
                    expected_tokens.detach().to(entry.state.tokens.device),
                    entry.state.tokens,
                ):
                    self._misses += 1
                    return None
            now = self.clock()
            entry.accessed = now
            entry.hits += 1
            self._hits += 1
            self._entries.move_to_end(identity)
            return entry.state.detach()

    def evict(self,identity:CacheIdentity)->bool:
        with self._guard:
            exists = identity in self._entries
            self._remove(identity,"manual")
            return exists

    def clear(self):
        with self._guard:
            for identity in list(self._entries):
                self._remove(identity,"clear")

    def prune(self) -> int:
        with self._guard:
            before = len(self._entries)
            self._expire()
            self._enforce()
            return before-len(self._entries)

    def statistics(self) -> dict:
        with self._guard:
            self._expire()
            return {
                "entries":len(self._entries),
                "payload_bytes":self._bytes,
                "payload_mib":self._bytes/(1024**2),
                "max_entries":self.policy.max_entries,
                "max_bytes":self.policy.max_bytes,
                "hits":self._hits,
                "misses":self._misses,
                "evictions":dict(self._evictions),
                "hit_rate":self._hits/max(1,self._hits+self._misses),
            }

    def describe(self) -> list[dict]:
        with self._guard:
            self._expire()
            now = self.clock()
            return [
                {
                    "model_fingerprint":key.model_fingerprint,
                    "prefix_length":key.token_count,
                    "bytes":item.bytes,
                    "age_seconds":item.age(now),
                    "idle_seconds":item.idle(now),
                    "hits":item.hits,
                    "creator":item.creator,
                }
                for key,item in self._entries.items()
            ]

    def __len__(self):
        return self.statistics()["entries"]


def fingerprint_model(model) -> str:
    """Hash model weights: useful for correctness, expensive for large models.

    Compute once after loading a model, never on the generation hot path.
    """
    digest = hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        values = value.detach().to("cpu").contiguous()
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def key_for_prompt(model_fingerprint:str,prompt:torch.Tensor):
    return CacheIdentity.for_tokens(model_fingerprint,prompt)


def cached_prefill(model,prompt:torch.Tensor,pool:CachePool,
                   model_fingerprint:str):
    """Return last-token logits and cached state.

    A KVState only contains intermediate attention keys/values, not the final
    output logits. Exact re-use therefore cannot avoid computing last-token
    logits unless a corresponding output is also stored. Until that extension,
    this function returns a cache hit flag and recomputes logits with normal
    prefill to guarantee correctness.
    """
    identity = key_for_prompt(model_fingerprint,prompt)
    hit = pool.get(identity,expected_tokens=prompt)
    logits,state = prefill(model,prompt)
    if hit is None:
        pool.insert(identity,state)
    return logits,state,hit is not None


def inspect_efficiency(model,prompt:torch.Tensor,
                       pool:CachePool,fingerprint:str) -> dict:
    start = time.perf_counter()
    _,state,hit = cached_prefill(model,prompt,pool,fingerprint)
    return {
        "cache_hit":hit,
        "wall_seconds":time.perf_counter()-start,
        "state":cache_report(state),
        "pool":pool.statistics(),
        "note":"Current cache hit confirms identity, not faster prefill",
    }


def validate_pool_invariants(pool:CachePool)->None:
    with pool._guard:
        total = sum(item.bytes for item in pool._entries.values())
        if total!=pool._bytes:
            raise AssertionError("Cache pool byte accounting mismatch")
        if pool._bytes>pool.policy.max_bytes:
            raise AssertionError("Byte quota exceeded")
        if len(pool._entries)>pool.policy.max_entries:
            raise AssertionError("Entry quota exceeded")
        for identity,item in pool._entries.items():
            if identity!=item.identity:
                raise AssertionError("Cache identity key changed")
            item.state.validate()
