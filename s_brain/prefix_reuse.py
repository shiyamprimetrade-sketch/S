"""Exact-prefix KV and logit reuse for S-Brain inference.

Unlike the Cycle 3 CachePool prototype, this cache stores both the last
logits *and* the KVState. When an exact prompt is repeated, no prefill model
execution is needed. For a verified prefix of a longer prompt, only the
remaining suffix tokens need attention projections and model execution.
Caches are scoped to model-weight fingerprints and local request data.

A cached prefix is only useful while length <= model.cfg.context. Context
rollover rebuilds a full active window using the model's relative RoPE.
Never reuse a prefix after editing its tokens or switching model weights.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import threading
import time
from collections.abc import Callable
from typing import Any

import torch

from .kv_cache import KVState, append, prefill


@dataclasses.dataclass(frozen=True)
class PrefixPolicy:
    max_entries: int = 32
    max_bytes: int = 128 * 1024 * 1024
    ttl_seconds: float = 90.0
    min_prefix_tokens: int = 2
    share_across_requests: bool = False

    def validate(self) -> None:
        if not 1 <= self.max_entries <= 100000:
            raise ValueError("max_entries out of range")
        if not 1 <= self.max_bytes <= 2**48:
            raise ValueError("max_bytes out of range")
        if not 0 < self.ttl_seconds <= 86400:
            raise ValueError("ttl_seconds out of range")
        if self.min_prefix_tokens < 1:
            raise ValueError("min_prefix_tokens must be positive")


@dataclasses.dataclass(frozen=True)
class PrefixKey:
    model_id: str
    token_hash: str
    length: int

    @classmethod
    def create(cls, model_id: str, ids: torch.Tensor) -> "PrefixKey":
        if not model_id or not isinstance(model_id, str):
            raise ValueError("model_id is required")
        if ids.ndim != 2 or ids.size(0) != 1 or ids.dtype != torch.long:
            raise ValueError("Prefix cache supports [1,time] long tokens")
        if ids.size(1) < 1:
            raise ValueError("Prefix must contain tokens")
        raw = ids.detach().cpu().contiguous().numpy().tobytes()
        return cls(model_id, hashlib.sha256(raw).hexdigest(), ids.size(1))


@dataclasses.dataclass
class PrefixValue:
    key: PrefixKey
    state: KVState
    logits: torch.Tensor
    inserted: float
    accessed: float
    project_scope: str
    hits: int = 0

    @property
    def payload_bytes(self) -> int:
        tokens = self.state.tokens
        token_bytes = 0 if tokens is None else tokens.numel() * tokens.element_size()
        return (
            self.state.bytes + token_bytes
            + self.logits.numel() * self.logits.element_size()
        )


@dataclasses.dataclass(frozen=True)
class PrefixHit:
    kind: str
    matched_tokens: int
    remaining_tokens: int
    saved_prefill_tokens: int
    cache_key: PrefixKey | None = None

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


class PrefixCache:
    """Thread-safe process-local snapshot cache with explicit project isolation."""

    def __init__(
        self,
        policy: PrefixPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy or PrefixPolicy()
        self.policy.validate()
        self.clock = clock
        self._lock = threading.RLock()
        self._entries: collections.OrderedDict[tuple[str, PrefixKey], PrefixValue] = (
            collections.OrderedDict()
        )
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions: collections.Counter[str] = collections.Counter()

    def _scope(self, scope: str) -> str:
        if self.policy.share_across_requests:
            return "_explicit_shared_scope_"
        if not scope:
            raise ValueError("project_scope required for isolated cache")
        return scope

    def _remove(self, identity, cause: str) -> None:
        value = self._entries.pop(identity, None)
        if value is not None:
            self._bytes -= value.payload_bytes
            self._evictions[cause] += 1

    def _prune(self) -> None:
        now = self.clock()
        expired = [
            identity
            for identity, value in self._entries.items()
            if now - value.inserted > self.policy.ttl_seconds
        ]
        for identity in expired:
            self._remove(identity, "ttl")
        while (
            len(self._entries) > self.policy.max_entries
            or self._bytes > self.policy.max_bytes
        ):
            self._remove(next(iter(self._entries)), "capacity")

    def put(
        self,
        model_id: str,
        state: KVState,
        logits: torch.Tensor,
        *,
        project_scope: str,
    ) -> bool:
        state.validate()
        if state.tokens is None or state.batch_size != 1:
            raise ValueError("Only single-sequence states with tokens can be cached")
        if state.positions < self.policy.min_prefix_tokens:
            return False
        if state.positions != state.tokens.size(1):
            raise ValueError("Cached state position mismatch")
        if logits.ndim != 3 or logits.size(0) != 1:
            raise ValueError("Logits must be [1,time,vocab]")
        if logits.size(1) not in (1, state.positions):
            raise ValueError("Logit sequence length mismatch")
        key = PrefixKey.create(model_id, state.tokens)
        now = self.clock()
        value = PrefixValue(
            key=key,
            state=state.detach(),
            logits=logits[:, -1:, :].detach().clone(),
            inserted=now,
            accessed=now,
            project_scope=self._scope(project_scope),
        )
        if value.payload_bytes > self.policy.max_bytes:
            return False
        identity = (self._scope(project_scope), key)
        with self._lock:
            self._prune()
            self._remove(identity, "replaced")
            self._entries[identity] = value
            self._bytes += value.payload_bytes
            self._prune()
            return identity in self._entries

    def lookup(
        self,
        model_id: str,
        tokens: torch.Tensor,
        *,
        project_scope: str,
        allow_prefix: bool = True,
    ) -> tuple[torch.Tensor | None, KVState | None, PrefixHit]:
        if tokens.ndim != 2 or tokens.size(0) != 1 or tokens.dtype != torch.long:
            raise ValueError("Expected [1,time] long token tensor")
        if tokens.size(1) < 1:
            raise ValueError("Empty prompt")
        scope = self._scope(project_scope)
        with self._lock:
            self._prune()
            best = None
            for identity, value in self._entries.items():
                if identity[0] != scope or value.key.model_id != model_id:
                    continue
                length = value.key.length
                if length > tokens.size(1):
                    continue
                if not allow_prefix and length != tokens.size(1):
                    continue
                if best is not None and best.key.length >= length:
                    continue
                # Both full prefix contents and digest must match.
                target = tokens[:, :length]
                if value.key != PrefixKey.create(model_id, target):
                    continue
                if not torch.equal(value.state.tokens, target.to(value.state.tokens.device)):
                    continue
                best = value
            if best is None:
                self._misses += 1
                return None, None, PrefixHit("miss", 0, tokens.size(1), 0)
            best.hits += 1
            best.accessed = self.clock()
            identity = (scope, best.key)
            self._entries.move_to_end(identity)
            self._hits += 1
            matched = best.key.length
            return (
                best.logits.detach().clone(),
                best.state.detach(),
                PrefixHit(
                    "exact" if matched == tokens.size(1) else "prefix",
                    matched,
                    tokens.size(1) - matched,
                    matched,
                    best.key,
                ),
            )

    def stats(self) -> dict:
        with self._lock:
            self._prune()
            return {
                "entries": len(self._entries),
                "payload_bytes": self._bytes,
                "hits": self._hits,
                "misses": self._misses,
                "hit_fraction": self._hits / max(1, self._hits + self._misses),
                "evictions": dict(self._evictions),
            }

    def clear(self, *, project_scope: str | None = None) -> int:
        with self._lock:
            identities = [
                identity for identity in self._entries
                if project_scope is None or identity[0] == self._scope(project_scope)
            ]
            for identity in identities:
                self._remove(identity, "manual")
            return len(identities)

    def validate(self) -> None:
        with self._lock:
            self._prune()
            expected = sum(x.payload_bytes for x in self._entries.values())
            if expected != self._bytes or self._bytes > self.policy.max_bytes:
                raise AssertionError("Prefix cache memory accounting mismatch")
            if len(self._entries) > self.policy.max_entries:
                raise AssertionError("Prefix cache entry limit exceeded")
            for identity, value in self._entries.items():
                value.state.validate()
                if identity[1] != value.key:
                    raise AssertionError("Cache identity changed")


@torch.inference_mode()
def prefill_reusing_prefix(
    model: Any,
    tokens: torch.Tensor,
    *,
    cache: PrefixCache,
    model_id: str,
    project_scope: str,
    remember: bool = True,
) -> tuple[torch.Tensor, KVState, PrefixHit]:
    """Avoid previously computed prefix work with exact layer-state equivalence."""
    if tokens.ndim != 2 or tokens.size(0) != 1:
        raise ValueError("Expected batch size one")
    if not 1 <= tokens.size(1) <= model.cfg.context:
        raise ValueError("Prompt must fit context")
    model.eval()
    logits, state, hit = cache.lookup(
        model_id, tokens, project_scope=project_scope
    )
    if state is None:
        logits, state = prefill(model, tokens)
    else:
        if state.positions > model.cfg.context:
            raise ValueError("Cached state exceeds model context")
        for position in range(hit.matched_tokens, tokens.size(1)):
            token = tokens[:, position:position + 1]
            logits, state = append(model, token, state)
    if remember:
        cache.put(model_id, state, logits, project_scope=project_scope)
    return logits, state, hit


@torch.inference_mode()
def prove_prefix_equivalence(
    model,
    ids: torch.Tensor,
    split_at: int,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict:
    """Compare full forward logits against cached prefix continuation."""
    if not 1 <= split_at < ids.size(1) <= model.cfg.context:
        raise ValueError("Invalid split")
    cache = PrefixCache(PrefixPolicy(min_prefix_tokens=1))
    model_id = "test-weights"
    prefill_reusing_prefix(
        model, ids[:, :split_at], cache=cache,
        model_id=model_id, project_scope="test",
    )
    final, _, hit = prefill_reusing_prefix(
        model, ids, cache=cache,
        model_id=model_id, project_scope="test",
    )
    reference, _ = model(ids)
    delta = (final[:, -1, :] - reference[:, -1, :]).abs().max().item()
    return {
        "match": torch.allclose(
            final[:, -1, :], reference[:, -1, :], atol=atol, rtol=rtol
        ),
        "max_abs_error": delta,
        "kind": hit.kind,
        "saved_prefill_tokens": hit.saved_prefill_tokens,
    }
