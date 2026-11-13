"""Explicit bridge from S-Memory to the opt-in grounded identity module.

Identity records use tags ('identity', 'identity:<key>') and contain only the
value. Unstructured notes are never silently treated as identity claims.
"""
from __future__ import annotations

from .grounded_identity import Fact, GroundedAnswer, answer_with_context
from .memory_store import MemoryStore, validate_scope

IDENTITY_KEYS = frozenset(("name", "location", "occupation", "interests", "projects", "goals"))


def identity_from_memory(
    store: MemoryStore,
    question: str,
    *,
    scope: str,
    enabled: bool = False,
    max_records: int = 20,
) -> GroundedAnswer:
    """Read authorized scoped records only when identity retrieval is enabled."""
    validate_scope(scope)
    if not enabled:
        return answer_with_context(question, ())
    if not 1 <= max_records <= 20:
        raise ValueError("Identity record budget must be 1..20")
    facts = []
    # list_scope enforces expiry and project isolation. Avoid lexical retrieval
    # because 'Who am I?' does not necessarily overlap identity record text.
    for record in store.list_scope(scope):
        if len(facts) >= max_records:
            break
        if "identity" not in record.tags:
            continue
        keys = [
            tag.removeprefix("identity:")
            for tag in record.tags
            if tag.startswith("identity:") and tag.removeprefix("identity:") in IDENTITY_KEYS
        ]
        if len(keys) != 1 or record.consent not in ("explicit", "import-authorized"):
            continue
        facts.append(Fact(keys[0], record.content, record.source, consented=True))
    return answer_with_context(question, facts)
