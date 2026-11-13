"""Opt-in, evidence-grounded personal context for S-Brain.

No personal information is bundled or persisted by this module.
The caller supplies authorized facts; the answer returns provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class Fact:
    key: str
    value: str
    source: str
    consented: bool = False

    def __post_init__(self) -> None:
        if not self.key.strip() or not self.source.strip():
            raise ValueError("Fact key and source must be nonempty")


@dataclass(frozen=True)
class GroundedAnswer:
    text: str
    intent: str
    sources: tuple[str, ...]
    used_keys: tuple[str, ...]


_IDENTITY_PATTERNS = (
    re.compile(r"^who\s+am\s+i\s*[?.!]*$", re.I),
    re.compile(r"^what\s+do\s+you\s+know\s+about\s+me\s*[?.!]*$", re.I),
    re.compile(r"^tell\s+me\s+about\s+myself\s*[?.!]*$", re.I),
)
_ALLOWED = ("name", "location", "occupation", "interests", "projects", "goals")


def detect_intent(message: str) -> str:
    normalized = " ".join(message.strip().split())
    if any(pattern.fullmatch(normalized) for pattern in _IDENTITY_PATTERNS):
        return "personal_identity"
    return "general"


def answer_with_context(message: str, facts: Iterable[Fact]) -> GroundedAnswer:
    """Return only explicitly consented facts; never infer unknown attributes."""
    intent = detect_intent(message)
    if intent != "personal_identity":
        return GroundedAnswer(
            "This module handles personal identity questions only.",
            intent, (), (),
        )

    chosen: dict[str, Fact] = {}
    for fact in facts:
        if fact.consented and fact.key in _ALLOWED and fact.value.strip():
            # Keep first authorized assertion to avoid silent source overwrites.
            chosen.setdefault(fact.key, fact)

    if not chosen:
        return GroundedAnswer(
            "I don't have authorized information to identify you.",
            intent, (), (),
        )

    clauses = []
    if "name" in chosen:
        clauses.append(f"Your name is {chosen['name'].value}.")
    for key, prefix in (
        ("location", "Your stated location is"),
        ("occupation", "Your stated occupation is"),
        ("interests", "Your interests include"),
        ("projects", "Your projects include"),
        ("goals", "Your goals include"),
    ):
        if key in chosen:
            clauses.append(f"{prefix} {chosen[key].value}.")

    keys = tuple(key for key in _ALLOWED if key in chosen)
    sources = tuple(dict.fromkeys(chosen[key].source for key in keys))
    return GroundedAnswer(" ".join(clauses), intent, sources, keys)
