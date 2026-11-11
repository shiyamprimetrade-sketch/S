"""Auditable offline review queue for local Ollama response supervision.

Teacher responses are never considered facts by default. This module tracks
explicit review decisions, preserves source/answer SHA-256 integrity and emits
eligible training examples only after an approving human action. It can reject
duplicate questions, answer changes and holdout contamination. It deliberately
does not automatically approve generated text or call network services.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from .teacher_data import (
    TeacherExample, audit_examples, digest, eligible_training_examples,
    load_examples, save_examples, review_example,
)

DECISIONS = ("approve", "reject", "defer")
CLAIM = re.compile(r"\b(?:always|never|guaranteed|100%|certainly)\b", re.I)
LINK = re.compile(r"https?://[^\s<>]+", re.I)


@dataclasses.dataclass(frozen=True)
class ReviewPolicy:
    require_review_notes: bool = True
    reject_holdout: bool = True
    reject_duplicate_questions: bool = True
    max_answer_bytes: int = 16000
    max_question_bytes: int = 4096
    max_assertive_claims: int = 20
    allow_unverified_links: bool = False
    forbid_empty_reviewers: bool = True

    def validate(self):
        if self.max_answer_bytes < 2 or self.max_question_bytes < 1:
            raise ValueError("Invalid response limits")
        if self.max_assertive_claims < 0:
            raise ValueError("Invalid claim threshold")


@dataclasses.dataclass(frozen=True)
class ReviewAction:
    identifier: str
    decision: str
    reviewer: str
    reason: str
    expected_answer_sha256: str
    expected_question_sha256: str

    def validate(self):
        if self.decision not in DECISIONS:
            raise ValueError("Unknown review decision")
        if not self.identifier or not self.reviewer.strip():
            raise ValueError("Reviewer identity and example ID required")
        if not self.expected_answer_sha256 or not self.expected_question_sha256:
            raise ValueError("Expected content fingerprints required")
        if not isinstance(self.reason, str):
            raise ValueError("Reason must be text")


@dataclasses.dataclass(frozen=True)
class ReviewEvent:
    identifier: str
    reviewer: str
    decision: str
    before_status: str
    after_status: str
    answer_sha256: str
    question_sha256: str
    reason: str

    def as_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ReviewOutcome:
    examples: list[TeacherExample]
    events: list[ReviewEvent]
    rejected_actions: list[dict]

    def audit(self):
        return {
            "examples": len(self.examples),
            "events": len(self.events),
            "rejected_actions": self.rejected_actions,
            "teacher_statistics": audit_examples(self.examples),
            "reviewers": dict(collections.Counter(x.reviewer for x in self.events)),
            "decisions": dict(collections.Counter(x.decision for x in self.events)),
            "note": "Review metadata is not a verification of factual correctness",
        }


def answer_flags(example: TeacherExample, policy: ReviewPolicy) -> list[str]:
    example.validate()
    policy.validate()
    flags = []
    if example.split != "train" and policy.reject_holdout:
        flags.append("not_train_split")
    if len(example.answer.encode("utf-8")) > policy.max_answer_bytes:
        flags.append("oversized_answer")
    if len(example.question.encode("utf-8")) > policy.max_question_bytes:
        flags.append("oversized_question")
    if len(CLAIM.findall(example.answer)) > policy.max_assertive_claims:
        flags.append("many_absolute_claims")
    if not policy.allow_unverified_links and LINK.search(example.answer):
        flags.append("unverified_external_link")
    return flags


def review_candidates(
    examples: Iterable[TeacherExample],
    policy: ReviewPolicy | None = None,
) -> list[dict]:
    policy = policy or ReviewPolicy()
    rows = []
    seen = set()
    for item in examples:
        flags = answer_flags(item, policy)
        if item.question_sha256 in seen and policy.reject_duplicate_questions:
            flags.append("duplicate_question")
        seen.add(item.question_sha256)
        rows.append({
            "identifier": item.identifier,
            "question_sha256": item.question_sha256,
            "answer_sha256": item.answer_sha256,
            "status": item.review_status,
            "requires_review": item.review_status == "pending",
            "flags": flags,
            "source": item.source,
            "permission": item.permission,
            "teacher_model": item.teacher_model,
            "category": item.category,
        })
    return rows


def apply_actions(
    examples: Iterable[TeacherExample],
    actions: Iterable[ReviewAction],
    policy: ReviewPolicy | None = None,
    *,
    strict: bool = True,
) -> ReviewOutcome:
    """Process an explicit review ledger; never auto-approve missing actions."""
    policy = policy or ReviewPolicy()
    policy.validate()
    by_id = {}
    order = []
    for item in examples:
        item.validate()
        if item.identifier in by_id:
            raise ValueError("Duplicate candidate identifier")
        by_id[item.identifier] = item
        order.append(item.identifier)
    seen_questions = set()
    for item in by_id.values():
        if item.question_sha256 in seen_questions and policy.reject_duplicate_questions:
            raise ValueError("Duplicate question content")
        seen_questions.add(item.question_sha256)

    events, rejected = [], []
    for action in actions:
        try:
            action.validate()
            original = by_id.get(action.identifier)
            if original is None:
                raise ValueError("Unknown example ID")
            if original.answer_sha256 != action.expected_answer_sha256:
                raise ValueError("Answer content changed since review")
            if original.question_sha256 != action.expected_question_sha256:
                raise ValueError("Question content changed since review")
            if action.decision in ("approve", "reject") and policy.require_review_notes:
                if not action.reason.strip():
                    raise ValueError("Reviewer reason required")
            if action.decision == "approve" and answer_flags(original, policy):
                raise ValueError("Candidate has unresolved review flags")
            if original.review_status != "pending":
                raise ValueError("Example was already decided")
            if action.decision == "approve":
                updated = review_example(original, approved=True, notes=action.reason)
            elif action.decision == "reject":
                updated = review_example(original, approved=False, notes=action.reason)
            else:
                updated = original
            by_id[action.identifier] = updated
            events.append(ReviewEvent(
                identifier=action.identifier, reviewer=action.reviewer,
                decision=action.decision,
                before_status=original.review_status,
                after_status=updated.review_status,
                answer_sha256=original.answer_sha256,
                question_sha256=original.question_sha256,
                reason=action.reason,
            ))
        except ValueError as exc:
            if strict:
                raise
            rejected.append({
                "identifier": action.identifier,
                "reason": type(exc).__name__ + ": " + str(exc),
            })
    return ReviewOutcome([by_id[key] for key in order], events, rejected)


def read_actions(path: Path) -> list[ReviewAction]:
    rows = []
    with path.open("r", encoding="utf-8") as input_file:
        for number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                item = ReviewAction(**json.loads(line))
                item.validate()
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            rows.append(item)
    return rows


def write_actions(path: Path, actions: Iterable[ReviewAction]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for action in actions:
            action.validate()
            output.write(json.dumps(dataclasses.asdict(action), ensure_ascii=False) + "\n")
            count += 1
    return count


def write_ledger(path: Path, events: Iterable[ReviewEvent]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for event in events:
            output.write(json.dumps(event.as_dict(), ensure_ascii=False) + "\n")


def read_ledger(path: Path) -> list[ReviewEvent]:
    results = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = ReviewEvent(**json.loads(line))
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            results.append(item)
    return results


def verify_ledger(events: list[ReviewEvent], examples: list[TeacherExample]) -> dict:
    final = {item.identifier: item for item in examples}
    issues = []
    seen_actions = set()
    for event in events:
        if event.identifier not in final:
            issues.append("unknown_reviewed_id")
            continue
        if event.identifier in seen_actions and event.after_status != "pending":
            issues.append("duplicate_final_action")
        seen_actions.add(event.identifier)
        item = final[event.identifier]
        if item.answer_sha256 != event.answer_sha256:
            issues.append("answer_hash_changed")
        if item.question_sha256 != event.question_sha256:
            issues.append("question_hash_changed")
        if event.after_status != "pending" and item.review_status != event.after_status:
            issues.append("final_status_mismatch")
    return {
        "valid": not issues,
        "issues": issues,
        "events": len(events),
        "reviewed_examples": len(seen_actions),
    }


def training_release(
    outcome: ReviewOutcome, *, require_at_least: int = 1,
) -> tuple[list[TeacherExample], dict]:
    eligible = eligible_training_examples(outcome.examples)
    if len(eligible) < require_at_least:
        raise ValueError("Insufficient reviewed training examples")
    if outcome.rejected_actions:
        raise ValueError("Unresolved review ledger errors")
    return eligible, {
        "eligible": len(eligible),
        "teacher_models": sorted({x.teacher_model for x in eligible}),
        "source_permissions": sorted({x.permission for x in eligible}),
        "question_fingerprints": sorted(x.question_sha256 for x in eligible),
        "review_event_count": len(outcome.events),
        "manual_review_required": True,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Review local Ollama training responses")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args(argv)
    outcome = apply_actions(
        load_examples(args.candidates), read_actions(args.actions)
    )
    save_examples(args.output, outcome.examples)
    write_ledger(args.ledger, outcome.events)
    print(json.dumps(outcome.audit(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
