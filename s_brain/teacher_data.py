"""Temporary Ollama teacher for independent S-Brain text-supervision datasets.

The teacher is consulted ONLY during explicitly invoked offline data creation.
Generated responses become supervised target text; these are NOT teacher logits
and this is NOT KL-divergence logit distillation. S-Brain later trains its own
weights using cross-entropy; inference requires no Ollama service.

By default this module can only contact the loopback Ollama HTTP endpoint.
Do not send proprietary, personal, or unlicensed text to an external service.
Generated records require a model name, source consent, hashes and review.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Callable, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,119}$")
MAX_INPUT_BYTES = 16_000
MAX_OUTPUT_BYTES = 65_536


class TeacherUnavailable(RuntimeError):
    pass


class TeacherResponseError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class TeacherSettings:
    model: str = "qwen2.5:1.5b"
    base_url: str = OLLAMA_ENDPOINT
    timeout_seconds: float = 120.0
    temperature: float = 0.25
    seed: int = 42
    max_output_tokens: int = 384
    system_prompt: str = (
        "You are preparing factual, concise educational training examples. "
        "If information is uncertain, clearly label uncertainty. "
        "Do not include fabricated sources or claim you ran tools. "
        "Respond to the question as clearly and accurately as possible."
    )

    def validate(self) -> None:
        if not MODEL_PATTERN.fullmatch(self.model):
            raise ValueError("Invalid local teacher model name")
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in ("localhost", "127.0.0.1")
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Teacher must use loopback HTTP Ollama endpoint")
        if parsed.port not in (None, 11434):
            raise ValueError("Unexpected Ollama port")
        if not 0 < self.timeout_seconds <= 1800:
            raise ValueError("Invalid timeout")
        if not 0 <= self.temperature <= 2 or not math.isfinite(self.temperature):
            raise ValueError("Invalid temperature")
        if not 1 <= self.max_output_tokens <= 8192:
            raise ValueError("Invalid teacher response limit")
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("System prompt required")


@dataclasses.dataclass(frozen=True)
class TeacherQuestion:
    identifier: str
    question: str
    category: str
    source: str
    permission: str
    split: str = "train"

    def validate(self):
        if not self.identifier or not self.identifier.strip():
            raise ValueError("Question ID required")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("Question text required")
        if len(self.question.encode("utf-8")) > MAX_INPUT_BYTES:
            raise ValueError("Teacher input too large")
        if not self.category.strip() or not self.source.strip():
            raise ValueError("Question category and source required")
        if not self.permission.strip():
            raise ValueError("Explicit permission or license required")
        if self.split not in ("train", "validation", "test"):
            raise ValueError("Invalid question split")


@dataclasses.dataclass(frozen=True)
class TeacherExample:
    identifier: str
    question: str
    answer: str
    category: str
    source: str
    permission: str
    split: str
    teacher_model: str
    question_sha256: str
    answer_sha256: str
    generation_seconds: float
    review_status: str = "pending"
    review_notes: str = ""

    def validate(self):
        if self.review_status not in ("pending", "approved", "rejected"):
            raise ValueError("Invalid review status")
        if self.split not in ("train", "validation", "test"):
            raise ValueError("Invalid data split")
        if not self.teacher_model or not self.permission:
            raise ValueError("Teacher model and permission required")
        if not self.answer or len(self.answer.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise ValueError("Answer missing or oversized")
        if digest(self.question) != self.question_sha256:
            raise ValueError("Question hash mismatch")
        if digest(self.answer) != self.answer_sha256:
            raise ValueError("Answer hash mismatch")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "TeacherExample":
        allowed = set(cls.__dataclass_fields__)
        if set(value) != allowed:
            raise ValueError("Teacher example fields do not match schema")
        result = cls(**value)
        result.validate()
        return result


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def question_digest(question: TeacherQuestion) -> str:
    return hashlib.sha256(
        json.dumps(dataclasses.asdict(question), sort_keys=True).encode("utf-8")
    ).hexdigest()


def ollama_payload(question: TeacherQuestion, settings: TeacherSettings) -> dict:
    question.validate()
    settings.validate()
    return {
        "model": settings.model,
        "stream": False,
        "messages": [
            {"role": "system", "content": settings.system_prompt},
            {"role": "user", "content": question.question},
        ],
        "options": {
            "temperature": settings.temperature,
            "seed": settings.seed,
            "num_predict": settings.max_output_tokens,
        },
    }


def validate_teacher_response(raw: object) -> str:
    if not isinstance(raw, dict):
        raise TeacherResponseError("Teacher result must be JSON object")
    value = raw.get("message")
    if not isinstance(value, dict):
        raise TeacherResponseError("Missing Ollama message")
    answer = value.get("content")
    if not isinstance(answer, str) or not answer.strip():
        raise TeacherResponseError("Empty teacher answer")
    if len(answer.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise TeacherResponseError("Teacher answer exceeded limit")
    return answer.strip()


def send_ollama(question: TeacherQuestion, settings: TeacherSettings) -> str:
    """Use local Ollama for one example; never silently consult a cloud model."""
    payload = ollama_payload(question, settings)
    request = Request(
        settings.base_url.rstrip("/") + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=settings.timeout_seconds) as response:
            raw = response.read(MAX_OUTPUT_BYTES + 4096)
    except HTTPError as exc:
        raise TeacherUnavailable(f"Ollama HTTP {exc.code}") from exc
    except (URLError, OSError, TimeoutError) as exc:
        raise TeacherUnavailable("Local Ollama service unavailable") from exc
    if len(raw) > MAX_OUTPUT_BYTES + 4095:
        raise TeacherResponseError("Oversized teacher JSON response")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TeacherResponseError("Invalid teacher JSON") from exc
    return validate_teacher_response(value)


def create_example(
    question: TeacherQuestion,
    settings: TeacherSettings,
    *,
    generator: Callable[[TeacherQuestion, TeacherSettings], str] = send_ollama,
    clock: Callable[[], float] = time.perf_counter,
) -> TeacherExample:
    question.validate()
    settings.validate()
    began = clock()
    answer = generator(question, settings)
    duration = max(0.0, clock() - began)
    if not isinstance(answer, str):
        raise TeacherResponseError("Teacher generator did not return text")
    example = TeacherExample(
        identifier=question.identifier, question=question.question,
        answer=answer.strip(), category=question.category,
        source=question.source, permission=question.permission,
        split=question.split, teacher_model=settings.model,
        question_sha256=digest(question.question),
        answer_sha256=digest(answer.strip()),
        generation_seconds=duration,
    )
    example.validate()
    return example


def load_questions(path: Path) -> list[TeacherQuestion]:
    results = []
    seen = set()
    with path.open("r", encoding="utf-8") as file:
        for lineno, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                question = TeacherQuestion(**obj)
                question.validate()
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid question at line {lineno}: {exc}") from exc
            if question.identifier in seen:
                raise ValueError(f"Duplicate question identifier: {question.identifier}")
            seen.add(question.identifier)
            results.append(question)
    return results


def save_questions(path: Path, items: list[TeacherQuestion]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = set()
    with path.open("w", encoding="utf-8") as file:
        for item in items:
            item.validate()
            if item.identifier in ids:
                raise ValueError("Duplicate question ID")
            ids.add(item.identifier)
            file.write(json.dumps(dataclasses.asdict(item), ensure_ascii=False) + "\n")
    return len(ids)


def load_examples(path: Path) -> list[TeacherExample]:
    result = []
    seen = set()
    with path.open("r", encoding="utf-8") as file:
        for lineno, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                item = TeacherExample.from_dict(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid example at line {lineno}: {exc}") from exc
            if item.identifier in seen:
                raise ValueError(f"Duplicate teacher example: {item.identifier}")
            seen.add(item.identifier)
            result.append(item)
    return result


def save_examples(path: Path, examples: list[TeacherExample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    identifiers = set()
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            example.validate()
            if example.identifier in identifiers:
                raise ValueError("Duplicate generated example")
            identifiers.add(example.identifier)
            file.write(json.dumps(example.as_dict(), ensure_ascii=False) + "\n")
    return len(identifiers)


def generate_dataset(
    questions: list[TeacherQuestion],
    settings: TeacherSettings,
    generator: Callable[[TeacherQuestion, TeacherSettings], str] = send_ollama,
    *,
    allowed_splits: tuple[str, ...] = ("train",),
) -> tuple[list[TeacherExample], dict]:
    """Generate only explicitly approved training splits, never blind tests."""
    results = []
    rejected = []
    for item in questions:
        if item.split not in allowed_splits:
            rejected.append(item.identifier)
            continue
        results.append(create_example(item, settings, generator=generator))
    return results, {
        "generated": len(results),
        "skipped_holdouts": len(rejected),
        "skipped_ids": rejected,
        "teacher": settings.model,
        "all_pending_human_review": True,
    }


def review_example(item: TeacherExample, *, approved: bool,
                   notes: str = "") -> TeacherExample:
    if len(notes) > 4000:
        raise ValueError("Review notes too long")
    next_item = dataclasses.replace(
        item,
        review_status="approved" if approved else "rejected",
        review_notes=notes,
    )
    next_item.validate()
    return next_item


def eligible_training_examples(examples: list[TeacherExample]) -> list[TeacherExample]:
    approved = []
    seen_questions = set()
    for item in examples:
        item.validate()
        if item.review_status != "approved" or item.split != "train":
            continue
        if item.question_sha256 in seen_questions:
            continue
        seen_questions.add(item.question_sha256)
        approved.append(item)
    return approved


def audit_examples(examples: list[TeacherExample]) -> dict:
    from collections import Counter
    return {
        "count": len(examples),
        "status": dict(Counter(item.review_status for item in examples)),
        "splits": dict(Counter(item.split for item in examples)),
        "categories": dict(Counter(item.category for item in examples)),
        "teacher_models": dict(Counter(item.teacher_model for item in examples)),
        "approved_train": len(eligible_training_examples(examples)),
    }


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate explicitly approved candidate training answers with local Ollama"
    )
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 10000:
        parser.error("Invalid example limit")
    questions = load_questions(args.questions)
    results, report = generate_dataset(
        questions[:args.limit], TeacherSettings(model=args.model)
    )
    save_examples(args.output, results)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
