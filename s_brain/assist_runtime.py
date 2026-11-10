"""Opt-in S-Brain assisted inference bound to the existing local HTTP runtime.

This deliberately separates four concerns: request validation, memory
snapshotting, independent model generation, and optional allowlisted tools.
No user-generated tool text is granted execution privileges. A local client
may request a read-only calculator, text statistic, or JSON extraction only
when the host has explicitly enabled this experimental endpoint at startup.
State changes, filesystem access, arbitrary network and subprocesses remain
outside this API. No Ollama fallback or automatic memory persistence.

Source-verification is performed against the EXACT memory excerpt snapshot
injected into the prompt, then checked for expiry/revision changes against
the local memory database. An absent or unsupported citation results in a
refusal, not a fabricated source-grounding success.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Callable, Mapping

from .memory_store import MemoryStore, validate_scope
from .rag_inference import (
    RetrievalOptions, EvidenceExcerpt, AnswerPacket, PromptAssembly,
    assemble_query, generate_with_memory, model_response,
)
from .grounded_answers import VerificationPolicy, verify_and_gate
from .tool_orchestration import (
    ToolIntent, ToolAuthorization, ToolOrchestrator,
    serialize_result_as_evidence, redact_trace,
)
from .tools_control import ToolController, ToolPolicy


class AssistantDisabled(PermissionError):
    pass


class AssistantRequestError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class AssistLimits:
    max_request_bytes: int = 12000
    max_query_bytes: int = 4096
    max_new_tokens: int = 128
    max_tool_intents: int = 3
    max_output_bytes: int = 65536
    memory_budget: int = 1024
    max_context_bytes: int = 16000
    minimum_model_context: int = 128

    def validate(self):
        values=(
            self.max_request_bytes,self.max_query_bytes,self.max_new_tokens,
            self.max_tool_intents,self.max_output_bytes,self.memory_budget,
            self.max_context_bytes,self.minimum_model_context,
        )
        if any(type(x) is not int or x < 1 for x in values):
            raise ValueError("Assistant limits must be positive integers")
        if self.max_query_bytes >= self.max_request_bytes:
            raise ValueError("Query must fit in request budget")
        if self.max_new_tokens>4096 or self.max_tool_intents>8:
            raise ValueError("Assistant work budget too large")


@dataclasses.dataclass(frozen=True)
class AssistRequest:
    question: str
    scope: str = "S"
    memory: bool = False
    cite_sources: bool = False
    max_new_tokens: int = 32
    seed: int = 7
    tool_intents: tuple[dict, ...] = ()

    @classmethod
    def parse(cls, value:object, limits:AssistLimits)->"AssistRequest":
        limits.validate()
        if not isinstance(value,dict):
            raise AssistantRequestError("JSON object required")
        allowed=set(cls.__dataclass_fields__)
        if set(value)-allowed:
            raise AssistantRequestError("Unknown assistant request fields")
        if "question" not in value:
            raise AssistantRequestError("Question required")
        if "tool_intents" in value and not isinstance(value["tool_intents"],list):
            raise AssistantRequestError("Tool intents must be a list")
        data=dict(value)
        data["tool_intents"]=tuple(data.get("tool_intents",()))
        try:
            parsed=cls(**data)
            parsed.validate(limits)
        except (TypeError,ValueError) as exc:
            raise AssistantRequestError(str(exc)) from exc
        if len(json.dumps(value,ensure_ascii=False).encode("utf-8"))>limits.max_request_bytes:
            raise AssistantRequestError("Assistant request too large")
        return parsed

    def validate(self,limits:AssistLimits):
        validate_scope(self.scope)
        if not isinstance(self.question,str) or not self.question.strip():
            raise ValueError("Question must be text")
        if len(self.question.encode("utf-8"))>limits.max_query_bytes:
            raise ValueError("Question exceeds maximum size")
        if type(self.memory) is not bool or type(self.cite_sources) is not bool:
            raise ValueError("Options must be booleans")
        if type(self.max_new_tokens) is not int or not 1<=self.max_new_tokens<=limits.max_new_tokens:
            raise ValueError("Generation token limit exceeded")
        if type(self.seed) is not int or not 0<=self.seed<2**32:
            raise ValueError("Seed must be a nonnegative 32-bit integer")
        if len(self.tool_intents)>limits.max_tool_intents:
            raise ValueError("Too many tool intents")
        for item in self.tool_intents:
            if not isinstance(item,dict):
                raise ValueError("Tool intents must be objects")
            if set(item)!={"id","name","arguments"}:
                raise ValueError("Tool intent must specify id, name and arguments")
            if not isinstance(item["id"],str) or not item["id"]:
                raise ValueError("Tool intent ID required")
            if not isinstance(item["name"],str) or not item["name"]:
                raise ValueError("Tool name required")
            if not isinstance(item["arguments"],dict):
                raise ValueError("Tool arguments must be object")


@dataclasses.dataclass(frozen=True)
class AssistConfiguration:
    enabled: bool = False
    memory_enabled: bool = False
    tools_enabled: bool = False
    verification_enabled: bool = True
    fail_closed_on_citation_error: bool = True
    allowed_scope: str = "S"
    read_only_tools: tuple[str,...] = ("calculator","text_stats","json_extract")
    limits: AssistLimits = dataclasses.field(default_factory=AssistLimits)

    def validate(self):
        validate_scope(self.allowed_scope)
        self.limits.validate()
        if self.memory_enabled and not self.enabled:
            raise ValueError("Memory requires assistant endpoint")
        if self.tools_enabled and not self.enabled:
            raise ValueError("Tools require assistant endpoint")
        if set(self.read_only_tools)-{"calculator","text_stats","json_extract"}:
            raise ValueError("Only read-only local tools permitted over HTTP")


@dataclasses.dataclass
class AssistAudit:
    request_sha256: str
    memory_requested: bool
    memory_excerpts: int
    tools_requested: int
    tools_executed: int
    tools_denied: int
    grounding_mode: str
    verified: bool
    failed_closed: bool
    elapsed_seconds: float

    def as_dict(self):
        return dataclasses.asdict(self)


def request_hash(request:AssistRequest)->str:
    return hashlib.sha256(json.dumps(
        {"question":request.question,"scope":request.scope,
         "memory":request.memory,"cite_sources":request.cite_sources,
         "seed":request.seed,"max_new_tokens":request.max_new_tokens,
         "intents":[{"id":x["id"],"name":x["name"]} for x in request.tool_intents]},
        sort_keys=True,
    ).encode("utf-8")).hexdigest()


def make_tool_controller(configuration:AssistConfiguration)->ToolController:
    # NO memory writes, no shell and no network. Explicit enablement
    # still required for every request; this controller is ephemeral.
    policy=ToolPolicy(
        max_calls=configuration.limits.max_tool_intents,
        permitted_tools=configuration.read_only_tools,
        allow_memory_write=False,
        max_batch=configuration.limits.max_tool_intents,
    )
    controller=ToolController(scope=configuration.allowed_scope,policy=policy)
    controller.enable()
    return controller


def parse_read_only_intents(request:AssistRequest)->list[ToolIntent]:
    result=[]
    for value in request.tool_intents:
        intent=ToolIntent(
            identifier=value["id"],name=value["name"],
            arguments=value["arguments"],scope=request.scope,
            requested_by="application",
        )
        intent.validate()
        result.append(intent)
    if len({x.identifier for x in result})!=len(result):
        raise AssistantRequestError("Duplicate tool intent IDs")
    return result


def run_read_only_tools(
    request:AssistRequest,
    configuration:AssistConfiguration,
)->tuple[list[str],dict]:
    if not request.tool_intents:
        return [],{"requested":[],"executed":[],"denied":[]}
    if not configuration.tools_enabled:
        raise AssistantDisabled("Assistant tools disabled by server policy")
    controller=make_tool_controller(configuration)
    intents=parse_read_only_intents(request)
    # This endpoint is loopback-host controlled and offers only read-only
    # tools. The CLI flag is the host's authorization to expose this list;
    # it does not establish per-user identity or authorize writes.
    approvals={
        intent.identifier:ToolAuthorization(
            tool_name=intent.name,scope=intent.scope,
            intent_id=intent.identifier,approved_by="host-startup-opt-in",
            enabled=True,
        )
        for intent in intents if intent.name in configuration.read_only_tools
    }
    trace=ToolOrchestrator(
        controller,max_intents=configuration.limits.max_tool_intents
    ).run(intents,approvals=approvals)
    evidence=[
        serialize_result_as_evidence(row,max_bytes=2048)
        for row in trace.executed if row.success
    ]
    return evidence,redact_trace(trace)


def evidence_snapshot(
    memory:MemoryStore|None,
    request:AssistRequest,
    configuration:AssistConfiguration,
    *,
    context:int,
)->PromptAssembly:
    if request.scope!=configuration.allowed_scope:
        raise PermissionError("Cross-project assistant scope")
    if request.memory and not configuration.memory_enabled:
        raise AssistantDisabled("Memory unavailable without startup opt-in")
    if request.memory and memory is None:
        raise AssistantDisabled("No local memory store")
    options=RetrievalOptions(
        enabled=request.memory,scope=request.scope,
        top_k=3,evidence_budget=configuration.limits.memory_budget,
        max_prompt_bytes=configuration.limits.max_context_bytes,
        require_sources=request.memory and request.cite_sources,
    )
    return assemble_query(
        memory,request.question,options,model_context=context,
    )


def append_tool_evidence(prompt:str,tool_results:list[str],context:int)->str:
    if not tool_results:
        return prompt
    body="\nBEGIN UNTRUSTED TOOL OUTPUT\n"+"\n".join(tool_results)+"\nEND UNTRUSTED TOOL OUTPUT\n"
    # Append only to a prompt that has room for the additional bytes.
    if len((prompt+body).encode("utf-8"))>context:
        raise ValueError("Tool output would truncate model context")
    return prompt+body


def response_envelope(
    packet:AnswerPacket,request:AssistRequest,
    tool_trace:dict,audit:AssistAudit,
)->dict:
    result={
        "model":packet.model_name,
        "response":packet.answer,
        "memory_mode":packet.mode,
        "verified_sources":packet.cited_sources,
        "source_verification_passed":packet.verified,
        "rejection_reason":packet.rejection_reason,
        "evidence":packet.evidence,
        "tool_trace":tool_trace,
        "audit":audit.as_dict(),
        "usage":packet.usage,
        "independent_inference":True,
        "no_ollama_fallback":True,
        "live_memory_write_allowed":False,
    }
    return result


class AssistantEngine:
    """Opt-in host entrypoint; use the same generation callback for all modes."""

    def __init__(
        self,model,*,configuration:AssistConfiguration|None=None,
        memory:MemoryStore|None=None,
        generator:Callable|None=None,
    ):
        self.model=model
        self.configuration=configuration or AssistConfiguration()
        self.configuration.validate()
        self.memory=memory
        self.generator=generator or model_response
        self._lock=threading.RLock()

    def health(self):
        return {
            "assistant_enabled":self.configuration.enabled,
            "memory_enabled":self.configuration.memory_enabled,
            "tools_enabled":self.configuration.tools_enabled,
            "scope":self.configuration.allowed_scope,
            "tool_names":list(self.configuration.read_only_tools)
            if self.configuration.tools_enabled else [],
            "model_context":self.model.cfg.context,
            "independent":True,
            "auto_capture_enabled":False,
        }

    def respond(self,request:AssistRequest)->dict:
        config=self.configuration
        if not config.enabled:
            raise AssistantDisabled("Experimental assistant endpoint disabled")
        request.validate(config.limits)
        if request.scope!=config.allowed_scope:
            raise PermissionError("Cross-project assistant scope rejected")
        began=time.perf_counter()
        with self._lock:
            snapshot=evidence_snapshot(
                self.memory,request,config,
                context=self.model.cfg.context,
            )
            tool_results,trace=run_read_only_tools(request,config)
            prompt=append_tool_evidence(
                snapshot.prompt,tool_results,self.model.cfg.context
            )
            result=self.generator(
                self.model,prompt,
                max_new_tokens=request.max_new_tokens,seed=request.seed,
            )
            if not isinstance(result,dict) or not isinstance(result.get("response"),str):
                raise TypeError("Independent generator returned invalid result")
            packet=AnswerPacket(
                question_sha256=hashlib.sha256(request.question.encode()).hexdigest(),
                answer=result["response"],mode=snapshot.mode,
                cited_sources=[],
                allowed_source_ids=[x.memory_id for x in snapshot.included],
                model_name=result.get("model","s-brain:experimental"),
                usage=result.get("usage") or {},
                evidence=snapshot.as_metadata(),
            )
            if request.cite_sources and config.verification_enabled:
                packet=verify_and_gate(
                    packet,list(snapshot.included),store=self.memory,
                    scope=request.scope,
                    policy=VerificationPolicy(require_current_memory=True),
                )
            elif snapshot.included:
                # Unchecked model citations may be present in raw text, but
                # never claim they are verified or publish verified sources.
                packet.verified=False
            audit=AssistAudit(
                request_sha256=request_hash(request),
                memory_requested=request.memory,
                memory_excerpts=len(snapshot.included),
                tools_requested=len(request.tool_intents),
                tools_executed=len(trace["execution_audits"]),
                tools_denied=len(trace["denied"]),
                grounding_mode=packet.mode,
                verified=packet.verified,
                failed_closed=packet.rejection_reason is not None,
                elapsed_seconds=time.perf_counter()-began,
            )
            payload=response_envelope(packet,request,trace,audit)
            if len(json.dumps(payload,ensure_ascii=False,allow_nan=False).encode("utf-8"))>config.limits.max_output_bytes:
                raise ValueError("Assistant result exceeds output size")
            return payload
