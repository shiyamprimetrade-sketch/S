"""Host-authorized, bounded S-Tools orchestration.

Model output proposes a JSON intent but never grants authorization.
The operator-controlled host must separately approve the specific tool and
scope before a call. No recursive tool loops or arbitrary execution.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Iterable

from .tools_control import (
    ToolController,ToolCall,ToolPermissionError,ToolSchemaError,ToolLimitError,
)
from .memory_store import validate_scope


@dataclasses.dataclass(frozen=True)
class ToolIntent:
    identifier:str
    name:str
    arguments:dict
    scope:str
    requested_by:str="model"
    rationale:str=""

    def validate(self):
        if not self.identifier or len(self.identifier)>128:
            raise ValueError("Invalid intent identifier")
        validate_scope(self.scope)
        if self.requested_by not in ("model","operator","application"):
            raise ValueError("Unknown tool requester")
        if not isinstance(self.arguments,dict):
            raise ValueError("Tool arguments must be object")
        if not isinstance(self.rationale,str) or len(self.rationale)>2048:
            raise ValueError("Invalid rationale")


@dataclasses.dataclass(frozen=True)
class ToolAuthorization:
    tool_name:str
    scope:str
    intent_id:str
    approved_by:str
    enabled:bool
    approval_token:str|None=None

    def validate(self):
        if not self.tool_name or not self.intent_id:
            raise ValueError("Specific intent authorization required")
        validate_scope(self.scope)
        if not self.approved_by:
            raise PermissionError("Host approval missing")


@dataclasses.dataclass
class ExecutedTool:
    intent_id:str
    tool:str
    success:bool
    result:dict|None
    failure_code:str|None
    audit_id:str
    elapsed_seconds:float

    def public(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ToolTrace:
    scope:str
    requested:list[str]
    executed:list[ExecutedTool]
    denied:list[dict]
    disabled:bool=True

    def public(self):
        return {
            "scope":self.scope,"requested":self.requested,
            "executed":[x.public() for x in self.executed],
            "denied":self.denied,"disabled_by_default":self.disabled,
        }


def parse_intent(text:str,*,scope:str)->ToolIntent:
    """Parse strictly structured model text, not execute it."""
    validate_scope(scope)
    if not isinstance(text,str) or len(text.encode("utf-8"))>8192:
        raise ValueError("Intent size outside range")
    stripped=text.strip()
    if stripped.startswith("~~~json") and stripped.endswith("~~~"):
        stripped=stripped[7:-3].strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        raise ValueError("JSON object required")
    try:
        value=json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON intent") from exc
    if not isinstance(value,dict):
        raise ValueError("Intent must be object")
    required={"id","name","arguments"}
    allowed=required|{"rationale"}
    if not required<=set(value) or set(value)-allowed:
        raise ValueError("Tool intent fields invalid")
    result=ToolIntent(
        identifier=value["id"],name=value["name"],
        arguments=value["arguments"],scope=scope,
        rationale=value.get("rationale",""),
    )
    result.validate()
    return result


def evaluate_authorization(
    intent:ToolIntent,
    permit:ToolAuthorization|None,
    controller:ToolController,
)->tuple[bool,str]:
    intent.validate()
    if not controller.enabled:
        return False,"tool_session_disabled"
    if intent.scope!=controller.scope:
        return False,"cross_project_scope"
    if permit is None or not permit.enabled:
        return False,"host_approval_required"
    permit.validate()
    if permit.tool_name!=intent.name:
        return False,"tool_name_mismatch"
    if permit.scope!=intent.scope:
        return False,"permission_scope_mismatch"
    if permit.intent_id!=intent.identifier:
        return False,"intent_identifier_mismatch"
    if intent.name not in controller.policy.permitted_tools:
        return False,"tool_not_allowlisted"
    if intent.name in ("memory_remember","memory_forget"):
        if not permit.approval_token:
            return False,"one_use_write_token_required"
    return True,"authorized"


def execute_intent(
    intent:ToolIntent,
    controller:ToolController,
    *,authorization:ToolAuthorization|None=None,
)->ExecutedTool:
    authorized,reason=evaluate_authorization(intent,authorization,controller)
    if not authorized:
        raise ToolPermissionError(reason)
    result=controller.invoke(ToolCall(
        name=intent.name,arguments=intent.arguments,
        scope=intent.scope,approval=authorization.approval_token,
    ))
    return ExecutedTool(
        intent_id=intent.identifier,tool=intent.name,
        success=result.success,result=result.output,
        failure_code=result.error_code,audit_id=result.audit_id,
        elapsed_seconds=result.elapsed_seconds,
    )


class ToolOrchestrator:
    """One bounded turn; no autonomous tool recursion."""

    def __init__(self,controller:ToolController,*,max_intents:int=4):
        if not 1<=max_intents<=16:
            raise ValueError("Intent limit outside range")
        self.controller=controller
        self.max_intents=max_intents

    def run(
        self,intents:Iterable[ToolIntent],
        *,approvals:dict[str,ToolAuthorization]|None=None,
    )->ToolTrace:
        supplied=list(intents)
        if len(supplied)>self.max_intents:
            raise ToolLimitError("Too many tool proposals")
        if len({x.identifier for x in supplied})!=len(supplied):
            raise ValueError("Duplicate intent ID")
        granted=approvals or {}
        trace=ToolTrace(
            scope=self.controller.scope,
            requested=[x.identifier for x in supplied],
            executed=[],denied=[],disabled=not self.controller.enabled,
        )
        for intent in supplied:
            try:
                result=execute_intent(
                    intent,self.controller,
                    authorization=granted.get(intent.identifier),
                )
                trace.executed.append(result)
            except (ToolPermissionError,ToolSchemaError,ToolLimitError,
                    ValueError,TypeError) as exc:
                trace.denied.append({
                    "intent_id":intent.identifier,
                    "tool":intent.name,"reason":type(exc).__name__,
                })
        return trace


def serialize_result_as_evidence(outcome:ExecutedTool,*,max_bytes:int=4096)->str:
    if not outcome.success or outcome.result is None:
        return "[tool-result-unavailable]"
    if max_bytes<32:
        raise ValueError("Context budget too small")
    raw=json.dumps(outcome.result,ensure_ascii=False,allow_nan=False)
    if len(raw.encode("utf-8"))>max_bytes:
        return "[tool-result-exceeded-context-budget]"
    return (
        f"[tool:{outcome.tool} audit_id={outcome.audit_id}]\n"
        +raw+"\n[/tool-data]"
    )


def tool_success_metrics(trace:ToolTrace)->dict:
    successes=sum(x.success for x in trace.executed)
    total=len(trace.requested)
    return {
        "intent_count":total,"executed":len(trace.executed),
        "denied":len(trace.denied),"successful":successes,
        "execution_rate":len(trace.executed)/max(1,total),
        "success_rate":successes/max(1,total),
    }


def redact_trace(trace:ToolTrace)->dict:
    return {
        "scope":trace.scope,
        "requested_ids":list(trace.requested),
        "execution_audits":[{
            "intent_id":row.intent_id,"tool":row.tool,
            "audit_id":row.audit_id,"success":row.success,
            "error_code":row.failure_code,
        } for row in trace.executed],
        "denied":list(trace.denied),"raw_results_excluded":True,
    }


def intent_fingerprint(intent:ToolIntent)->str:
    intent.validate()
    return hashlib.sha256(json.dumps({
        "id":intent.identifier,"name":intent.name,
        "arguments":intent.arguments,"scope":intent.scope,
    },sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def verify_no_tool_side_effects(trace:ToolTrace)->bool:
    write_names={"memory_remember","memory_forget"}
    return all(
        row.tool not in write_names or not row.success
        for row in trace.executed
    )
