"""Explicit, reversible integration plan for S Continuum.

This bridge does NOT modify continuum.engine or storage. It supplies an
optional isolated adapter and a readiness assessment that can be used by a
future opt-in UI. The operator must explicitly enable it after verifying
checkpoint, local service, data boundaries and user expectations.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Callable,Protocol

from .client import ClientConfig,SBrainClient,ServiceUnavailable
from .compat import ModelRegistry,BackendCapabilities,MockBackend


@dataclasses.dataclass(frozen=True)
class IntegrationPolicy:
    enabled: bool = False
    endpoint: str = "http://127.0.0.1:8766"
    identifier: str = "s-brain:experimental"
    require_health: bool = True
    allow_automatic_fallback: bool = False
    allow_multi_turn: bool = False
    max_new_tokens: int = 128
    model_disclaimer: str = (
        "Experimental byte-level next-token model, not instruction tuned"
    )

    def validate(self):
        if self.allow_automatic_fallback:
            raise ValueError("Silent fallback is prohibited")
        if self.allow_multi_turn:
            raise ValueError("Multi-turn S-Brain chat is not ready")
        if self.identifier != "s-brain:experimental":
            raise ValueError("Unsupported experimental model identifier")
        if not 1<=self.max_new_tokens<=512:
            raise ValueError("Invalid output limit")
        ClientConfig(base_url=self.endpoint).validate()

    @classmethod
    def load(cls,path:Path):
        value=json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value,dict):
            raise ValueError("Integration configuration must be a JSON object")
        unknown=set(value)-set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("Unknown integration policy keys")
        policy=cls(**value)
        policy.validate()
        return policy

    def save(self,path:Path):
        self.validate()
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(
            json.dumps(dataclasses.asdict(self),indent=2),
            encoding="utf-8"
        )


@dataclasses.dataclass(frozen=True)
class ReadinessFinding:
    name: str
    passed: bool
    details: str
    required: bool = True


@dataclasses.dataclass
class ReadinessReport:
    enabled: bool
    findings: list[ReadinessFinding]
    generated_at: float = dataclasses.field(default_factory=time.time)

    @property
    def ready(self):
        return self.enabled and all(
            item.passed for item in self.findings if item.required
        )

    def as_dict(self):
        return {
            "enabled":self.enabled,
            "ready":self.ready,
            "findings":[dataclasses.asdict(item) for item in self.findings],
            "generated_at":self.generated_at,
            "note":"Readiness does not prove model intelligence or user satisfaction",
        }


class ReadinessAuditor:
    def __init__(self,policy:IntegrationPolicy,
                 client_factory:Callable[...,SBrainClient]=SBrainClient):
        self.policy=policy
        self.client_factory=client_factory

    def check(self)->ReadinessReport:
        self.policy.validate()
        findings=[]
        findings.append(ReadinessFinding(
            "explicit_enablement",self.policy.enabled,
            "Operator controls the experimental model switch"
        ))
        if not self.policy.enabled:
            findings.append(ReadinessFinding(
                "no_live_connection",True,
                "Disabled configuration does not contact model service",
                required=False
            ))
            return ReadinessReport(False,findings)
        try:
            client=self.client_factory(ClientConfig(base_url=self.policy.endpoint))
            health=client.health()
            ok=bool(health.get("ready") and health.get("independent"))
            findings.append(ReadinessFinding(
                "local_model_health",ok,
                "Loopback independent service responded" if ok else "Invalid health response"
            ))
            if ok:
                context=int(health.get("context",0))
                findings.append(ReadinessFinding(
                    "context_advertised",context>0,
                    f"Model reports context capacity {context}"
                ))
        except (ValueError,ServiceUnavailable,OSError) as exc:
            findings.append(ReadinessFinding(
                "local_model_health",False,
                "Service unavailable: "+type(exc).__name__
            ))
        findings.append(ReadinessFinding(
            "default_fallback_off",not self.policy.allow_automatic_fallback,
            "Failures remain visible instead of silently switching models"
        ))
        findings.append(ReadinessFinding(
            "multi_turn_disabled",not self.policy.allow_multi_turn,
            "No unsupported conversation-history behavior"
        ))
        return ReadinessReport(self.policy.enabled,findings)


class ExperimentalBridge:
    """Safe bridge that accepts one prompt and returns explicit metadata."""

    def __init__(self,policy:IntegrationPolicy,
                 client:SBrainClient|None=None):
        policy.validate()
        if not policy.enabled:
            raise PermissionError("S-Brain integration is disabled")
        self.policy=policy
        self.client=client or SBrainClient(ClientConfig(base_url=policy.endpoint))

    def health(self):
        return self.client.health()

    def generate(self,prompt:str,history:list[dict]|None=None)->dict:
        if history:
            raise ValueError("S-Brain has no chat-history contract")
        if not isinstance(prompt,str) or not prompt.strip():
            raise ValueError("Prompt must be nonempty text")
        started=time.perf_counter()
        result=self.client.generate(
            prompt,max_new_tokens=self.policy.max_new_tokens,
            temperature=0.8,use_cache=True
        )
        if not isinstance(result.get("response"),str):
            raise ValueError("Model returned invalid text")
        return {
            "role":"assistant",
            "content":result["response"],
            "model":self.policy.identifier,
            "experimental":True,
            "independent":True,
            "elapsed_ms":round(1000*(time.perf_counter()-started)),
            "disclaimer":self.policy.model_disclaimer,
            "usage":result.get("usage"),
        }

    def stream(self,prompt:str):
        if not isinstance(prompt,str) or not prompt:
            raise ValueError("Prompt must be nonempty text")
        yield from self.client.stream(
            prompt,max_new_tokens=self.policy.max_new_tokens
        )


class OptInModelRouter:
    """Controlled routing without replacing Ollama or auto-fallback.

    Existing S can continue routing all model IDs to its current implementation.
    A future plugin may call this adapter for precisely the experimental ID.
    """

    def __init__(self,bridge:ExperimentalBridge|None=None):
        self.bridge=bridge

    def known_models(self):
        return ["s-brain:experimental"] if self.bridge else []

    def dispatch(self,model:str,prompt:str,history=None):
        if model!="s-brain:experimental":
            raise KeyError("Model ID does not belong to S-Brain router")
        if self.bridge is None:
            raise PermissionError("S-Brain was not explicitly enabled")
        return self.bridge.generate(prompt,history)

    def ready(self):
        if self.bridge is None:
            return {"enabled":False,"models":[]}
        return {
            "enabled":True,
            "models":self.known_models(),
            "automatic_fallback":False,
        }


def disabled_by_default_config(path:Path)->dict:
    """Create an opt-in template, but NEVER activate it."""
    policy=IntegrationPolicy()
    policy.save(path)
    return dataclasses.asdict(policy)


def check_local_port_conflict(main_port:int=8765,
                              brain_port:int=8766)->bool:
    if not (1<=main_port<=65535 and 1<=brain_port<=65535):
        raise ValueError("Invalid ports")
    return main_port!=brain_port


def explain_integration_status(policy:IntegrationPolicy)->dict:
    policy.validate()
    return {
        "enabled":policy.enabled,
        "backend":"independent S-Brain",
        "current_S_server_unchanged":True,
        "brain_local_url":policy.endpoint,
        "expected_model_id":policy.identifier,
        "ready_for_multi_turn":False,
        "ready_for_production":False,
        "warning":policy.model_disclaimer,
    }


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description="S-Brain opt-in integration readiness")
    p.add_argument("--config",type=Path,default=Path("s_brain/local_integration.json"))
    p.add_argument("--create-disabled-template",action="store_true")
    a=p.parse_args(argv)
    if a.create_disabled_template:
        print(json.dumps(disabled_by_default_config(a.config),indent=2))
        return 0
    policy=IntegrationPolicy.load(a.config)
    report=ReadinessAuditor(policy).check()
    print(json.dumps(report.as_dict(),indent=2))
    return 0 if report.ready else 2


if __name__=="__main__":
    raise SystemExit(main())
