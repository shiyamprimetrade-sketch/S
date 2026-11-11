"""Non-invasive compatibility contracts for eventual S Continuum integration.

This module does not patch the existing S server or silently replace Ollama.
It defines an explicit opt-in adapter that can be called by S when the owner
chooses to enable the experimental brain. It also provides offline mock
adapters for tests and a capability registry to make limitations explicit.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass,field
from typing import Callable,Protocol

from .client import ClientConfig,SBrainClient,ServiceUnavailable


class ChatBackend(Protocol):
    def respond(self,prompt: str,history: list[dict] | None = None) -> str:
        ...


@dataclass(frozen=True)
class BackendCapabilities:
    identifier: str
    model_family: str
    independent_weights: bool
    training_supported: bool
    tool_calls: bool = False
    image_input: bool = False
    audio_input: bool = False
    context_tokens: int | None = None
    streaming: bool = False
    external_cloud_required: bool = False
    verified_capabilities: tuple[str,...] = ()

    def display(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


S_BRAIN_CAPABILITIES = BackendCapabilities(
    identifier="s-brain:experimental",
    model_family="byte-level decoder-only transformer",
    independent_weights=True,
    training_supported=True,
    streaming=True,
    verified_capabilities=(
        "next-token logits","weight checkpoint loading",
        "standalone HTTP generation",
    ),
)


@dataclass
class AdapterResult:
    response: str
    model: str
    latency_ms: float
    independent: bool
    metadata: dict = field(default_factory=dict)


class IndependentSBrainAdapter:
    """Opt-in synchronous client, never a background learning process."""

    def __init__(self,client: SBrainClient | None = None,
                 max_new_tokens: int = 128):
        self.client = client or SBrainClient()
        if not 1 <= max_new_tokens <= 512:
            raise ValueError("Invalid max_new_tokens")
        self.max_new_tokens = max_new_tokens

    def health(self):
        return self.client.health()

    def respond(self,prompt: str,history: list[dict] | None = None) -> str:
        return self.complete(prompt,history).response

    def complete(self,prompt: str,history: list[dict] | None = None) -> AdapterResult:
        if history:
            # Explicitly prevent the false impression of conversation support:
            # this experimental model is not chat fine-tuned and only takes one prompt.
            raise ValueError("S-Brain adapter does not yet accept multi-turn history")
        if not isinstance(prompt,str) or not prompt.strip():
            raise ValueError("Prompt must be nonempty string")
        start = time.perf_counter()
        raw = self.client.generate(
            prompt,max_new_tokens=self.max_new_tokens,
            temperature=0.8,use_cache=True
        )
        elapsed = 1000*(time.perf_counter()-start)
        return AdapterResult(
            response=raw["response"],
            model=raw.get("model","s-brain:experimental"),
            latency_ms=elapsed,
            independent=True,
            metadata={"source":"local-s-brain","usage":raw.get("usage")},
        )


class MockBackend:
    """Deterministic testing backend; not a neural network."""

    def __init__(self,response="TEST RESPONSE"):
        self.value = response
        self.calls = []

    def respond(self,prompt: str,history: list[dict] | None = None) -> str:
        self.calls.append((prompt,history))
        return self.value


class ModelRegistry:
    """Explicitly configured adapters; never infer a fallback from failures."""

    def __init__(self):
        self.backends: dict[str,ChatBackend] = {}
        self.capabilities: dict[str,BackendCapabilities] = {}

    def register(self,name: str,backend: ChatBackend,
                 capabilities: BackendCapabilities):
        if not name or name in self.backends:
            raise ValueError("Model name empty or already registered")
        if name != capabilities.identifier:
            raise ValueError("Capability ID mismatch")
        self.backends[name] = backend
        self.capabilities[name] = capabilities

    def names(self) -> list[str]:
        return sorted(self.backends)

    def lookup(self,name: str) -> ChatBackend:
        if name not in self.backends:
            raise KeyError(f"Unknown model: {name}")
        return self.backends[name]

    def describe(self,name: str) -> dict:
        self.lookup(name)
        return self.capabilities[name].display()

    def ask(self,name: str,prompt: str,
            history: list[dict] | None = None) -> str:
        return self.lookup(name).respond(prompt,history)

    def readiness(self) -> dict:
        return {
            "models":[self.describe(name) for name in self.names()],
            "default_model":None,
            "auto_fallback":False,
            "integration_state":"opt-in; not installed in existing S server",
        }


def create_experimental_registry(client: SBrainClient | None = None):
    registry = ModelRegistry()
    registry.register(
        S_BRAIN_CAPABILITIES.identifier,
        IndependentSBrainAdapter(client),
        S_BRAIN_CAPABILITIES,
    )
    return registry


def validate_chat_history(messages: list[dict],limit: int = 20) -> list[dict]:
    """Type-validate history without passing untrusted roles as system prompts."""
    if not isinstance(messages,list) or len(messages)>limit:
        raise ValueError("History must be a bounded list")
    cleaned = []
    for item in messages:
        if not isinstance(item,dict):
            raise ValueError("History item must be an object")
        role,text = item.get("role"),item.get("content")
        if role not in ("user","assistant") or not isinstance(text,str):
            raise ValueError("Unsupported history role/content")
        if len(text)>4000:
            raise ValueError("History message too long")
        cleaned.append({"role":role,"content":text})
    return cleaned


def format_legacy_chat_response(result: AdapterResult) -> dict:
    """Minimal adapter output; persistence belongs to existing S storage."""
    if not isinstance(result.response,str):
        raise ValueError("Response must be string")
    return {
        "role":"assistant",
        "content":result.response,
        "model":result.model,
        "elapsed_ms":round(result.latency_ms),
        "source":result.metadata.get("source","unknown"),
    }


def health_to_capability(health: dict) -> BackendCapabilities:
    """Report only features that the live service explicitly advertises."""
    if not health.get("ready") or not health.get("independent"):
        raise ValueError("Unexpected health response")
    return BackendCapabilities(
        identifier=str(health.get("name","s-brain:experimental")),
        model_family="byte-level decoder-only transformer",
        independent_weights=True,
        training_supported=True,
        streaming=True,
        context_tokens=int(health["context"]),
        verified_capabilities=("local model loaded",),
    )


def adapter_smoke_test(client: SBrainClient,
                       prompt: str = "S",max_new_tokens:int = 3) -> dict:
    """Live opt-in smoke test without altering S settings or database."""
    health = client.health()
    if not health.get("ready"):
        raise ServiceUnavailable("Brain is not ready")
    result = client.generate(prompt,max_new_tokens=max_new_tokens,seed=42)
    return {
        "ready":True,
        "model":result.get("model"),
        "output_is_text":isinstance(result.get("response"),str),
        "expected_source":"loopback-only independent S-Brain",
        "persistent_changes":False,
    }
