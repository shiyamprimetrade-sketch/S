"""Standard-library bridge from S Continuum to its optional S-Brain service.

No PyTorch import or new dependencies inside Continuum. Explicit operator
enablement requires AXIOM_ENABLE_S_BRAIN=1 and selecting the model ID in the
existing chat UI. The service is localhost-only, no automatic fallback.
"""
from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL = "s-brain:experimental"
ENDPOINT = "http://127.0.0.1:8766"
ENABLE_VARIABLE = "AXIOM_ENABLE_S_BRAIN"


class BrainBridgeError(RuntimeError):
    pass


def enabled(env=None) -> bool:
    environment = os.environ if env is None else env
    return environment.get(ENABLE_VARIABLE) == "1"


def endpoint():
    # Intentionally non-configurable to prevent SSRF and public endpoints.
    return ENDPOINT


def _request(path: str, payload: dict | None = None):
    if path not in ("/health", "/generate"):
        raise ValueError("Unrecognized S-Brain endpoint")
    url = endpoint() + path
    if payload is None:
        return Request(url, headers={"Accept": "application/json"})
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"}
    )


def read_health(*, timeout: float = 1.5) -> dict:
    if not enabled():
        return {"ready": False, "enabled": False}
    try:
        with urlopen(_request("/health"), timeout=timeout) as reply:
            value = json.load(reply)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        return {
            "ready": False, "enabled": True,
            "error": type(exc).__name__,
        }
    if not isinstance(value, dict):
        return {"ready": False, "enabled": True, "error": "Invalid health object"}
    return {
        "ready": bool(value.get("ready") and value.get("independent")),
        "enabled": True,
        "model": value.get("name", MODEL),
        "context": value.get("context"),
    }


def available(*, timeout: float = 1.5) -> bool:
    return read_health(timeout=timeout).get("ready") is True


def generate(prompt: str, *, max_new_tokens: int = 128,
             timeout: float = 120.0) -> dict:
    if not enabled():
        raise BrainBridgeError("Independent S-Brain is disabled")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must contain text")
    if not 1 <= len(prompt.encode("utf-8")) <= 16000:
        raise ValueError("Prompt exceeds S-Brain byte limit")
    if not 1 <= max_new_tokens <= 512:
        raise ValueError("Invalid generation limit")
    body = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.8,
        "use_cache": True,
    }
    try:
        with urlopen(_request("/generate", body), timeout=timeout) as reply:
            value = json.load(reply)
    except HTTPError as exc:
        raise BrainBridgeError(f"S-Brain returned HTTP {exc.code}") from exc
    except (URLError, OSError, TimeoutError, ValueError) as exc:
        raise BrainBridgeError("S-Brain loopback service unavailable") from exc
    if not isinstance(value, dict) or not isinstance(value.get("response"), str):
        raise BrainBridgeError("Malformed S-Brain generation response")
    return {
        "response": value["response"],
        "model": MODEL,
        "usage": value.get("usage"),
        "experimental": True,
    }


def chat_text(prompt: str, *, timeout: float = 120.0) -> str:
    return generate(prompt, timeout=timeout)["response"]


def configuration_summary(env=None) -> dict:
    return {
        "name": MODEL,
        "enabled": enabled(env),
        "endpoint": ENDPOINT,
        "requires_trained_checkpoint": True,
        "requires_ollama": False,
        "training_in_chat": False,
        "supports_multiturn": False,
        "automatic_fallback": False,
        "public_internet": False,
    }
