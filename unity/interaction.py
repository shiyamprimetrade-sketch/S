"""Trusted adapter boundary; no physical action driver is installed here."""
import hashlib
import json
import uuid

def observe(image, analyzer):
    if not isinstance(image,bytes) or not image or len(image)>20_000_000:
        raise ValueError("invalid image bytes")
    findings=analyzer(image)
    if not isinstance(findings,dict):
        raise ValueError("invalid observation")
    return {"sha256":hashlib.sha256(image).hexdigest(),"findings":findings}

def interact(text, expected_window, *, authorized=False, driver, capture, analyzer, verify):
    """Only bounded text entry; callbacks must be trusted and operator supplied."""
    if authorized is not True:
        raise PermissionError("explicit authorization required")
    if not isinstance(text,str) or not text or len(text)>4096:
        raise ValueError("invalid text")
    if not isinstance(expected_window,str) or not expected_window.strip():
        raise ValueError("target window required")
    before=observe(capture(),analyzer)
    command={"protocol_version":"1.0","command_id":str(uuid.uuid4()),
             "target":"windows","action":"type_text",
             "arguments":{"text":text,"expected_window":expected_window}}
    receipt=driver(command)
    if not isinstance(receipt,dict) or receipt.get("command_id")!=command["command_id"] or receipt.get("status")!="completed":
        raise RuntimeError("action receipt rejected")
    after=observe(capture(),analyzer)
    verified=verify(before,after,command) is True
    return {"command_id":command["command_id"],"acknowledged":True,
            "verified":verified,"before":before,"after":after}
