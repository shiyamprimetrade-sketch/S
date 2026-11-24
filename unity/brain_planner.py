"""Experimental model proposals are data, never executable instructions."""
import json
from continuum import s_brain_bridge

def plan(requirement,*,enabled=False,health=None,generate=None):
    if not isinstance(requirement,str) or not requirement.strip() or len(requirement.encode("utf-8"))>4000:
        raise ValueError("invalid requirement")
    if enabled is not True:
        return {"status":"disabled","authorized":False}
    health=health or s_brain_bridge.read_health
    generate=generate or s_brain_bridge.generate
    state=health()
    if state.get("ready") is not True:
        return {"status":"unavailable","authorized":False}
    prompt='Return only JSON with keys kind and features. kind must be calculator or inventory. Features must be a list of supported names. Calculator: addition, subtraction, multiplication, division. Inventory: register, adjust, search. Requirement: '+json.dumps(requirement)
    try:
        raw=generate(prompt,max_new_tokens=256,timeout=30)["response"]
        if not isinstance(raw,str) or len(raw)>8192:
            raise ValueError("invalid response")
        value=json.loads(raw)
        if not isinstance(value,dict) or set(value)!={"kind","features"}:
            raise ValueError("invalid schema")
        allowed={"calculator":{"addition","subtraction","multiplication","division"},"inventory":{"register","adjust","search"}}
        kind=value["kind"]
        features=value["features"]
        if not isinstance(kind,str) or kind not in allowed or not isinstance(features,list) or not features or any(not isinstance(f,str) or f not in allowed[kind] for f in features):
            raise ValueError("unsupported proposal")
    except (ValueError,TypeError,KeyError,s_brain_bridge.BrainBridgeError):
        return {"status":"rejected","authorized":False}
    return {"status":"proposed","kind":kind,"features":sorted(set(features)),"authorized":False,"verified":False,"provider":"s-brain:experimental"}
