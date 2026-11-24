"""Probe the existing experimental S-Brain service without executing output."""
import argparse
import hashlib
import json
import time
from continuum import s_brain_bridge

def probe(prompt,*,approved=False):
    if not isinstance(prompt,str) or not prompt.strip() or len(prompt.encode("utf-8"))>4000:
        raise ValueError("invalid prompt")
    if approved is not True:
        return {"status":"awaiting_authorization"}
    state=s_brain_bridge.read_health()
    if state.get("ready") is not True:
        return {"status":"unavailable","health":state}
    start=time.monotonic()
    try:
        result=s_brain_bridge.generate(prompt,max_new_tokens=128,timeout=60)
    except s_brain_bridge.BrainBridgeError as exc:
        return {"status":"failed","error":str(exc)}
    response=result["response"]
    if not response.strip() or len(response.encode("utf-8"))>65536:
        return {"status":"rejected"}
    return {"status":"response_received","provider":result["model"],
            "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
            "response_sha256":hashlib.sha256(response.encode()).hexdigest(),
            "elapsed_ms":round((time.monotonic()-start)*1000),
            "response":response,"verified":False,"executed":False}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--approve",action="store_true")
    a=p.parse_args()
    result=probe(a.prompt,approved=a.approve)
    print(json.dumps(result,indent=2))
    return 0 if result["status"]=="response_received" else 1

if __name__=="__main__":
    raise SystemExit(main())
