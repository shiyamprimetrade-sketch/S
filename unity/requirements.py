"""Deterministic specification extraction; never implies unsupported execution."""
import re
from .genesis import interpret

def specify(text):
    if not isinstance(text,str) or not text.strip() or len(text)>4000:
        raise ValueError("requirement must contain 1–4000 characters")
    plan=interpret(text) if len(text)<=1000 else None
    if plan:
        return {"version":1,"status":"ready","kind":"calculator","plan":plan,
                "acceptance":["independent arithmetic equality"],"authorized":False}
    match=re.fullmatch(r"(?:build|create|make) (?:a |an )?(website|calculator|inventory app)(?: (?:for|called) (.+))?",text.strip(),re.I)
    if match:
        kind=match.group(1).lower()
        name=match.group(2)
        return {"version":1,"status":"needs_clarification","kind":kind,
                "name":name,"questions":["What features are required?","What inputs and outputs are expected?","What acceptance examples must pass?"],
                "authorized":False,"execution_supported":False}
    return {"version":1,"status":"unsupported","kind":None,"authorized":False,
            "questions":["Describe the desired software, features and expected behavior."]}
