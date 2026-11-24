"""Natural-language entry point for supported arithmetic requirements."""
import argparse
import json
import re
from .cli import calculate

NUMBER=r"[+-]?\d{1,12}(?:\.\d{1,6})?"
RULES=[
    (rf"add ({NUMBER}) (?:and|to) ({NUMBER})", "+", False),
    (rf"subtract ({NUMBER}) from ({NUMBER})", "-", True),
    (rf"multiply ({NUMBER}) by ({NUMBER})", "*", False),
    (rf"(?:calculate|compute|what is) ({NUMBER})\s*([+*-])\s*({NUMBER})", None, False),
]

def interpret(requirement):
    if not isinstance(requirement,str) or len(requirement)>1000:
        raise ValueError("invalid requirement")
    normalized=" ".join(requirement.strip().lower().split()).rstrip("?")
    for pattern,op,reverse in RULES:
        match=re.fullmatch(pattern,normalized)
        if match:
            if op is None:
                a,op,b=match.groups()
            else:
                a,b=match.groups()
            if reverse:
                a,b=b,a
            return {"operation":"calculator","expression":f"{a} {op} {b}"}
    return None

def request(requirement,workspace,*,approved=False):
    plan=interpret(requirement)
    if plan is None:
        return {"status":"unsupported","requirement":requirement}
    if approved is not True:
        return {"status":"awaiting_authorization","plan":plan}
    return calculate(plan["expression"],workspace,approved=True)

def main():
    parser=argparse.ArgumentParser(description="S Genesis bounded natural language terminal")
    parser.add_argument("requirement")
    parser.add_argument("--workspace",default="unity-runtime")
    parser.add_argument("--approve",action="store_true")
    args=parser.parse_args()
    result=request(args.requirement,args.workspace,approved=args.approve)
    print(json.dumps(result,indent=2))
    return 0 if result["status"] in ("delivered","awaiting_authorization") else 1

if __name__=="__main__":
    raise SystemExit(main())
