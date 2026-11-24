"""Local developer preview CLI; calculator only, no arbitrary code execution."""
import argparse
import json
import operator
import re
import secrets
import uuid
from decimal import Decimal, localcontext
from pathlib import Path
from .development import develop

OPS={"+":operator.add,"-":operator.sub,"*":operator.mul}
PATTERN=re.compile(r"\s*([+-]?\d{1,12}(?:\.\d{1,6})?)\s*([+*-])\s*([+-]?\d{1,12}(?:\.\d{1,6})?)\s*")

def calculate(expression, workspace, *, approved=False):
    match=PATTERN.fullmatch(expression)
    if not match:
        raise ValueError("supported: two decimal numbers with +, - or *")
    a,op,b=match.groups()
    with localcontext() as ctx:
        ctx.prec=50
        expected=OPS[op](Decimal(a),Decimal(b))
    if approved is not True:
        return {"status":"denied","expression":expression}
    root=Path(workspace).resolve()
    root.mkdir(parents=True,exist_ok=True)
    task=dict(version=3,id=uuid.uuid4().hex,operation="calculator",args=dict(a=a,op=op,b=b))
    def verify(t,r,root,db):
        return Decimal(r["data"]["value"])==expected
    return develop(expression,planner=lambda _:task,
                   authorize=lambda req,t:t==task,key=secrets.token_bytes(32),
                   root=root,db=root/"alpha.db",journal_db=root/"omega.db",
                   allowed=("calculator",),verifiers={"calculator":verify},max_repairs=0)

def main(argv=None):
    parser=argparse.ArgumentParser(description="Project S Unity developer preview")
    sub=parser.add_subparsers(dest="command",required=True)
    calc=sub.add_parser("calculate")
    calc.add_argument("expression")
    calc.add_argument("--workspace",default="unity-runtime")
    calc.add_argument("--approve",action="store_true")
    sub.add_parser("status")
    args=parser.parse_args(argv)
    if args.command=="status":
        result={"product":"Project S Unity","stage":"developer preview",
                "supported":["calculator"],"physical_control":False,
                "general_code_generation":False}
    else:
        try:
            result=calculate(args.expression,args.workspace,approved=args.approve)
        except (ValueError,OSError,ImportError) as exc:
            result={"status":"error","error":type(exc).__name__}
    print(json.dumps(result,indent=2))
    return 0 if result.get("status") not in ("error","denied","failed") else 1

if __name__=="__main__":
    raise SystemExit(main())
