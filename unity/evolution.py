"""Interactive terminal; explicit approval required for each development action."""
import json
import shlex
from pathlib import Path
from .genesis import request
from .project_builder import build_project
from .evolution_requirements import extract
from .website_pipeline import build_website

def dispatch(line,workspace,*,approved=False):
    parts=shlex.split(line)
    if not parts:
        return {"status":"empty"}
    if parts==["status"]:
        return {"status":"ready","capabilities":["arithmetic","static website","responsive website","calculator project","inventory project"],"general_programming":False}
    if parts in (["build","calculator"],["build","inventory"]):
        return build_project(workspace,parts[1],approved=approved)
    if parts[:2]==["build","website"]:
        if len(parts)<7 or (len(parts)-5)%2:
            raise ValueError("build website NAME HEADING DESCRIPTION SECTION TITLE_TEXT [SECTION TITLE_TEXT ...]")
        options=dict(name=parts[2],heading=parts[3],description=parts[4],sections=list(zip(parts[5::2],parts[6::2])))
        return build_project(workspace,"website",options=options,approved=approved)
    if parts[0]=="website" and len(parts)==4:
        return build_website(Path(workspace),*parts[1:],approved=approved)
    spec=extract(line)
    if spec.get('kind')=='website' and spec['status']=='ready':
        return build_website(workspace,spec['name'],spec['heading'],spec['description'],approved=approved)
    if spec['status']=='needs_clarification':
        return spec
    return request(line,workspace,approved=approved)

def main():
    from uuid import uuid4
    while True:
        try:
            line=input("S> ")
            if line.strip() in ("exit","quit"):
                return 0
            root=Path("evolution-projects")/uuid4().hex
            result=dispatch(line,root)
            print(json.dumps(result,indent=2))
            if result["status"]=="awaiting_authorization" and input("Authorize this task? [y/N] ").strip().lower()=="y":
                print(json.dumps(dispatch(line,root,approved=True),indent=2))
        except (EOFError,KeyboardInterrupt):
            return 0
        except (ValueError,OSError,RuntimeError,ImportError) as exc:
            print(json.dumps({"status":"error","type":type(exc).__name__}))

if __name__=="__main__":
    raise SystemExit(main())
