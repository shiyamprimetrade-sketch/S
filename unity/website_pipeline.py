"""Bounded offline website delivery pipeline."""
import argparse
import json
from pathlib import Path
from .planning import architecture
from .generation import generate_website
from .website_acceptance import verify_website
from .website_repair import repair_website
from .engineering_memory import EngineeringMemory

def build_website(workspace,name,heading,description,*,approved=False,repair_approved=False):
    plan=architecture("website",name,["heading","description"])
    if approved is not True:
        return {"status":"awaiting_authorization","plan":plan}
    root=Path(workspace)
    root.mkdir(parents=True,exist_ok=False)
    site=root/"site"
    generated=generate_website(site,name,heading,description,approved=True)
    evidence=verify_website(site,name,heading,description)
    repair=None
    if evidence["status"]!="verified" or evidence.get("sha256")!=generated["sha256"]:
        if evidence["status"]=="verified":
            return {"status":"rejected","reason":"digest mismatch"}
        repair=repair_website(site,root/"repaired",name,heading,description,approved=repair_approved)
        if repair["status"]!="repaired":
            return {"status":"not_delivered","repair":repair}
        site=root/"repaired"
        evidence=repair["after"]
    memory=EngineeringMemory(root/"memory.db")
    ident=memory.remember(site,name,heading,description)
    recalled=memory.recall(ident)
    if not recalled["currently_verified"] or recalled["sha256"]!=evidence["sha256"]:
        return {"status":"rejected","reason":"artifact changed"}
    return {"status":"delivered","plan":plan,"artifact":str(site/"index.html"),
            "evidence":evidence,"memory_id":ident,"repair":repair}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("workspace")
    p.add_argument("name")
    p.add_argument("heading")
    p.add_argument("description")
    p.add_argument("--approve",action="store_true")
    p.add_argument("--approve-repair",action="store_true")
    a=p.parse_args()
    result=build_website(a.workspace,a.name,a.heading,a.description,approved=a.approve,repair_approved=a.approve_repair)
    print(json.dumps(result,indent=2))
    return 0 if result["status"] in ("delivered","awaiting_authorization") else 1

if __name__=="__main__":
    raise SystemExit(main())
