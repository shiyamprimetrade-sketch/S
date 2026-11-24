"""Regenerate a trusted project in a fresh directory; never execute suspect source."""
from pathlib import Path
from .project_builder import build_project, verify_project

def repair_project(original,destination,kind,*,options=None,accept,approved=False):
    original=Path(original).resolve()
    destination=Path(destination).resolve()
    if not original.is_dir():
        raise ValueError("original project missing")
    if destination==original or original in destination.parents or destination in original.parents:
        raise ValueError("repair destination overlaps original")
    intact=verify_project(original)
    if intact and accept(original) is True:
        return {"status":"no_repair_needed","attempts":0}
    diagnosis="integrity_failure" if not intact else "functional_failure"
    if approved is not True:
        return {"status":"awaiting_authorization","diagnosis":diagnosis,"attempts":0}
    build_project(destination,kind,options=options,approved=True)
    passed=verify_project(destination) and accept(destination) is True
    return {"status":"repaired" if passed else "repair_failed","diagnosis":diagnosis,
            "attempts":1,"destination":str(destination),"verified":passed}
