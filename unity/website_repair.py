"""Repair a rejected static website in a new directory, preserving original."""
from pathlib import Path
from .generation import generate_website
from .website_acceptance import verify_website

def repair_website(original,destination,name,heading,description,*,approved=False):
    original=Path(original)
    destination=Path(destination)
    before=verify_website(original,name,heading,description)
    if before["status"]=="verified":
        return {"status":"no_repair_needed","evidence":before}
    if approved is not True:
        return {"status":"awaiting_repair_authorization","evidence":before}
    if destination.resolve()==original.resolve() or original.resolve() in destination.resolve().parents:
        raise ValueError("repair destination must be separate")
    generated=generate_website(destination,name,heading,description,approved=True)
    after=verify_website(destination,name,heading,description)
    ok=after["status"]=="verified" and after["sha256"]==generated["sha256"]
    return {"status":"repaired" if ok else "repair_failed","attempts":1,
            "before":before,"after":after,"original_preserved":True}
