"""Assemble bounded projects from trusted generators; integrity is not semantic verification."""
import json
from hashlib import sha256
from pathlib import Path
from .calculator_app import generate_calculator
from .inventory_app import generate_inventory
from .responsive_site import generate as generate_site

def build_project(root,kind,*,options=None,approved=False):
    options={} if options is None else dict(options)
    if kind not in ("calculator","inventory","website"):
        raise ValueError("unsupported project kind")
    if set(options)!=({"name","heading","description","sections"} if kind=="website" else set()):
        raise ValueError("invalid project options")
    if approved is not True:
        return {"status":"awaiting_authorization","kind":kind}
    root=Path(root)
    root.mkdir(parents=True,exist_ok=False)
    src=root/"src"
    if kind=="calculator":
        generate_calculator(src,approved=True)
        entry="src/calculator.py"
    elif kind=="inventory":
        generate_inventory(src,approved=True)
        entry="src/inventory.py"
    else:
        generate_site(src,approved=True,**options)
        entry="src/index.html"
    (root/"README.txt").write_text("S Evolution project: "+kind+"\nEntry: "+entry+"\n",encoding="utf-8")
    files={p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}
    manifest={"version":1,"kind":kind,"entry":entry,"files":files}
    (root/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2),encoding="utf-8")
    return {"status":"generated","manifest":manifest,"verified":False}

def verify_project(root):
    root=Path(root)
    if root.is_symlink():
        return False
    try:
        paths=list(root.rglob("*"))
        if any(p.is_symlink() for p in paths):
            return False
        manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
        actual={p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file() and p!=root/"manifest.json"}
        return manifest["version"]==1 and manifest["entry"] in actual and manifest["files"]==actual
    except (OSError,ValueError,KeyError,TypeError):
        return False
