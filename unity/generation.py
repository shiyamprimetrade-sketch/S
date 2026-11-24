"""Bounded template generation, not arbitrary learned code synthesis."""
from html import escape
from pathlib import Path
from hashlib import sha256
from .planning import architecture

def generate_website(root,name,heading,description,*,approved=False):
    plan=architecture("website",name,["heading","description"])
    for value in (heading,description):
        if not isinstance(value,str) or not value.strip() or len(value)>4000:
            raise ValueError("invalid website content")
    if approved is not True:
        return {"status":"awaiting_authorization","plan":plan}
    data=("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
          "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
          f"<title>{escape(name)}</title></head><body><main>"
          f"<h1>{escape(heading)}</h1><p>{escape(description)}</p>"
          "</main></body></html>\n").encode("utf-8")
    root=Path(root)
    root.mkdir(parents=True,exist_ok=False)
    path=root/"index.html"
    with path.open("xb") as f:
        f.write(data)
    return {"status":"generated","file":"index.html","sha256":sha256(data).hexdigest(),
            "bytes":len(data),"verified":False}
