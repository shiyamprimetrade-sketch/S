"""Bounded responsive website composition."""
from html import escape
from pathlib import Path
from .composition import Component, compose
from .planning import architecture

CSS="""*{box-sizing:border-box}body{margin:0;background:#101827;color:#f1f5f9;font:1rem/1.6 system-ui,sans-serif}header,main,footer{max-width:72rem;margin:auto;padding:1.5rem}nav{display:flex;gap:1rem;flex-wrap:wrap}a{color:#93c5fd}h1{font-size:clamp(2rem,6vw,4rem)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,16rem),1fr));gap:1rem}article{background:#1e293b;padding:1.5rem;border-radius:1rem}footer{border-top:1px solid #475569}"""
def render(name,heading,description,sections):
    architecture("website",name,["heading","description"])
    if not isinstance(sections,list) or not 1<=len(sections)<=8:
        raise ValueError("one to eight sections required")
    for value in [heading,description]+[v for pair in sections if isinstance(pair,(tuple,list)) for v in pair]:
        if not isinstance(value,str) or not value.strip() or len(value)>2000:
            raise ValueError("invalid content")
    if any(not isinstance(p,(tuple,list)) or len(p)!=2 for p in sections):
        raise ValueError("invalid section")
    nav="".join(f'<a href="#section-{i}">{escape(t)}</a>' for i,(t,d) in enumerate(sections))
    cards="".join(f'<article id="section-{i}"><h2>{escape(t)}</h2><p>{escape(d)}</p></article>' for i,(t,d) in enumerate(sections))
    registry={
        "head":Component("head",f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{escape(name)}</title><style>{CSS}</style></head><body>'),
        "hero":Component("hero",f'<header><nav aria-label="Sections">{nav}</nav><h1>{escape(heading)}</h1><p>{escape(description)}</p></header>',("head",)),
        "sections":Component("sections",f'<main class="cards">{cards}</main>',("hero",)),
        "footer":Component("footer",f'<footer>{escape(name)}</footer></body></html>',("sections",))}
    return compose(registry,["footer"])

def generate(root,name,heading,description,sections,*,approved=False):
    result=render(name,heading,description,sections)
    if approved is not True:
        return {"status":"awaiting_authorization","components":result["components"]}
    root=Path(root)
    root.mkdir(parents=True,exist_ok=False)
    with (root/"index.html").open("xb") as f:
        f.write(result["source"].encode("utf-8"))
    return {"status":"generated","sha256":result["sha256"],"verified":False}
