"""Compose trusted source components; never execute generated code."""
from dataclasses import dataclass
import re
from hashlib import sha256

@dataclass(frozen=True)
class Component:
    name: str
    source: str
    dependencies: tuple = ()

def compose(registry, requested):
    if not isinstance(requested,(list,tuple)) or not requested:
        raise ValueError("components required")
    active=set()
    done=set()
    ordered=[]
    def visit(name):
        if not isinstance(name,str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}",name):
            raise ValueError("invalid component name")
        if name in active:
            raise ValueError("dependency cycle")
        if name in done:
            return
        if name not in registry:
            raise ValueError("unknown component")
        component=registry[name]
        if not isinstance(component,Component) or component.name!=name:
            raise ValueError("invalid component")
        if not isinstance(component.source,str) or len(component.source)>65536:
            raise ValueError("invalid source")
        active.add(name)
        for dep in component.dependencies:
            visit(dep)
        active.remove(name)
        done.add(name)
        ordered.append(component)
    for name in requested:
        visit(name)
    source="\n\n".join(c.source for c in ordered)+"\n"
    return {"components":[c.name for c in ordered],"source":source,
            "sha256":sha256(source.encode("utf-8")).hexdigest(),
            "authorized":False,"verified":False}
