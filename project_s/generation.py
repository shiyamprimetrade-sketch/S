from pathlib import Path
from html import escape
import re, shutil, json

def plan(requirement):
    if not isinstance(requirement,str) or not 0<len(requirement.strip())<=512: raise ValueError('Invalid requirement')
    registry=json.loads(Path(__file__).with_name('CAPABILITY-REGISTRY.json').read_text(encoding='utf-8'))
    matches=[k for k in (entry['id'] for entry in registry['capabilities']) if re.search(r'\b'+k+r'\b',requirement,re.I)]
    if len(matches)>1: raise ValueError('Clarify: calculator, website or inventory')
    if re.search(r'\b(delete|upload|network|shell|password|payment)\b',requirement,re.I): raise ValueError('Unsupported scope')
    from .intelligence import interpret
    advice=interpret(requirement)
    if not matches:
        if advice['capability'] not in [entry['id'] for entry in registry['capabilities']] or advice['score']<0.8:
            raise ValueError('Clarify: calculator, website or inventory')
        matches=[advice['capability']]
        selection='learned suggestion; explicit approval required'
    else:
        selection='explicit capability keyword'
    name=re.search(r'\b(?:named|called|for)\s+(.+)',requirement,re.I)
    return {'schema_version':1,'capability':matches[0],'requirement':requirement,'name':name.group(1).strip() if name else 'My Business','mechanism':'deterministic bounded generator','intent':advice,'selection':selection}

def generate(p,destination):
    d=Path(destination); d.mkdir(exist_ok=False)
    k=p['capability']
    if k in ('calculator','inventory'):
        target=d/(k+'.py'); shutil.copyfile(Path(__file__).with_name(k+'.py'),target)
    elif k=='website':
        target=d/'index.html'; name=escape(p['name']); target.write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{name}</title><style>body{{font:18px system-ui;margin:0;background:#101820;color:white}}header,main,footer{{padding:2rem;max-width:900px;margin:auto}}a{{color:#8ee}}section{{padding:1rem;border:1px solid #789;margin:1rem 0}}@media(max-width:600px){{header,main,footer{{padding:1rem}}}}</style><header><h1>{name}</h1><nav><a href="#services">Services</a> | <a href="#contact">Contact</a></nav></header><main><section id="services"><h2>Services</h2><p>Describe your products or services here.</p></section><section id="contact"><h2>Contact</h2><button id="details">Show information</button><p id="info" hidden>Contact details pending configuration.</p></section></main><footer>{name}</footer><script>document.getElementById("details").onclick=()=>{{document.getElementById("info").hidden=false}}</script></html>',encoding='utf-8')
    else: raise ValueError('Unsupported capability')
    return target
