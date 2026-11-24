"""Plan bounded software projects without executing or writing files."""
import re

def architecture(kind,name,features):
    if not isinstance(name,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,63}",name):
        raise ValueError("invalid project name")
    contracts={
        "website":({"heading","description"},["index.html"],["heading and description rendered","no external dependencies"]),
        "calculator":({"addition","subtraction","multiplication"},["calculator.py","test_calculator.py"],["signed decimal arithmetic","reject invalid input"]),
        "inventory app":({"register","adjust","search"},["inventory.py","test_inventory.py"],["persist stock","reject negative stock","search registered items"])
    }
    if kind not in contracts:
        return {"status":"unsupported","authorized":False}
    supported,files,checks=contracts[kind]
    if not isinstance(features,list) or not features or any(not isinstance(f,str) for f in features):
        raise ValueError("explicit features required")
    unknown=set(features)-supported
    if unknown:
        return {"status":"unsupported_features","features":sorted(unknown),"authorized":False}
    return {"version":1,"status":"planned","kind":kind,"name":name,
            "features":sorted(set(features)),"files":files,"dependencies":[],
            "acceptance":checks,"authorized":False,"implemented":False}
