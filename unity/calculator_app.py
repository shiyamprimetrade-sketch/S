"""Generate a bounded standalone Python calculator; no generated code execution."""
from pathlib import Path
from hashlib import sha256
from .composition import Component, compose

ENGINE = '''from decimal import Decimal, localcontext
import re

def calculate(a, op, b):
    for value in (a,b):
        if not re.fullmatch(r"[+-]?[0-9]{1,12}(?:\\.[0-9]{1,6})?",value):
            raise ValueError("Invalid number")
    with localcontext() as ctx:
        ctx.prec=50
        x,y=Decimal(a),Decimal(b)
        if op=="+": return x+y
        if op=="-": return x-y
        if op=="*": return x*y
        if op=="/":
            if not y: raise ValueError("Division by zero")
            return x/y
        raise ValueError("Invalid operation")
'''
INTERFACE = '''def main():
    while True:
        try:
            line=input("Calculator (a op b, or quit): ")
            if line.strip()=="quit": return
            a,op,b=line.split()
            print(calculate(a,op,b))
        except (EOFError,KeyboardInterrupt):
            return
        except ValueError as exc:
            print("Error:",exc)

if __name__=="__main__":
    main()
'''
def generate_calculator(root,*,approved=False):
    result=compose({"engine":Component("engine",ENGINE),"interface":Component("interface",INTERFACE,("engine",))},["interface"])
    if approved is not True:
        return {"status":"awaiting_authorization","components":result["components"]}
    root=Path(root)
    root.mkdir(parents=True,exist_ok=False)
    data=result["source"].encode("utf-8")
    with (root/"calculator.py").open("xb") as f:
        f.write(data)
    return {"status":"generated","artifact":str(root/"calculator.py"),"sha256":sha256(data).hexdigest(),"verified":False}
