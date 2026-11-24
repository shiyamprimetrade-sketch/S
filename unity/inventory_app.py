"""Generate a standalone offline inventory CLI."""
from pathlib import Path
from .composition import Component, compose

STORE = '''import sqlite3
from contextlib import closing
from pathlib import Path
import argparse
import json

def operate(db, action, sku, value=None):
    if not sku.strip() or len(sku)>64:
        raise ValueError("Invalid SKU")
    with closing(sqlite3.connect(db)) as c, c:
        c.execute("CREATE TABLE IF NOT EXISTS stock (sku TEXT PRIMARY KEY, quantity INTEGER NOT NULL CHECK(quantity BETWEEN 0 AND 1000000000))")
        if action=="register":
            if not isinstance(value,int) or not 0<=value<=1000000000:
                raise ValueError("Invalid quantity")
            c.execute("INSERT INTO stock VALUES (?,?)",(sku,value))
        elif action=="adjust":
            if not isinstance(value,int) or abs(value)>1000000000:
                raise ValueError("Invalid adjustment")
            cur=c.execute("UPDATE stock SET quantity=quantity+? WHERE sku=? AND quantity+? BETWEEN 0 AND 1000000000",(value,sku,value))
            if cur.rowcount!=1:
                raise ValueError("Unknown SKU or invalid resulting stock")
        elif action!="search":
            raise ValueError("Invalid action")
        row=c.execute("SELECT sku,quantity FROM stock WHERE sku=?",(sku,)).fetchone()
        return {"sku":row[0],"quantity":row[1]} if row else None
'''
CLI = '''def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["register","adjust","search"])
    p.add_argument("sku")
    p.add_argument("value",type=int,nargs="?")
    p.add_argument("--db",default=str(Path(__file__).with_name("inventory.db")))
    a=p.parse_args()
    try:
        print(json.dumps(operate(a.db,a.action,a.sku,a.value)))
        return 0
    except (ValueError,sqlite3.Error) as exc:
        print(json.dumps({"error":str(exc)}))
        return 1

if __name__=="__main__":
    raise SystemExit(main())
'''
def generate_inventory(root,*,approved=False):
    result=compose({"store":Component("store",STORE),"cli":Component("cli",CLI,("store",))},["cli"])
    if approved is not True:
        return {"status":"awaiting_authorization","components":result["components"]}
    root=Path(root)
    root.mkdir(parents=True,exist_ok=False)
    path=root/"inventory.py"
    with path.open("xb") as f:
        f.write(result["source"].encode("utf-8"))
    return {"status":"generated","artifact":str(path),"sha256":result["sha256"],"verified":False}
