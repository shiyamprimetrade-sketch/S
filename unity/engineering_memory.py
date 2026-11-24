"""Local engineering history; records are not training data or trusted signatures."""
import sqlite3
import json
from contextlib import closing
from pathlib import Path
from uuid import uuid4
from .website_acceptance import verify_website

class EngineeringMemory:
    def __init__(self,db):
        self.db=str(db)
        with closing(sqlite3.connect(self.db)) as c, c:
            c.execute("CREATE TABLE IF NOT EXISTS website_history (id TEXT PRIMARY KEY, spec TEXT NOT NULL, artifact TEXT NOT NULL, digest TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP)")
    def remember(self,root,name,heading,description):
        root=Path(root).resolve()
        evidence=verify_website(root,name,heading,description)
        if evidence["status"]!="verified":
            raise ValueError("unverified artifact")
        ident=uuid4().hex
        spec=json.dumps({"name":name,"heading":heading,"description":description},ensure_ascii=False,sort_keys=True)
        with closing(sqlite3.connect(self.db)) as c,c:
            c.execute("INSERT INTO website_history(id,spec,artifact,digest) VALUES (?,?,?,?)",(ident,spec,str(root),evidence["sha256"]))
        return ident
    def recall(self,ident):
        with closing(sqlite3.connect(self.db)) as c:
            row=c.execute("SELECT spec,artifact,digest FROM website_history WHERE id=?",(ident,)).fetchone()
        if row is None:
            return None
        spec,root,digest=row
        spec=json.loads(spec)
        evidence=verify_website(root,**spec)
        return {"id":ident,"spec":spec,"artifact":root,"sha256":digest,
                "currently_verified":evidence["status"]=="verified" and evidence.get("sha256")==digest}
