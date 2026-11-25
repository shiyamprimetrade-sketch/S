import sqlite3,json,uuid,hashlib,datetime
from pathlib import Path
from contextlib import closing
from .generation import plan,generate
from .verification import verify
class Orchestrator:
    def __init__(self,root):
        self.root=Path(root).resolve(); self.root.mkdir(parents=True,exist_ok=True); self.db=self.root/'tasks.db'
        with closing(sqlite3.connect(self.db)) as c,c:
            c.execute('CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, state TEXT, plan TEXT, evidence TEXT)'); c.execute('CREATE TABLE IF NOT EXISTS journal(id TEXT, previous TEXT, state TEXT, timestamp TEXT)')
    def create(self,requirement):
        p=plan(requirement); task=uuid.uuid4().hex
        with closing(sqlite3.connect(self.db)) as c,c:
            c.execute('INSERT INTO tasks VALUES (?,?,?,?)',(task,'AWAITING_AUTHORIZATION',json.dumps(p),None)); c.execute('INSERT INTO journal VALUES (?,?,?,?)',(task,'CREATED','AWAITING_AUTHORIZATION',datetime.datetime.now(datetime.timezone.utc).isoformat()))
        return task,p
    def transition(self,task,old,new,evidence=None):
        with closing(sqlite3.connect(self.db)) as c,c:
            if c.execute('UPDATE tasks SET state=?,evidence=? WHERE id=? AND state=?',(new,json.dumps(evidence),task,old)).rowcount!=1: raise ValueError('Invalid state or replay')
            c.execute('INSERT INTO journal VALUES (?,?,?,?)',(task,old,new,datetime.datetime.now(datetime.timezone.utc).isoformat()))
    def execute(self,task,approve):
        if type(approve)!=bool: raise ValueError('Explicit decision required')
        with closing(sqlite3.connect(self.db)) as c: row=c.execute('SELECT plan FROM tasks WHERE id=?',(task,)).fetchone()
        if row is None: raise ValueError('Unknown task')
        if not approve: self.transition(task,'AWAITING_AUTHORIZATION','DENIED'); return None
        self.transition(task,'AWAITING_AUTHORIZATION','AUTHORIZED'); p=json.loads(row[0]); self.transition(task,'AUTHORIZED','GENERATING')
        try:
            f=generate(p,self.root/task); self.transition(task,'GENERATING','VERIFYING'); result=verify(f,p['capability']); result['artifact']=str(f.relative_to(self.root)); self.transition(task,'VERIFYING','COMPLETED' if result['passed'] else 'FAILED',result); return result
        except Exception as e:
            with closing(sqlite3.connect(self.db)) as c: state=c.execute('SELECT state FROM tasks WHERE id=?',(task,)).fetchone()[0]
            self.transition(task,state,'FAILED',{'error':str(e)}); raise
