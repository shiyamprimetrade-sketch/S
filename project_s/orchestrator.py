import sqlite3,json,uuid,hashlib,datetime,time
from pathlib import Path
from contextlib import closing
from .generation import plan,generate
from .verification import verify
from .contracts import request_for, receipt
class Orchestrator:
    def __init__(self,root):
        self.root=Path(root).resolve(); self.root.mkdir(parents=True,exist_ok=True); self.db=self.root/'tasks.db'
        with closing(sqlite3.connect(self.db)) as c,c:
            c.execute('CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, state TEXT, plan TEXT, evidence TEXT)'); c.execute('CREATE TABLE IF NOT EXISTS journal(id TEXT, previous TEXT, state TEXT, timestamp TEXT)')
            c.execute('CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY, request TEXT NOT NULL)')
    def create(self,requirement):
        p=plan(requirement)
        request=request_for(requirement,p['capability'])
        task=request['operation_id']
        with closing(sqlite3.connect(self.db)) as c,c:
            c.execute('INSERT INTO requests VALUES (?,?)',(task,json.dumps(request)))
            c.execute('INSERT INTO tasks VALUES (?,?,?,?)',(task,'AWAITING_AUTHORIZATION',json.dumps(p),None)); c.execute('INSERT INTO journal VALUES (?,?,?,?)',(task,'CREATED','AWAITING_AUTHORIZATION',datetime.datetime.now(datetime.timezone.utc).isoformat()))
        return task,p
    def original_request(self,task):
        with closing(sqlite3.connect(self.db)) as c: row=c.execute('SELECT request FROM requests WHERE id=?',(task,)).fetchone()
        if row is None: raise ValueError('Missing original request; create a new task')
        from .contracts import validate
        return validate(json.loads(row[0]))
    def transition(self,task,old,new,evidence=None):
        with closing(sqlite3.connect(self.db)) as c,c:
            if c.execute('UPDATE tasks SET state=?,evidence=? WHERE id=? AND state=?',(new,json.dumps(evidence),task,old)).rowcount!=1: raise ValueError('Invalid state or replay')
            c.execute('INSERT INTO journal VALUES (?,?,?,?)',(task,old,new,datetime.datetime.now(datetime.timezone.utc).isoformat()))
    def execute(self,task,approve,deadline_ms=120000):
        if type(deadline_ms) is not int or not 1 <= deadline_ms <= 120000: raise ValueError('Invalid execution deadline')
        deadline=time.monotonic()+deadline_ms/1000
        def check_deadline():
            if time.monotonic()>=deadline: raise TimeoutError('Execution deadline exceeded')
        if type(approve)!=bool: raise ValueError('Explicit decision required')
        with closing(sqlite3.connect(self.db)) as c: row=c.execute('SELECT plan FROM tasks WHERE id=?',(task,)).fetchone()
        if row is None: raise ValueError('Unknown task')
        if not approve: self.transition(task,'AWAITING_AUTHORIZATION','DENIED'); return None
        self.transition(task,'AWAITING_AUTHORIZATION','AUTHORIZED'); p=json.loads(row[0]); self.transition(task,'AUTHORIZED','GENERATING')
        try:
            check_deadline(); f=generate(p,self.root/task); check_deadline(); self.transition(task,'GENERATING','VERIFYING'); result=verify(f,p['capability']); check_deadline(); result['artifact']=str(f.relative_to(self.root)); result['receipt']=receipt(task,'COMPLETED',artifact_sha256=result['sha256'],evidence_ids=[task+':verification']) if result['passed'] else receipt(task,'FAILED',error_code='VERIFICATION_FAILED'); self.transition(task,'VERIFYING','COMPLETED' if result['passed'] else 'FAILED',result); return result
        except Exception as e:
            with closing(sqlite3.connect(self.db)) as c: state=c.execute('SELECT state FROM tasks WHERE id=?',(task,)).fetchone()[0]
            self.transition(task,state,'FAILED',{'error':str(e),'receipt':receipt(task,'FAILED',error_code='DEADLINE_EXCEEDED' if isinstance(e,TimeoutError) else 'EXECUTION_FAILED')}); raise
