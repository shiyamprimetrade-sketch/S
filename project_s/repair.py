"""Repair by verified successor generation; never overwrite original artifacts."""
import hashlib,hmac,time
from contextlib import closing
import sqlite3,json,uuid,datetime
from .secure import digest

class Repair:
    def __init__(self,engine):
        self.engine=engine
        with closing(sqlite3.connect(engine.db)) as c,c:
            c.execute('CREATE TABLE IF NOT EXISTS repairs(parent TEXT, attempt INTEGER, child TEXT, status TEXT, PRIMARY KEY(parent,attempt))')
    def reconcile(self):
        """Call only under exclusive workspace lock at startup."""
        with closing(sqlite3.connect(self.engine.db)) as c,c:
            return c.execute("UPDATE repairs SET status='INTERRUPTED' WHERE status='RESERVED'").rowcount
    def propose(self,task):
        r=self.engine.get(task)
        if r['state']!='FAILED':raise ValueError('Only failed tasks eligible')
        with closing(sqlite3.connect(self.engine.db)) as c:
            if c.execute('SELECT 1 FROM repairs WHERE child=?',(task,)).fetchone():raise ValueError('Repair successor cannot start a new chain')
            count=c.execute('SELECT COUNT(*) FROM repairs WHERE parent=?',(task,)).fetchone()[0]
        if count>=3:raise ValueError('Repair limit reached')
        return {'task':task,'plan':digest(r['plan']),'attempt':count+1,'operation':'regenerate successor','diagnosis':r['evidence'],'expires':time.time()+120}
    def authorize(self,proposal,approve):
        if approve is not True:raise ValueError('Explicit repair approval required')
        return {'proposal':proposal,'mac':hmac.new(self.engine.key,digest(proposal).encode(),'sha256').hexdigest()}
    def execute(self,token):
        p=token['proposal']; e=self.engine
        if not hmac.compare_digest(token['mac'],hmac.new(e.key,digest(p).encode(),'sha256').hexdigest()):raise ValueError('Invalid repair authorization')
        if p['expires']<=time.time() or p!=dict(p,operation='regenerate successor'):raise ValueError('Invalid repair scope')
        r=e.get(p['task'])
        if r['state']!='FAILED' or digest(r['plan'])!=p['plan']:raise ValueError('Changed repair target')
        with closing(sqlite3.connect(e.db)) as c,c:
            c.execute('BEGIN IMMEDIATE')
            count=c.execute('SELECT COUNT(*) FROM repairs WHERE parent=?',(p['task'],)).fetchone()[0]
            if count>=3 or p['attempt']!=count+1:raise ValueError('Repair limit or replay')
            if c.execute('SELECT 1 FROM repairs WHERE child=?',(p['task'],)).fetchone():raise ValueError('Repair successor cannot start a new chain')
            child=uuid.uuid4().hex
            c.execute('INSERT INTO tasks VALUES (?,?,?,?)',(child,'AWAITING_AUTHORIZATION',json.dumps(r['plan']),None))
            c.execute('INSERT INTO journal VALUES (?,?,?,?)',(child,'CREATED','AWAITING_AUTHORIZATION',datetime.datetime.now(datetime.timezone.utc).isoformat()))
            c.execute('INSERT INTO repairs VALUES (?,?,?,?)',(p['task'],p['attempt'],child,'RESERVED'))
        try:
            result=e.execute_token(child,e.issue(child))
            status='VERIFIED' if result['passed'] else 'FAILED'
            return {'child':child,'status':status,'evidence':result}
        except Exception:
            status='FAILED'
            raise
        finally:
            with closing(sqlite3.connect(e.db)) as c,c:
                c.execute('UPDATE repairs SET child=?,status=? WHERE parent=? AND attempt=?',(child,status,p['task'],p['attempt']))
