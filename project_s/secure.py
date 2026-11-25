import hashlib,hmac,json,time,secrets,os,uuid,sqlite3
from pathlib import Path
from contextlib import closing
from .orchestrator import Orchestrator
from .generation import generate
from .verification import verify

def digest(p): return hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()
class SecureEngine(Orchestrator):
    def __init__(self,root):
        super().__init__(root)
        key=self.root/'authorization.key'
        try:
            fd=os.open(key,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'wb') as f:f.write(secrets.token_bytes(32))
        except FileExistsError: pass
        self.key=key.read_bytes()
    def get(self,task):
        if not isinstance(task,str) or len(task)!=32 or uuid.UUID(hex=task).hex!=task: raise ValueError('Invalid task')
        with closing(sqlite3.connect(self.db)) as c:
            r=c.execute('SELECT state,plan,evidence FROM tasks WHERE id=?',(task,)).fetchone()
        if not r:raise ValueError('Unknown task')
        return {'id':task,'state':r[0],'plan':json.loads(r[1]),'evidence':json.loads(r[2]) if r[2] else None}
    def history(self):
        with closing(sqlite3.connect(self.db)) as c: ids=[r[0] for r in c.execute('SELECT id FROM tasks ORDER BY rowid DESC')]
        return [self.get(i) for i in ids]
    def issue(self,task,ttl=120):
        r=self.get(task)
        if r['state']!='AWAITING_AUTHORIZATION':raise ValueError('Not awaiting approval')
        p={'task':task,'plan':digest(r['plan']),'operation':'generate','workspace':str(self.root/task),'issued':time.time(),'expires':time.time()+ttl,'nonce':secrets.token_hex(16)}
        return {'scope':p,'mac':hmac.new(self.key,digest(p).encode(),'sha256').hexdigest()}
    def execute_token(self,task,token):
        r=self.get(task); p=token['scope']; now=time.time()
        if not hmac.compare_digest(token['mac'],hmac.new(self.key,digest(p).encode(),'sha256').hexdigest()):raise ValueError('Invalid authentication')
        if not (p['issued']<=now<p['expires'] and p['task']==task and p['plan']==digest(r['plan']) and p['operation']=='generate' and p['workspace']==str(self.root/task)):raise ValueError('Invalid scope or expiry')
        d=self.root/task
        if d.exists() or d.is_symlink():raise ValueError('Workspace already exists')
        return super().execute(task,True)
    def execute(self,task,approve):
        if approve is False:return super().execute(task,False)
        raise ValueError('Authenticated token required')
    def recover_interrupted(self):
        """Reconcile interrupted tasks at exclusive server startup; never repeat side effects."""
        outcomes=[]
        for r in self.history():
            if r['state'] not in ('AUTHORIZED','GENERATING','VERIFYING','RECOVERING'):continue
            task=r['id']; old=r['state']
            if old!='RECOVERING':self.transition(task,old,'RECOVERING')
            d=self.root/task
            try:
                if d.is_symlink():raise ValueError('Unsafe recovery workspace')
                name='index.html' if r['plan']['capability']=='website' else r['plan']['capability']+'.py'
                f=d/name
                if not f.is_file() or f.is_symlink():raise ValueError('Incomplete artifact; fresh authorization required')
                if f.resolve().parent!=d:raise ValueError('Unsafe recovery artifact')
                # Do not execute uncommitted files: their provenance cannot be established.
                raise ValueError('Interrupted artifact not committed; fresh authorization required')
            except Exception as e:
                self.transition(task,'RECOVERING','FAILED',{'error':str(e),'recovery':'fail_closed','workspace_preserved':True})
                outcomes.append(task)
        return outcomes
    def artifact(self,task):
        r=self.get(task)
        if r['state']!='COMPLETED':raise ValueError('Not completed')
        f=self.root/r['evidence']['artifact']
        if f.is_symlink() or f.parent.is_symlink() or f.resolve().parent!=self.root/task:raise ValueError('Invalid artifact path')
        if hashlib.sha256(f.read_bytes()).hexdigest()!=r['evidence']['sha256']:raise ValueError('Artifact tampered')
        return f
