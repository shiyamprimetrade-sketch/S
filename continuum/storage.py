"""SQLite project memory. No cloud account or proprietary DB needed."""
from __future__ import annotations
import json
from contextlib import contextmanager,closing
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS projects(
 id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions(
 id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
 title TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages(
 id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
 role TEXT NOT NULL CHECK(role IN ('user','assistant')), content TEXT NOT NULL,
 created_at TEXT NOT NULL, model TEXT, elapsed_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id,created_at);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id,created_at);
CREATE TABLE IF NOT EXISTS notes(
 id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
 title TEXT NOT NULL, body TEXT NOT NULL, kind TEXT NOT NULL,
 evidence TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_project ON notes(project_id,created_at);
CREATE TABLE IF NOT EXISTS word_counts(
 project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
 token TEXT NOT NULL, count INTEGER NOT NULL, PRIMARY KEY(project_id,token)
);
CREATE TABLE IF NOT EXISTS word_links(
 project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
 a TEXT NOT NULL, b TEXT NOT NULL, count INTEGER NOT NULL,
 PRIMARY KEY(project_id,a,b)
);
CREATE TABLE IF NOT EXISTS archive_extras(
 id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
 role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL, source_id TEXT, imported_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events(
 id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL,
 action TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT NOT NULL
);
"""
KINDS = {"lesson", "hypothesis", "experiment", "evidence", "decision", "question"}

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def new_id() -> str:
    return uuid.uuid4().hex

class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('PRAGMA busy_timeout=30000')
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def list_projects(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM projects ORDER BY created_at,name')]

    def create_project(self, name):
        name = str(name).strip()
        if not 1 <= len(name) <= 100: raise ValueError('Project name must be 1–100 characters')
        item = {'id': new_id(), 'name': name, 'created_at': now()}
        with self.connect() as db:
            db.execute('INSERT INTO projects VALUES (:id,:name,:created_at)',item)
        return item

    def get_project(self, pid):
        with self.connect() as db:
            row = db.execute('SELECT * FROM projects WHERE id=?',(pid,)).fetchone()
        if row is None: raise ValueError('Unknown project')
        return dict(row)

    def create_session(self,pid,title='New conversation'):
        self.get_project(pid)
        title = str(title).strip()[:120] or 'New conversation'
        item={'id':new_id(),'project_id':pid,'title':title,'created_at':now()}
        with self.connect() as db:
            db.execute('INSERT INTO sessions VALUES (:id,:project_id,:title,:created_at)',item)
        return item

    def list_sessions(self,pid):
        self.get_project(pid)
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM sessions WHERE project_id=? ORDER BY created_at DESC',(pid,))]

    def get_session(self,sid):
        with self.connect() as db:
            r=db.execute('SELECT * FROM sessions WHERE id=?',(sid,)).fetchone()
        if r is None: raise ValueError('Unknown conversation')
        return dict(r)

    def add_message(self,sid,role,content,model=None,elapsed_ms=None):
        ses=self.get_session(sid)
        if role not in ('user','assistant'): raise ValueError('Invalid role')
        content=str(content).strip()
        if not 1<=len(content)<=50000:raise ValueError('Message length must be 1–50000')
        item={'id':new_id(),'session_id':sid,'role':role,'content':content,'model':model,'elapsed_ms':elapsed_ms,'created_at':now()}
        with self.connect() as db:
            db.execute('INSERT INTO messages VALUES (:id,:session_id,:role,:content,:created_at,:model,:elapsed_ms)',item)
            first=db.execute('SELECT count(*) FROM messages WHERE session_id=?',(sid,)).fetchone()[0]
            if first==1 and role=='user':
                db.execute('UPDATE sessions SET title=? WHERE id=?',(content[:76],sid))
        if role=='user':
            from .relations import learn_words
            learn_words(self,ses['project_id'],content)
        return item

    def list_messages(self,sid,limit=500):
        self.get_session(sid)
        with self.connect() as db:
            rows=db.execute('SELECT id,session_id,role,content,created_at,model,elapsed_ms FROM (SELECT rowid AS seq,* FROM messages WHERE session_id=? ORDER BY rowid DESC LIMIT ?) ORDER BY seq',(sid,limit))
            return [dict(x) for x in rows]

    def context_messages(self,sid,limit=14):
        with self.connect() as db:
            rows=db.execute('SELECT role,content FROM messages WHERE session_id=? ORDER BY rowid DESC LIMIT ?',(sid,limit))
            return [dict(x) for x in reversed(list(rows))]

    def add_note(self,pid,title,body,kind='lesson',evidence=''):
        self.get_project(pid)
        if kind not in KINDS: raise ValueError('Unknown note type')
        title=str(title).strip()
        body=str(body).strip()
        evidence=str(evidence).strip()
        if not 1<=len(title)<=150 or not 1<=len(body)<=10000 or len(evidence)>2000:raise ValueError('Invalid note length')
        if kind=='evidence' and not evidence:raise ValueError('Evidence note needs a test or artifact reference')
        item={'id':new_id(),'project_id':pid,'title':title,'body':body,'kind':kind,'evidence':evidence,'created_at':now()}
        with self.connect() as db:
            db.execute('INSERT INTO notes VALUES (:id,:project_id,:title,:body,:kind,:evidence,:created_at)',item)
        return item

    def list_notes(self,pid,limit=100):
        self.get_project(pid)
        with self.connect() as db:
            return [dict(x) for x in db.execute('SELECT * FROM notes WHERE project_id=? ORDER BY rowid DESC LIMIT ?',(pid,limit))]

    def log(self,pid,action,detail):
        with self.connect() as db:
            db.execute('INSERT INTO events(project_id,action,detail,created_at) VALUES(?,?,?,?)',(pid,action,detail[:2000],now()))

    def export_project(self,pid):
        p=self.get_project(pid)
        sessions=self.list_sessions(pid)
        return {'format':'axiomflow-continuum-project','version':1,'exported_at':now(),
                'project':p,'sessions':[{'session':s,'messages':self.list_messages(s['id'],1000000)} for s in sessions],
                'notes':self.list_notes(pid,1000000)}

    def import_project(self,document):
        if document.get('format')!='axiomflow-continuum-project' or document.get('version')!=1:
            raise ValueError('Unsupported export file')
        project=document['project']
        new=self.create_project(project['name']+' (imported)')
        old_to_new={}
        with self.connect() as db:
            for section in document['sessions']:
                s=section['session']
                sid=new_id();old_to_new[s['id']]=sid
                db.execute('INSERT INTO sessions VALUES(?,?,?,?)',(sid,new['id'],s['title'],now()))
                for m in section['messages']:
                    if m['role'] not in ('user','assistant'):raise ValueError('Bad role in import')
                    db.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?)',(new_id(),sid,m['role'],m['content'],now(),m.get('model'),m.get('elapsed_ms')))
            for n in document['notes']:
                if n['kind'] not in KINDS:raise ValueError('Bad note type in import')
                db.execute('INSERT INTO notes VALUES(?,?,?,?,?,?,?)',(new_id(),new['id'],n['title'],n['body'],n['kind'],n.get('evidence',''),now()))
        return new

    def handoff(self,pid):
        """Plain-text context capsule readable by any chat model."""
        p=self.get_project(pid)
        notes=self.list_notes(pid,30)
        sessions=self.list_sessions(pid)
        out=[f"# AxiomFlow Continuum — {p['name']}",
             "", "This is portable project context, not model weights or autonomous memory.",
             "Respect project boundaries. Separate claims, hypotheses, evidence and verified results.",
             "## Lessons and checkpoints"]
        for n in reversed(notes):
            out.extend([f"### [{n['kind'].upper()}] {n['title']}",n['body']])
            if n['evidence']:out.append('Evidence: '+n['evidence'])
        out.append('## Recent conversation excerpts')
        for ses in sessions[:3]:
            out.append('### Conversation: '+ses['title'])
            for m in self.list_messages(ses['id'],12):
                out.append(f"**{m['role']}**: {m['content'][:3000]}")
        out.extend(['','## Next research prompt',
                    'Continue from the most recent verified checkpoint. Do not assume untested outcomes are proven. Ask one focused question if needed.'])
        return '\n\n'.join(out)

    def backup(self,destination):
        destination=Path(destination)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if self.path.resolve()==destination.resolve():raise ValueError('Cannot back up over live database')
        with self.connect() as src, closing(sqlite3.connect(destination)) as dst:
            src.backup(dst)
        with closing(sqlite3.connect(destination)) as check:
            result=check.execute('PRAGMA integrity_check').fetchone()[0]
            if result!='ok':raise RuntimeError('Backup integrity check failed')
        return destination

    def seed_if_empty(self):
        if self.list_projects():return False
        p=self.create_project('Independent Computing & KEY26')
        lessons=[
          ('lesson','Keyboard as an invention interface','Letters and symbols can be combined into words and instructions. Existing language syntax is a design choice, not a law of nature.',''),
          ('lesson','Words, numbers and relationships','ABCA and CABA each contain four letters. Four can be written FOUR; APPLE → FIVE → FOUR → FOUR is a deterministic word-length chain.',''),
          ('hypothesis','Emergent word meaning','Explore whether invented words can discover useful meanings or operations through interactions and feedback. This is a research goal, not demonstrated general intelligence.',''),
          ('hypothesis','Reconstructive logical memory','Study storing compact rules to regenerate information as needed; do not confuse unbounded descriptions with unbounded physical storage.',''),
          ('lesson','Errors as information','A terminal can respond to unexpected outcomes by diagnosing, proposing, testing and preserving alternative methods rather than simply stopping. Labels do not change the actual result.',''),
          ('decision','Invention protocol','Preserve project boundaries, distinguish established facts from hypotheses, measure work with executable tests and record evidence. Ask one clarifying question when needed.',''),
        ]
        for kind,title,body,evid in lessons:self.add_note(p['id'],title,body,kind,evid)
        self.create_session(p['id'],'Session 03 — Continue the invention')
        return True

# The importer is deliberately a separate transaction; it must not call
# add_message() for every historic message (that would wrongly train word
# relations and make a large import unnecessarily slow).
def _import_chatgpt(self, document, title_contains='', redact=True):
    from .transcript import parse_export
    convs=parse_export(document,title_contains,redact)
    project=self.create_project('ChatGPT Archive'+(' — '+title_contains[:65] if title_contains else ''))
    imported=0;other=0
    with self.connect() as db:
        for c in convs:
            sid=new_id()
            db.execute('INSERT INTO sessions VALUES(?,?,?,?)',(sid,project['id'],c['title'],now()))
            for m in c['messages']:
                if m['role'] in ('user','assistant'):
                    # Preserve source timestamps and full text, even for long messages.
                    db.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?)',
                               (new_id(),sid,m['role'],m['content'],m['created_at'],m.get('model'),None))
                    imported+=1
                else:
                    db.execute('INSERT INTO archive_extras VALUES(?,?,?,?,?,?,?)',
                               (new_id(),sid,m['role'],m['content'],m['created_at'],m.get('source_id',''),now()))
                    other+=1
    return {'project_id':project['id'],'project_name':project['name'],
            'conversations':len(convs),'chat_messages':imported,'other_messages':other,
            'redacted':redact,'media':'metadata placeholders only'}

def _search_history(self,pid,query,limit=20):
    """Bounded, relevance-ranked keyword recall within one project only."""
    import re
    self.get_project(pid)
    query=str(query).strip()
    if not 2<=len(query)<=500:raise ValueError('Search needs 2–500 characters')
    stop={'what','when','where','why','which','with','this','that','have','from',
          'about','your','mine','does','said','tell','please','could','would','should',
          'the','and','for','are','was','were','did','how','can','you','me','my','our'}
    words=[]
    for w in re.findall(r'[^\W_]{3,}',query.casefold()):
        if w not in stop and w not in words:words.append(w)
    words=words[:7]
    if not words:raise ValueError('Search needs a meaningful word')
    pats=['%'+w.replace('\\','\\\\').replace('%','\\%').replace('_','\\_')+'%' for w in words]
    rank=' + '.join('CASE WHEN m.content LIKE ? ESCAPE \'\\\' THEN 1 ELSE 0 END' for _ in words)
    match=' OR '.join('m.content LIKE ? ESCAPE \'\\\'' for _ in words)
    sql=('SELECT m.id,m.role,m.content,m.created_at,s.id AS session_id,s.title, ('+rank+') AS relevance '
         'FROM messages m JOIN sessions s ON m.session_id=s.id '
         'WHERE s.project_id=? AND ('+match+') ORDER BY relevance DESC, m.rowid DESC LIMIT ?')
    params=pats+[pid]+pats+[min(100,max(1,int(limit)))]
    with self.connect() as db:
        return [{**dict(r),'content':r['content'][:1300]} for r in db.execute(sql,params)]

Store.import_chatgpt=_import_chatgpt
Store.search_history=_search_history
