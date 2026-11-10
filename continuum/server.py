"""Loopback-only web server; no third-party Python dependencies."""
import json
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from .engine import chat,ollama_models,DEFAULT_MODEL,LocalModelError,TEACHER_MODEL,CHAR_MODEL,S_RESEARCH_MODEL
from .relations import related
from .storage import Store
from .api_keys import create_key,list_keys,revoke_key,authenticate_key
from .invention import InventionLab
from .meaning import TeacherMemory
from .apprenticeship import Apprenticeship

ROOT=Path(__file__).resolve().parent.parent
WEB=ROOT/'web'

class API(BaseHTTPRequestHandler):
    server_version='S-Local/0.7'
    def log_message(self,fmt,*args):
        # Avoid recording private message contents in server logs.
        if os.environ.get('AXIOM_HTTP_LOG')=='1':super().log_message(fmt,*args)
    def _json(self,status,value):
        body=json.dumps(value,ensure_ascii=False).encode('utf8')
        self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Security-Policy',"default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.send_header('Content-Length',str(len(body)))
        self.end_headers();self.wfile.write(body)
    def _body(self):
        n=int(self.headers.get('Content-Length','0'))
        max_bytes=45*1024*1024 if urlparse(self.path).path=='/api/import-chatgpt' else 100000
        if n<2 or n>max_bytes:raise ValueError('Invalid request size')
        return json.loads(self.rfile.read(n))
    def _query(self):
        return parse_qs(urlparse(self.path).query)
    def _param(self,name):
        return self._query().get(name,[''])[0]
    def _static(self,name):
        allow={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
        if name not in allow:return self._json(404,{'error':'Not found'})
        p=WEB/allow[name]
        body=p.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type',mimetypes.guess_type(p)[0] or 'text/html')
        self.send_header('Content-Length',str(len(body)))
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers();self.wfile.write(body)
    def _api(self,verb):
        path=urlparse(self.path).path
        s=self.server.store
        if verb=='GET':
            if path=='/api/apprenticeship/readiness':return self.server.apprentice.readiness(self._param('project_id'))
            if path=='/api/apprenticeship/cases':return self.server.apprentice.cases(self._param('project_id'))
            if path=='/api/apprenticeship/runs':return self.server.apprentice.runs(self._param('project_id'))
            if path=='/api/memory/lessons':return self.server.memory.lessons(self._param('project_id'))
            if path=='/api/memory/concepts':return self.server.memory.concepts(self._param('project_id'))
            if path=='/api/memory/recall':return self.server.memory.recall(self._param('project_id'),self._param('q'))
            if path=='/api/inventions':return self.server.lab.list(self._param('project_id'))
            if path=='/api/invention':return self.server.lab.detail(self._param('project_id'),self._param('goal_id'))
            if path=='/api/keys':return list_keys(s)
            if path=='/api/health':
                from .teacherlab import MODEL
                models=[S_RESEARCH_MODEL]
                if MODEL.exists():models.append(TEACHER_MODEL)
                from .charllm import DEFAULT_CKPT
                from importlib.util import find_spec
                if DEFAULT_CKPT.exists() and find_spec('torch') is not None:models.append(CHAR_MODEL)
                models.extend(ollama_models())
                return {'ok':True,'app':'S','models':models,'default_model':S_RESEARCH_MODEL,
                        'data_file':str(s.path),'inference':'local only','cloud_api':False}
            if path=='/api/projects':return s.list_projects()
            if path=='/api/sessions':return s.list_sessions(self._param('project_id'))
            if path=='/api/messages':return s.list_messages(self._param('session_id'))
            if path=='/api/feedback':return s.list_feedback(self._param('project_id'))
            if path=='/api/notes':return s.list_notes(self._param('project_id'))
            if path=='/api/search':return s.search_history(self._param('project_id'),self._param('q'))
            if path=='/api/relations':return related(s,self._param('project_id'),self._param('word'))
            if path=='/api/export':return s.export_project(self._param('project_id'))
            if path=='/api/handoff':return {'markdown':s.handoff(self._param('project_id'))}
            if path=='/api/teacherlab':
                from .teacherlab import REPORT
                return json.loads(REPORT.read_text(encoding='utf8')) if REPORT.exists() else {'trained':False}
        else:
            b=self._body()
            lab=self.server.lab
            if path=='/api/apprenticeship/cases':return self.server.apprentice.add_case(
                b.get('project_id'),b.get('question'),b.get('expected_lesson_id'))
            if path=='/api/apprenticeship/delete':return self.server.apprentice.delete_case(
                b.get('project_id'),b.get('case_id'))
            if path=='/api/apprenticeship/run':return self.server.apprentice.run(b.get('project_id'))
            if path=='/api/memory/teach':return self.server.memory.teach(
                b.get('project_id'),b.get('title',''),b.get('principle',''),b.get('cues',[]))
            if path=='/api/memory/concept':return self.server.memory.define_concept(
                b.get('project_id'),b.get('label',''),b.get('variants',[]))
            if path=='/api/memory/review':return self.server.memory.review(
                b.get('project_id'),b.get('lesson_id'),b.get('decision'))
            if path=='/api/memory/revise':return self.server.memory.revise(
                b.get('project_id'),b.get('lesson_id'),b.get('principle',''),b.get('cues',[]))
            if path=='/api/memory/transfer':return self.server.memory.transfer(
                b.get('source_project_id'),b.get('destination_project_id'),
                b.get('lesson_id'),bool(b.get('include_concepts',False)))
            if path=='/api/memory/archive-candidates':return self.server.memory.suggest_archive_corrections(
                b.get('project_id'),b.get('limit',30))
            if path=='/api/inventions':return lab.create(b.get('project_id'),b.get('objective',''),b.get('success_criteria',''))
            if path=='/api/invention/example':return lab.add_example(b.get('project_id'),b.get('goal_id'),b.get('input'),b.get('expected'),b.get('role','teach'))
            if path=='/api/invention/correct':return lab.correct(b.get('project_id'),b.get('goal_id'),b.get('example_id'),b.get('expected'),b.get('teacher_note',''))
            if path=='/api/invention/synthesize':return lab.synthesize(b.get('project_id'),b.get('goal_id'))
            if path=='/api/invention/evaluate':return lab.evaluate(b.get('project_id'),b.get('goal_id'))
            if path=='/api/invention/predict':return lab.predict(b.get('project_id'),b.get('goal_id'),b.get('input'))
            if path=='/api/keys':return create_key(s,b.get('name',''),b.get('scopes',['read','chat']))
            if path=='/api/revoke-key':return revoke_key(s,b.get('id',''))
            if path=='/api/projects':return s.create_project(b.get('name',''))
            if path=='/api/sessions':return s.create_session(b.get('project_id',''),b.get('title','New conversation'))
            if path=='/api/notes':return s.add_note(b.get('project_id',''),b.get('title',''),b.get('body',''),b.get('kind','lesson'),b.get('evidence',''))
            if path=='/api/chat':return chat(s,b.get('session_id',''),b.get('text',''),b.get('model',S_RESEARCH_MODEL))
            if path=='/api/delete-session':return s.delete_session(b.get('session_id',''))
            if path=='/api/feedback':
                outcome = s.set_feedback(b.get('message_id',''),b.get('rating'),b.get('note',''))
                if b.get('rating') == -1 and str(b.get('note','')).strip():
                    msg = b.get('message_id','')
                    with s.connect() as db:
                        row = db.execute('SELECT se.project_id FROM messages m JOIN sessions se '
                                         'ON se.id=m.session_id WHERE m.id=?',(msg,)).fetchone()
                    if row:
                        try:
                            candidate = self.server.memory.propose_feedback(row['project_id'],msg,b['note'])
                        except __import__('sqlite3').IntegrityError:
                            candidate = None  # Already proposed on earlier feedback
                        outcome['memory_candidate_id'] = candidate['id'] if candidate else None
                return outcome
            if path=='/api/import-chatgpt':
                if not isinstance(b.get('document'),(list,dict)):raise ValueError('Invalid ChatGPT export')
                return s.import_chatgpt(b['document'],b.get('title_contains',''),bool(b.get('redact',True)))
        raise ValueError('Unknown route')
    def _v1(self,verb):
        """Credential-protected loopback API for third-party local clients.

        Existing /api routes remain the trusted local browser UI; /v1 explicitly
        requires a scoped Bearer key. No external-network listener is enabled.
        """
        path=urlparse(self.path).path
        scope='read' if verb=='GET' else 'chat'
        header=self.headers.get('Authorization','')
        if not header.startswith('Bearer '):
            return self._json(401,{'error':'Missing or invalid S API key'})
        if not authenticate_key(self.server.store,header[7:],scope):
            return self._json(401,{'error':'Missing, revoked or insufficient S API key'})
        if verb=='GET' and path=='/v1/models':
            from .teacherlab import MODEL
            from .charllm import DEFAULT_CKPT
            from importlib.util import find_spec
            models=[S_RESEARCH_MODEL]
            if MODEL.exists():models.append(TEACHER_MODEL)
            if DEFAULT_CKPT.exists() and find_spec('torch') is not None:models.append(CHAR_MODEL)
            models.extend(ollama_models())
            return self._json(200,{'object':'list','data':[{'id':m,'object':'model'} for m in models]})
        if verb=='POST' and path=='/v1/chat':
            b=self._body()
            if not isinstance(b,dict) or not isinstance(b.get('session_id'),str) or not isinstance(b.get('text'),str):
                raise ValueError('session_id and text are required')
            return self._json(200,chat(self.server.store,b['session_id'],b['text'],b.get('model',S_RESEARCH_MODEL)))
        return self._json(404,{'error':'Unknown S API route'})
    def _guard(self):
        # Only local origins may POST, blocking cross-origin browser attempts.
        host=self.headers.get('Host','')
        allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        if host not in allowed:raise PermissionError('Invalid host')
        origin=self.headers.get('Origin')
        if origin and origin not in {f'http://{a}' for a in allowed}:raise PermissionError('Cross-origin requests rejected')
    def do_GET(self):
        try:
            self._guard()
            if urlparse(self.path).path.startswith('/v1/'):
                return self._v1('GET')
            if urlparse(self.path).path.startswith('/api/'):
                return self._json(200,self._api('GET'))
            return self._static(urlparse(self.path).path)
        except PermissionError as e:self._json(403,{'error':str(e)})
        except ValueError as e:self._json(400,{'error':str(e)})
        except Exception as e:self._json(500,{'error':'Internal error: '+type(e).__name__})
    def do_POST(self):
        try:
            self._guard()
            if urlparse(self.path).path.startswith('/v1/'):
                return self._v1('POST')
            return self._json(200,self._api('POST'))
        except PermissionError as e:self._json(403,{'error':str(e)})
        except (ValueError,json.JSONDecodeError) as e:self._json(400,{'error':str(e)})
        except LocalModelError as e:self._json(503,{'error':str(e)})
        except Exception as e:self._json(500,{'error':'Internal error: '+type(e).__name__})

class AppServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,host,port,db_path):
        if host not in ('127.0.0.1','localhost'):
            raise ValueError('Only loopback binding is supported')
        super().__init__((host,port),API)
        self.store=Store(db_path)
        self.store.seed_if_empty()
        self.lab=InventionLab(self.store)
        self.memory=TeacherMemory(self.store)
        self.apprentice=Apprenticeship(self.store)

def serve(db_path,port=8765):
    with AppServer('127.0.0.1',port,db_path) as app:
        print(f'S Local Human Computer: http://127.0.0.1:{port}',flush=True)
        print(f'Data: {app.store.path}',flush=True)
        app.serve_forever()
