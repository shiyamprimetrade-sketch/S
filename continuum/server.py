"""Loopback-only web server; no third-party Python dependencies."""
import json
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from .engine import chat,ollama_models,DEFAULT_MODEL,LocalModelError,TEACHER_MODEL,CHAR_MODEL
from .relations import related
from .storage import Store

ROOT=Path(__file__).resolve().parent.parent
WEB=ROOT/'web'

class API(BaseHTTPRequestHandler):
    server_version='AxiomFlow/0.1'
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
            if path=='/api/health':
                from .teacherlab import MODEL
                models=ollama_models()
                if MODEL.exists():models=[TEACHER_MODEL]+models
                from .charllm import DEFAULT_CKPT
                if DEFAULT_CKPT.exists():models=[CHAR_MODEL]+models
                return {'ok':True,'models':models,'default_model':TEACHER_MODEL if MODEL.exists() else DEFAULT_MODEL,'data_file':str(s.path)}
            if path=='/api/projects':return s.list_projects()
            if path=='/api/sessions':return s.list_sessions(self._param('project_id'))
            if path=='/api/messages':return s.list_messages(self._param('session_id'))
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
            if path=='/api/projects':return s.create_project(b.get('name',''))
            if path=='/api/sessions':return s.create_session(b.get('project_id',''),b.get('title','New conversation'))
            if path=='/api/notes':return s.add_note(b.get('project_id',''),b.get('title',''),b.get('body',''),b.get('kind','lesson'),b.get('evidence',''))
            if path=='/api/chat':return chat(s,b.get('session_id',''),b.get('text',''),b.get('model',DEFAULT_MODEL))
            if path=='/api/import-chatgpt':
                if not isinstance(b.get('document'),(list,dict)):raise ValueError('Invalid ChatGPT export')
                return s.import_chatgpt(b['document'],b.get('title_contains',''),bool(b.get('redact',True)))
        raise ValueError('Unknown route')
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
            if urlparse(self.path).path.startswith('/api/'):
                return self._json(200,self._api('GET'))
            return self._static(urlparse(self.path).path)
        except PermissionError as e:self._json(403,{'error':str(e)})
        except ValueError as e:self._json(400,{'error':str(e)})
        except Exception as e:self._json(500,{'error':'Internal error: '+type(e).__name__})
    def do_POST(self):
        try:
            self._guard()
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

def serve(db_path,port=8765):
    with AppServer('127.0.0.1',port,db_path) as app:
        print(f'AxiomFlow Continuum: http://127.0.0.1:{port}',flush=True)
        print(f'Data: {app.store.path}',flush=True)
        app.serve_forever()
