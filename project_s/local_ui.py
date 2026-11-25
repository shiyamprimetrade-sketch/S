from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from pathlib import Path
import json, secrets, argparse, html
from .secure import SecureEngine
from .calculator import calculate
PAGE='''<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Project S</title><style>body{background:#10141d;color:#eee;font:16px system-ui;max-width:800px;margin:32px auto;padding:16px}textarea,button,input{font:inherit;padding:12px;margin:6px;background:#202b39;color:white;border:1px solid #657789;border-radius:8px}textarea{width:90%}button{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#192330;padding:16px}a{color:#8de}iframe{width:100%;height:360px;background:white}</style><h1>Project S</h1><p>Local software generation · bounded prototype</p><textarea id=req placeholder="Create a website for River Studio"></textarea><button onclick="create()">Create plan</button><pre id=out></pre><div id=actions></div><h2>Project history</h2><div id=history></div><script>const token='TOKEN';let current;async function api(path,data){let r=await fetch(path,{method:data?'POST':'GET',headers:{'Content-Type':'application/json','X-Session':token},body:data?JSON.stringify(data):undefined});let v=await r.json();if(!r.ok)throw Error(v.error);return v}function show(v){current=v;out.textContent=JSON.stringify(v,null,2);actions.replaceChildren();if(v.state==='AWAITING_AUTHORIZATION'){for(let a of [true,false]){let b=document.createElement('button');b.textContent=a?'Approve generation':'Deny';b.onclick=()=>run(a);actions.append(b)}}if(v.state==='COMPLETED'){let a=document.createElement('a');a.href='/artifact/'+v.id;a.textContent='Export generated file';actions.append(a);if(v.plan.capability==='calculator'){let input=document.createElement('input');input.placeholder='(2+3)*4';let result=document.createElement('pre');let b=document.createElement('button');b.textContent='Calculate';b.onclick=async()=>{try{result.textContent=(await api('/calculate',{id:v.id,expression:input.value})).result}catch(e){result.textContent=e.message}};actions.append(input,b,result)}if(v.plan.capability==='website'){let f=document.createElement('iframe');f.sandbox='allow-scripts';f.src='/preview/'+v.id;actions.append(f)}}}async function create(){try{show(await api('/create',{requirement:req.value}));await refresh()}catch(e){out.textContent=e.message}}async function run(approve){try{out.textContent='Executing authorized task…';show(await api('/execute',{id:current.id,approve}));await refresh()}catch(e){out.textContent=e.message}}async function refresh(){history.replaceChildren();for(let v of await api('/history')){let b=document.createElement('button');b.textContent=v.plan.requirement+' — '+v.state;b.onclick=()=>show(v);history.append(b)}}refresh();</script>'''
def serve(root,port=8768):
    engine=SecureEngine(root); session=secrets.token_hex(32)
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,body,kind='application/json',extra=None):
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            if extra:self.send_header(*extra)
            self.end_headers();self.wfile.write(body)
        def js(self,v,status=200):self.send(status,json.dumps(v).encode())
        def do_GET(self):
            try:
                if self.headers.get('Host')!=f'127.0.0.1:{port}':raise ValueError('Invalid host')
                if self.path=='/health':return self.js({'status':'ready','port':port})
                if self.path=='/':return self.send(200,PAGE.replace('TOKEN',session).encode(),'text/html; charset=utf-8')
                if self.path=='/history' and self.headers.get('X-Session')==session:return self.js(engine.history())
                if self.path.startswith(('/artifact/','/preview/')):
                    f=engine.artifact(self.path.split('/')[-1]); preview=self.path.startswith('/preview/') and f.suffix=='.html';return self.send(200,f.read_bytes(),'text/html' if preview else 'application/octet-stream',('Content-Security-Policy',"sandbox allow-scripts; default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'") if preview else ('Content-Disposition',f'attachment; filename="{f.name}"'))
                self.js({'error':'Not found'},404)
            except Exception as e:self.js({'error':str(e)},400)
        def do_POST(self):
            try:
                if self.headers.get('Host')!=f'127.0.0.1:{port}' or self.headers.get('X-Session')!=session:raise ValueError('Invalid session')
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=8192:raise ValueError('Invalid body size')
                d=json.loads(self.rfile.read(n))
                if self.path=='/calculate':
                    task=d['id']; r=engine.get(task)
                    if r['plan']['capability']!='calculator':raise ValueError('Not a calculator')
                    engine.artifact(task)
                    return self.js({'result':calculate(d['expression'])})
                if self.path=='/create':task,_=engine.create(d['requirement'])
                elif self.path=='/execute':
                    task=d['id'];a=d['approve']
                    if a is True:engine.execute_token(task,engine.issue(task))
                    elif a is False:engine.execute(task,False)
                    else:raise ValueError('Explicit decision required')
                else:raise ValueError('Unknown endpoint')
                self.js(engine.get(task))
            except Exception as e:self.js({'error':str(e)},400)
    with ThreadingHTTPServer(('127.0.0.1',port),Handler) as server:
        port=server.server_address[1]
        print(json.dumps({'status':'ready','url':f'http://127.0.0.1:{port}'}),flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path.home()/'.project-s'));p.add_argument('--port',type=int,default=8768);a=p.parse_args();serve(a.root,a.port)
