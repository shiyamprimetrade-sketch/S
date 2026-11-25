import subprocess,sys,hashlib,tempfile,json,os
from pathlib import Path
from engine.arithmetic_verifier import ArithmeticVerifier
from html.parser import HTMLParser

def verify(path,capability):
    path=Path(path).resolve(); evidence=[]
    if capability in ('calculator','inventory'):
        trusted=Path(__file__).with_name(capability+'.py').read_bytes()
        if path.read_bytes()!=trusted:raise ValueError('Untrusted generated Python; execution refused')
    if capability=='calculator':
        code='import calculator as c; import json; print(json.dumps([c.calculate(x) for x in ["(2+3)*4","0.1+0.2","9/4","-3+2"]]))'
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'calculator.py').write_bytes(trusted)
            out=subprocess.check_output([sys.executable,'-I','-c','import sys; sys.path.insert(0,sys.argv[1]); '+code,tmp],cwd=tmp,timeout=10,text=True,env={k:v for k,v in os.environ.items() if k in ('SYSTEMROOT','WINDIR','PATH','TEMP','TMP')})
        for expr,result in zip(['(2+3)*4','0.1+0.2','9/4','-3+2'],json.loads(out),strict=True):
            evidence.append(ArithmeticVerifier().verify({'expression':expr,'result':result,'precision':2})['verified'])
    elif capability=='inventory':
        with tempfile.TemporaryDirectory() as tmp:
            code='from inventory import Inventory; import sys; i=Inventory(sys.argv[1]); i.register("Apple",4); i.adjust("Apple",3); assert Inventory(sys.argv[1]).search("app")==[("Apple",7)]'
            Path(tmp,'inventory.py').write_bytes(trusted)
            subprocess.run([sys.executable,'-I','-c','import sys; sys.path.insert(0,sys.argv.pop(1)); '+code,tmp,str(Path(tmp)/'test.db')],cwd=tmp,timeout=10,check=True,env={k:v for k,v in os.environ.items() if k in ('SYSTEMROOT','WINDIR','PATH','TEMP','TMP')}); evidence.append(True)
    elif capability=='website':
        class Parser(HTMLParser):
            def __init__(self): super().__init__(); self.ids=set(); self.links=[]
            def handle_starttag(self,tag,attrs):
                a=dict(attrs)
                if 'id' in a:self.ids.add(a['id'])
                if tag=='a':self.links.append(a.get('href',''))
        p=Parser(); p.feed(path.read_text()); evidence=[{'services','contact'}<=p.ids,all(x.startswith('#') and x[1:] in p.ids for x in p.links),len(p.links)==2]
    else: raise ValueError('Unsupported capability')
    return {'passed':bool(evidence) and all(evidence),'checks':evidence,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'method':'recovered independent Decimal AST' if capability=='calculator' else 'external functional subprocess' if capability=='inventory' else 'static HTML structure only'}
