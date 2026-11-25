import hashlib,json,os,subprocess,sys,tempfile,venv
from pathlib import Path
from urllib.request import Request,urlopen
def main():
    wheel,=Path('dist').glob('*.whl')
    wheel=wheel.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        venv.EnvBuilder(with_pip=True).create(root/'venv')
        python=root/'venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        subprocess.run([str(python),'-m','pip','install','--no-index','--no-deps',str(wheel)],cwd=root,check=True)
        launcher=python.parent/('project-s.exe' if os.name=='nt' else 'project-s')
        p=subprocess.Popen([str(launcher),'--root',str(root/'workspace'),'--port','0'],cwd=root,stdout=subprocess.PIPE,text=True)
        try:
            url=json.loads(p.stdout.readline())['url']
            page=urlopen(url,timeout=10).read().decode()
            token=page.split("const token='",1)[1].split("'",1)[0]
            def post(path,data):
                return json.load(urlopen(Request(url+path,json.dumps(data).encode(),{'Content-Type':'application/json','X-Session':token}),timeout=20))
            for capability in ('calculator','website','inventory'):
                task=post('/create',{'requirement':'Create '+capability})
                result=post('/execute',{'id':task['id'],'approve':True})
                assert result['state']=='COMPLETED',result
                print('PASS installed',capability,flush=True)
            print('WHEEL SHA256',hashlib.sha256(wheel.read_bytes()).hexdigest(),flush=True)
        finally:
            p.terminate();p.wait(timeout=10)
if __name__=='__main__':main()
