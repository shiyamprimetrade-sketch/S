import unittest, subprocess, sys, tempfile, json, urllib.request, re, time, pathlib, os, signal
class HTTPProduct(unittest.TestCase):
    def test_cold_starts_and_workflow(self):
        with tempfile.TemporaryDirectory() as root:
            ids=[]
            for cycle in range(4):
                p=subprocess.Popen([sys.executable,'-m','project_s.local_ui','--root',root,'--port','0'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                try:
                    import selectors
                    with selectors.DefaultSelector() as sel:
                        sel.register(p.stdout,selectors.EVENT_READ)
                        self.assertTrue(sel.select(15),'startup timeout')
                        base=json.loads(p.stdout.readline())['url']
                    def get(path):return urllib.request.urlopen(base+path,timeout=10).read()
                    self.assertEqual(json.loads(get('/health'))['status'],'ready')
                    token=re.search(rb"const token='([a-f0-9]+)'",get('/')).group(1).decode()
                    def api(path,data=None):
                        req=urllib.request.Request(base+path,data=json.dumps(data).encode() if data is not None else None,headers={'X-Session':token,'Content-Type':'application/json'})
                        return json.load(urllib.request.urlopen(req,timeout=20))
                    self.assertEqual(len(api('/history')),len(ids))
                    for task in ids:self.assertTrue(get('/artifact/'+task))
                    if cycle==3:
                        self.assertEqual(api('/inventory',{'id':ids[2],'action':'search','name':'Apple'})['products'],[['Apple',7]])
                        continue
                    v=api('/create',{'requirement':['Create a calculator','Create a website for Cedar Studio','Create an inventory'][cycle]})
                    self.assertEqual(v['state'],'AWAITING_AUTHORIZATION')
                    v=api('/execute',{'id':v['id'],'approve':True});self.assertEqual(v['state'],'COMPLETED');self.assertTrue(v['evidence']['passed']);self.assertTrue(get('/artifact/'+v['id']));ids.append(v['id'])
                    if cycle==0:
                        self.assertEqual(api('/calculate',{'id':v['id'],'expression':'(2+3)*4'})['result'],'20')
                        self.assertEqual(api('/calculate',{'id':v['id'],'expression':'0.1+0.2'})['result'],'0.3')
                    if cycle==2:
                        self.assertEqual(api('/inventory',{'id':v['id'],'action':'register','name':'Apple','stock':4})['products'],[['Apple',4]])
                        self.assertEqual(api('/inventory',{'id':v['id'],'action':'adjust','name':'Apple','stock':3})['products'],[['Apple',7]])
                    if cycle==1:self.assertIn(b'Cedar Studio',get('/preview/'+v['id']))
                finally:
                    p.send_signal(signal.SIGINT);p.communicate(timeout=10)
                    self.assertEqual(p.returncode,0)
if __name__=='__main__':unittest.main()
