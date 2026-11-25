import tempfile, unittest
from pathlib import Path
from project_s.secure import SecureEngine

class RecoveryTests(unittest.TestCase):
    def test_interrupted_generation(self):
        with tempfile.TemporaryDirectory() as root:
            e=SecureEngine(root)
            task,_=e.create('Create calculator')
            e.transition(task,'AWAITING_AUTHORIZATION','AUTHORIZED')
            e.transition(task,'AUTHORIZED','GENERATING')
            d=Path(root)/task
            d.mkdir()
            f=d/'calculator.py'
            f.write_text('partial')
            e=SecureEngine(root)
            self.assertEqual(e.recover_interrupted(),[task])
            self.assertEqual(e.get(task)['state'],'FAILED')
            self.assertEqual(f.read_text(),'partial')
            self.assertEqual(e.recover_interrupted(),[])
            with self.assertRaises(ValueError):e.issue(task)

    def test_forced_process_termination(self):
        import subprocess,sys,json
        from project_s.workspace_lock import workspace_lock
        with tempfile.TemporaryDirectory() as root:
            code="""import sys,time
from pathlib import Path
from project_s.workspace_lock import workspace_lock
from project_s.secure import SecureEngine
with workspace_lock(sys.argv[1]):
    e=SecureEngine(sys.argv[1])
    task,_=e.create('Create calculator')
    e.transition(task,'AWAITING_AUTHORIZATION','AUTHORIZED')
    e.transition(task,'AUTHORIZED','GENERATING')
    d=Path(sys.argv[1])/task
    d.mkdir()
    (d/'calculator.py').write_text('partial')
    print(task,flush=True)
    time.sleep(60)
"""
            p=subprocess.Popen([sys.executable,'-c',code,root],stdout=subprocess.PIPE,text=True)
            try:
                task=p.stdout.readline().strip()
                self.assertEqual(len(task),32)
                with self.assertRaises(OSError):
                    with workspace_lock(root):pass
            finally:
                p.kill()
                p.communicate(timeout=10)
            server=subprocess.Popen([sys.executable,'-m','project_s.local_ui','--root',root,'--port','0'],stdout=subprocess.PIPE,text=True)
            try:
                self.assertEqual(json.loads(server.stdout.readline())['status'],'ready')
                e=SecureEngine(root)
                self.assertEqual(e.get(task)['state'],'FAILED')
                self.assertEqual((Path(root)/task/'calculator.py').read_text(),'partial')
            finally:
                server.kill()
                server.communicate(timeout=10)
