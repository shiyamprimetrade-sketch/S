import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from project_s.secure import SecureEngine
from project_s.repair import Repair

class RepairTests(unittest.TestCase):
    def test_repair_limits_and_preservation(self):
        with tempfile.TemporaryDirectory() as root:
            e=SecureEngine(root)
            task,_=e.create('Create calculator')
            e.transition(task,'AWAITING_AUTHORIZATION','FAILED',{'error':'injected defect'})
            d=Path(root)/task
            d.mkdir()
            original=d/'calculator.py'
            original.write_text('broken original')
            repair=Repair(e)
            proposal=repair.propose(task)
            with self.assertRaises(ValueError):repair.authorize(proposal,False)
            token=repair.authorize(proposal,True)
            tampered=dict(token,mac='0'*64)
            with self.assertRaises(ValueError):repair.execute(tampered)
            with patch('project_s.orchestrator.verify',side_effect=RuntimeError('injected verifier failure')):
                with self.assertRaises(RuntimeError):repair.execute(token)
            self.assertEqual(original.read_text(),'broken original')
            with self.assertRaises(ValueError):repair.execute(token)
            result=repair.execute(repair.authorize(repair.propose(task),True))
            self.assertEqual(result['status'],'VERIFIED')
            self.assertTrue(e.artifact(result['child']).is_file())
            repair.execute(repair.authorize(repair.propose(task),True))
            with self.assertRaises(ValueError):repair.propose(task)
            self.assertEqual(original.read_text(),'broken original')
            self.assertEqual(e.get(task)['state'],'FAILED')

    def test_concurrent_repair_and_chain(self):
        from concurrent.futures import ThreadPoolExecutor
        import sqlite3
        from contextlib import closing
        with tempfile.TemporaryDirectory() as root:
            e=SecureEngine(root)
            task,_=e.create('Create calculator')
            e.transition(task,'AWAITING_AUTHORIZATION','FAILED',{'error':'injected'})
            repair=Repair(e)
            token=repair.authorize(repair.propose(task),True)
            def attempt():
                try:return repair.execute(token)['status']
                except ValueError:return 'REJECTED'
            with ThreadPoolExecutor(max_workers=2) as pool:
                results=list(pool.map(lambda _:attempt(),range(2)))
            self.assertCountEqual(results,['VERIFIED','REJECTED'])
            with closing(sqlite3.connect(e.db)) as c:
                child=c.execute('SELECT child FROM repairs WHERE parent=?',(task,)).fetchone()[0]
            e.transition(child,'COMPLETED','FAILED',{'error':'injected successor defect'})
            with self.assertRaisesRegex(ValueError,'chain'):repair.propose(child)

    def test_interrupted_reservation(self):
        import sqlite3
        from contextlib import closing
        with tempfile.TemporaryDirectory() as root:
            e=SecureEngine(root)
            task,_=e.create('Create calculator')
            e.transition(task,'AWAITING_AUTHORIZATION','FAILED',{'error':'injected'})
            repair=Repair(e)
            child,_=e.create('Create calculator')
            with closing(sqlite3.connect(e.db)) as c,c:
                c.execute('INSERT INTO repairs VALUES (?,?,?,?)',(task,1,child,'RESERVED'))
            self.assertEqual(repair.reconcile(),1)
            self.assertEqual(repair.reconcile(),0)
            self.assertEqual(repair.propose(task)['attempt'],2)
            with closing(sqlite3.connect(e.db)) as c:
                self.assertEqual(c.execute('SELECT status FROM repairs WHERE parent=?',(task,)).fetchone()[0],'INTERRUPTED')

    def test_killed_repair_process(self):
        import subprocess,sys,queue,threading,json,sqlite3
        from contextlib import closing
        with tempfile.TemporaryDirectory() as root:
            e=SecureEngine(root)
            task,_=e.create('Create calculator')
            e.transition(task,'AWAITING_AUTHORIZATION','FAILED',{'error':'injected'})
            d=Path(root)/task
            d.mkdir()
            (d/'calculator.py').write_text('original')
            code="""import sys,time
from unittest.mock import patch
from project_s.secure import SecureEngine
from project_s.repair import Repair
from project_s.workspace_lock import workspace_lock
with workspace_lock(sys.argv[1]):
    e=SecureEngine(sys.argv[1])
    r=Repair(e)
    def pause(*args):
        print('reserved',flush=True)
        time.sleep(60)
    with patch.object(e,'execute_token',side_effect=pause):
        r.execute(r.authorize(r.propose(sys.argv[2]),True))
"""
            def read_ready(p):
                q=queue.Queue()
                threading.Thread(target=lambda:q.put(p.stdout.readline()),daemon=True).start()
                return q.get(timeout=15)
            p=subprocess.Popen([sys.executable,'-c',code,root,task],stdout=subprocess.PIPE,text=True)
            try:self.assertEqual(read_ready(p).strip(),'reserved')
            finally:
                p.kill()
                p.communicate(timeout=10)
            server=subprocess.Popen([sys.executable,'-m','project_s.local_ui','--root',root,'--port','0'],stdout=subprocess.PIPE,text=True)
            try:
                self.assertEqual(json.loads(read_ready(server))['status'],'ready')
                with closing(sqlite3.connect(e.db)) as c:
                    rows=c.execute('SELECT status FROM repairs WHERE parent=?',(task,)).fetchall()
                self.assertEqual(rows,[('INTERRUPTED',)])
                self.assertEqual(Repair(e).propose(task)['attempt'],2)
                self.assertEqual((d/'calculator.py').read_text(),'original')
            finally:
                server.kill()
                server.communicate(timeout=10)
