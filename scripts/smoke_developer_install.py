"""Exercise an installed S entry point outside the repository, with a fresh DB.

Stdlib only. Does not assert visual browser behavior or clean-Windows installer H.
Usage: python scripts/smoke_developer_install.py --launcher /path/to/S
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import queue
import threading
from urllib.request import Request, urlopen


def request(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    with urlopen(Request(base+path, data=data, headers={'Content-Type':'application/json'}), timeout=10) as response:
        raw=response.read()
        return json.loads(raw) if path.startswith('/api/') else raw


def launch(executable, directory):
    process = subprocess.Popen([executable,'--db',str(directory/'s.sqlite3'),'--port','0',
                                '--no-browser','develop','application'], cwd=directory,
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    # Bound startup even when the child never prints its expected URL.
    output = queue.Queue()
    threading.Thread(target=lambda: output.put(process.stdout.readline()), daemon=True).start()
    try:
        line = output.get(timeout=15)
    except queue.Empty:
        stop(process)
        raise RuntimeError('Installed launcher did not print a URL within 15 seconds')
    match = re.search(r'(http://127\.0\.0\.1:\d+)/developer#request=develop\+application',line)
    if not match:
        process.terminate()
        _,error=process.communicate(timeout=10)
        raise RuntimeError('Installed launcher failed: '+line+error)
    return process,match.group(1)


def stop(process):
    process.terminate()
    process.communicate(timeout=10)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--launcher',default=shutil.which('S'))
    args=parser.parse_args()
    if not args.launcher: parser.error('Install S first or pass --launcher')
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        install=subprocess.run([args.launcher,'--db',str(directory/'s.sqlite3'),'model','install-demo'],
                               cwd=directory,capture_output=True,text=True,timeout=20)
        assert install.returncode==0,install.stderr
        assert json.loads(install.stdout)['installed'] is True
        process,base=launch(args.launcher,directory)
        try:
            assert b'requirements-form' in request(base,'/developer')
            assert b'function save' in request(base,'/developer.js')
            spec=request(base,'/api/developer/spec',dict(name='Installed smoke test',request='S develop application',
                         answers={'purpose':'Save a project specification'}))
            assert spec['revision']==1 and len(spec['clarifications'])==6
        finally:stop(process)
        process,base=launch(args.launcher,directory)
        try:
            reopened=request(base,'/api/developer/spec?project_id='+spec['project_id'])
            assert reopened==spec
            assert request(base,'/api/developer/history?project_id='+spec['project_id'])==[spec]
        finally:stop(process)
    print('PASS: installed S, packaged neural checkpoint install, UI assets, HTTP specification save, restart and checkpoint reopen')


if __name__=='__main__':main()
