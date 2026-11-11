"""Fail-closed, digest-pinned Docker execution. No host-execution fallback."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from .bundles import RECIPES, validate_files, redact

IMAGE_TAGS={'python':'python:3.12-slim','gcc':'gcc:14'}
DRIVERS={
 'static-site': '''import http.server, threading, urllib.request
from html.parser import HTMLParser
class Document(HTMLParser):
    def __init__(self): super().__init__(); self.title=False
    def handle_starttag(self,tag,attrs):
        if tag=='title': self.title=True
p=Document(); p.feed(open('/src/index.html').read()); assert p.title, 'Missing HTML title'
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self,*a,**k):super().__init__(*a,directory='/src',**k)
    def log_message(self,*a):pass
s=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
threading.Thread(target=s.serve_forever,daemon=True).start()
try:
    with urllib.request.urlopen('http://127.0.0.1:%s/index.html'%s.server_port,timeout=3) as r:
        assert r.status==200; assert r.read(), 'Empty website'
finally:s.shutdown();s.server_close()
print('Static HTML parse and actual container-local HTTP GET passed; browser behavior not tested')
''',
 'python-cli': '''import sys,unittest
sys.path.insert(0,'/src')
suite=unittest.defaultTestLoader.discover('/src',pattern='test_*.py')
assert suite.countTestCases()>0, 'No tests discovered'
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
''',
 'c-cli': '''#!/bin/sh
set -eu
cc -std=c17 -Wall -Wextra -Werror /src/main.c -o /work/program
/work/program
''',
}

# No expected answers or hidden cases are mounted into the container. The host
# compares JSON output independently, rather than trusting generated test code.
EVALUATION_DRIVER='''import json,sys,importlib.util
payload=json.loads(sys.stdin.buffer.read(8193))
spec=importlib.util.spec_from_file_location('candidate','/src/app.py')
candidate=importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
answer=candidate.solve(payload)
print(json.dumps(answer,ensure_ascii=False,allow_nan=False))
'''


def docker_program():
    path=shutil.which('docker')
    if not path:raise RuntimeError('Docker is unavailable; builds are blocked. There is no unrestricted host fallback.')
    return path


def docker_env():
    # Never inherit a remote Docker endpoint, TLS selectors or project env secrets.
    keep=('PATH','SystemRoot','WINDIR','HOME','USERPROFILE','LOCALAPPDATA','APPDATA','TMPDIR','TEMP','TMP','DOCKER_CONFIG')
    return {k:v for k,v in __import__('os').environ.items() if k in keep}


def prepare_image(identifier,approve_download=False):
    if identifier not in IMAGE_TAGS:raise ValueError('Unknown official runtime image')
    tag=IMAGE_TAGS[identifier]
    if not approve_download:return {'image':tag,'download_required':True,'installed':False}
    executable=docker_program()
    # This explicit operator command is the only image-download path.
    result=subprocess.run([executable,'--context','default','pull',tag],env=docker_env(),
                          stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300,check=False)
    if result.returncode:raise RuntimeError('Official runtime image pull failed: '+redact(result.stdout.decode(errors='replace')))
    result=subprocess.run([executable,'--context','default','image','inspect',tag,'--format','{{json .RepoDigests}}'],
                          env=docker_env(),capture_output=True,text=True,timeout=15,check=True)
    candidates=json.loads(result.stdout)
    digest=next((s for s in candidates if re.fullmatch(r'(?:docker.io/library/)?'+identifier+r'@sha256:[0-9a-f]{64}',s)),None)
    if digest is None:raise ValueError('Runtime provenance has no official content-addressed digest')
    return {'image':digest,'installed':True,'download_required':False}


class DockerRunner:
    def __init__(self,images):self.images=images
    def command(self,executable,name,recipe,source,driver):
        key=RECIPES[recipe]['image'];image=self.images.get(key,'')
        if not re.fullmatch(r'(?:docker.io/library/)?'+key+r'@sha256:[0-9a-f]{64}',image):
            raise PermissionError('Operator must explicitly prepare and pin the official '+key+' image')
        for path in (source,driver):
            if ',' in str(path) or '\n' in str(path):raise ValueError('Unsupported container mount path')
        entry=['/bin/sh','/driver/run.sh'] if recipe=='c-cli' else ['python','-I','-S','/driver/run.py']
        return [executable,'--context','default','run','--rm','--pull=never','--name',name,
                '--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges',
                '--user=65534:65534','--pids-limit=64','--memory=512m','--memory-swap=512m','--cpus=1',
                '--log-driver=none','--ulimit','fsize=65536:65536','--ulimit','nofile=128:128',
                '--tmpfs','/tmp:rw,noexec,nosuid,size=32m,mode=1777',
                '--tmpfs','/work:rw,exec,nosuid,nodev,size=64m,mode=1777','--workdir=/work',
                '--mount',f'type=bind,src={source},dst=/src,readonly',
                '--mount',f'type=bind,src={driver},dst=/driver,readonly',
                '--env','HOME=/tmp','--env','HTTP_PROXY=','--env','HTTPS_PROXY=',
                '--env','ALL_PROXY=','--env','http_proxy=','--env','https_proxy=',
                '--env','all_proxy=','--env','NO_PROXY=','--env','no_proxy=',
                '--env','FTP_PROXY=','--env','ftp_proxy=',
                '--entrypoint='+entry[0],image,*entry[1:]]
    def run(self,files,recipe,timeout=30,evaluation_input=None):
        files=validate_files(files,recipe)
        payload=None
        if evaluation_input is not None:
            if recipe!='python-cli':raise ValueError('Independent evaluation currently supports Python only')
            payload=json.dumps(evaluation_input,allow_nan=False).encode('utf-8')
            if len(payload)>8192:raise ValueError('Evaluation input exceeds 8 KB')
        executable=docker_program()
        with tempfile.TemporaryDirectory(prefix='s-build-') as temporary:
            root=Path(temporary);source=root/'source';driver=root/'driver'
            source.mkdir();driver.mkdir()
            # The temporary parent and both mounts need read access for uid 65534.
            root.chmod(0o755);source.chmod(0o755);driver.chmod(0o755)
            for relative,content in files.items():
                target=source/relative;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text(content,encoding='utf-8');target.chmod(0o644)
            (driver/('run.sh' if recipe=='c-cli' else 'run.py')).write_text(EVALUATION_DRIVER if payload is not None else DRIVERS[recipe],encoding='utf-8')
            (driver/('run.sh' if recipe=='c-cli' else 'run.py')).chmod(0o644)
            name='s-build-'+uuid.uuid4().hex
            command=self.command(executable,name,recipe,source,driver)
            if payload is not None:command.insert(command.index('run')+1,'--interactive')
            pieces=[];overflow=threading.Event();limit=64000
            start=time.monotonic();process=None
            try:
                process=subprocess.Popen(command,env=docker_env(),stdin=subprocess.PIPE if payload is not None else subprocess.DEVNULL,
                                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=False)
                writer=None
                if payload is not None:
                    def send_input():
                        try:process.stdin.write(payload);process.stdin.close()
                        except (OSError,ValueError):pass
                    writer=threading.Thread(target=send_input,daemon=True);writer.start()
                def reader():
                    size=0
                    while True:
                        chunk=process.stdout.read(4096)
                        if not chunk:break
                        remaining=limit-size;pieces.append(chunk[:max(0,remaining)]);size+=len(chunk)
                        if size>limit:overflow.set();break
                thread=threading.Thread(target=reader,daemon=True);thread.start()
                timed_out=False
                while process.poll() is None:
                    if overflow.is_set() or time.monotonic()-start>timeout:
                        timed_out=time.monotonic()-start>timeout;process.kill();break
                    time.sleep(0.02)
                process.wait(timeout=5)
            finally:
                # Killing a CLI alone does not stop its container. Always remove it.
                subprocess.run([executable,'--context','default','rm','-f',name],env=docker_env(),
                               stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10,check=False)
                if process is not None:
                    thread.join(timeout=5)
                    process.stdout.close()
                    if writer is not None:
                        writer.join(timeout=5)
                        try:process.stdin.close()
                        except (OSError,ValueError):pass
            output=redact(b''.join(pieces).decode('utf-8',errors='replace'))
            status='toolchain_passed' if process.returncode==0 and not timed_out and not overflow.is_set() else 'failed'
            return {'status':status,'returncode':process.returncode,'output':output,'timed_out':timed_out,
                    'output_limit_exceeded':overflow.is_set(),'elapsed_seconds':round(time.monotonic()-start,3),
                    'isolation':'docker-network-none-readonly','image':self.images[RECIPES[recipe]['image']],
                    'business_acceptance':'unverified; toolchain/test exit status is not complete requirements acceptance'}
