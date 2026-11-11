"""Install additive S-Brain integration into a stopped, existing S v0.6 app."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil

SOURCE = Path(__file__).resolve().parents[1]


def install(root, checkpoint):
    root, checkpoint = Path(root).resolve(), Path(checkpoint).resolve()
    from s_brain.checkpoint import load_weights
    model, manifest = load_weights(checkpoint)
    if manifest.step < 1:
        raise ValueError('A trained, verified checkpoint is required')
    targets = [root/'s_brain', root/'s-brain-host.json', root/'continuum/brain_console.py']
    if any(p.exists() for p in targets):
        raise FileExistsError('Existing integration detected; preserve it for an explicit upgrade')
    server = (root/'continuum/server.py').read_text(encoding='utf8')
    if 'self.brain=' in server:
        raise ValueError('Already integrated')
    anchors = {
        "        if name not in allow:": "        allow.update({'/brain':'brain.html','/brain.js':'brain.js','/brain.css':'brain.css'})\n        if name not in allow:",
        "        s=self.server.store\n": "        s=self.server.store\n        if path.startswith('/api/brain/'):\n            return self.server.brain.request(verb,path,self._query(),None if verb=='GET' else self._body())\n",
        "            if path=='/api/chat':return chat(": "            if path=='/api/chat' and b.get('model')=='s-brain:experimental':return self.server.brain.chat(b)\n            if path=='/api/chat':return chat(",
        "        self.store.seed_if_empty()": "        self.store.seed_if_empty()\n        from .brain_console import BrainConsole\n        self.brain=BrainConsole(self.store)",
    }
    for before, after in anchors.items():
        if server.count(before) != 1:
            raise ValueError('Unexpected server layout; refusing blind patch')
        server = server.replace(before, after)
    html = (root/'web/index.html').read_text(encoding='utf8')
    if html.count('</body>') != 1:
        raise ValueError('Unexpected HTML layout')
    html = html.replace('</body>','<a href="/brain">S Native, Memory, Tools and Learning</a></body>')
    backup = root/'upgrade-backups'/('s-brain-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    for name in ('continuum/server.py','web/index.html'):
        target=backup/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target)
    shutil.copytree(SOURCE/'s_brain',root/'s_brain',ignore=shutil.ignore_patterns('checkpoints','__pycache__'))
    destination=root/'s_brain/checkpoints/integrated'
    shutil.copytree(checkpoint,destination)
    shutil.copy2(SOURCE/'continuum/brain_console.py',root/'continuum/brain_console.py')
    for name in ('brain.html','brain.js','brain.css'):
        shutil.copy2(SOURCE/'web'/name,root/'web'/name)
    (root/'continuum/server.py').write_text(server,encoding='utf8')
    (root/'web/index.html').write_text(html,encoding='utf8')
    # Host enables only the reviewed read-only tool set; every call needs UI approval.
    (root/'s-brain-host.json').write_text(json.dumps({'enabled':True,'memory_enabled':True,
        'tools_enabled':True,'checkpoint':'s_brain/checkpoints/integrated'},indent=2))
    print('Installed verified experimental S-Brain; backup:',backup)


if __name__ == '__main__':
    import sys
    sys.path.insert(0,str(SOURCE))
    parser=argparse.ArgumentParser()
    parser.add_argument('root');parser.add_argument('checkpoint')
    args=parser.parse_args();install(args.root,args.checkpoint)
