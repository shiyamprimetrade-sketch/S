"""Reviewed source-only upgrade for a stopped existing S native-console app.

Default is a dry run. Only known baseline hashes may be replaced. Local DB,
weights and host settings never leave the PC and are not replaced.
"""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import tempfile
import uuid

SOURCE=Path(__file__).resolve().parents[1]
# Known native-console checkpoint and initial Developer checkpoint only.
KNOWN = {
  "continuum/server.py": [
    "6e75cf62dd4d1182a415f5257402cef7a0583b566a04a6abe099450fbc887c98",
    "b7cdc4d44e9f1faba14cfa24d864ce5c4d3275cf46245690290a3b207a3d8497"
  ],
  "web/index.html": [
    "48eaf288b56a8663a991f1c5cd3b2b5b277004e8bd92fbdd5b3eae259b180961",
    "bed166269016f200631a7e5a84b2d57d13f7bc28edfd4a6d6ef210dbe1faf01d"
  ],
  "web/brain.html": [
    "4f095eb99ca3a2639cd8acf236c122bb05d04e00133400159cf06bba2fae74e5",
    "76beaabf49317bf58787ee2a3a7dced6e2c28a2ca4f62fcad0f7f19f42e8319f"
  ],
  "web/developer.html": [
    "74e963c2e54b0c72e94091e0aef37c3898a2dff21ba035face831bed0dec5e00"
  ],
  "web/developer.css": [
    "a49c7221306c5faf7e59a22a7e5bddbd4d49bd32a4088912fcd223770bdb7ea0"
  ],
  "web/developer.js": [
    "472739c707aa1bdad16bcc9948226447596861a10b4d44b01e923f478c137e5b"
  ],
  "scripts/S.cmd": [
    "c91693105f5257e1ba466bf3931b69ec2db77710fabd68fd67904197fe0e6527"
  ],
  "scripts/s_launcher.py": [
    "a195ef5727407b2e6556012ffe5f10f48d8e2837139142918635f05decae3b8f"
  ],
  "s_developer/__init__.py": [
    "320c369629ff9f1e170a76e111160b3f32929493752566c0502a3e9a0e21d70a"
  ],
  "s_developer/cli.py": [
    "dee501f2da72c7c43c8d5c015eddda9abfc2beac9542d79722638be028ef1bc8"
  ],
  "s_developer/intent.py": [
    "2a832a82d132a606eefbbd4b8370296e56b17af323e057257cbc1bb25b57f004"
  ],
  "s_developer/registry.py": [
    "ba5811b55557e1f23926a77846462ea7ef0cc66a540fdbdd6fe04a1deafec2e5"
  ],
  "s_developer/specifications.py": [
    "2437b57e452ec18edf708f9c2822fa971659daa8b0df78762c84a6a905c39d73"
  ]
}


def digest(path):
    # Git/Windows may check out the same reviewed text with CRLF endings.
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def bounded(root, relative):
    if relative not in KNOWN:
        raise ValueError('Unexpected upgrade file')
    path=root/relative
    for parent in (path,*path.parents):
        if parent==root:break
        if parent.is_symlink():raise ValueError('Symlink in upgrade path')
    if not path.resolve().is_relative_to(root):raise ValueError('File outside app boundary')
    return path


def plan(root, source=SOURCE):
    root,source=Path(root).resolve(),Path(source).resolve()
    if root==source:raise ValueError('Use a separate checkout as the upgrade source')
    if not (root/'continuum'/'storage.py').is_file():raise ValueError('Not an existing S source installation')
    result=[]
    for name,known in KNOWN.items():
        incoming=source/name
        if not incoming.is_file():raise ValueError('Upgrade source missing '+name)
        target=bounded(root,name)
        before=digest(target) if target.is_file() else None
        after=digest(incoming)
        if target.exists() and not target.is_file():raise ValueError('Not a regular file: '+name)
        if before is not None and before not in known and before!=after:
            raise ValueError('Unrecognized local changes; preserved '+name)
        result.append(dict(path=name,before=before,after=after,change=before!=after))
    return result


def atomic_copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as handle:
            temporary=Path(handle.name)
        shutil.copy2(source,temporary)
        os.replace(temporary,target)
    finally:
        if temporary and temporary.exists():temporary.unlink()


def apply(root,source=SOURCE):
    root,source=Path(root).resolve(),Path(source).resolve()
    items=plan(root,source)
    changes=[x for x in items if x['change']]
    if not changes:return {'changed':0,'backup':None}
    parent=root/'upgrade-backups'
    if parent.is_symlink():raise ValueError('Symlinked backup directory')
    backup=parent/('s-developer-'+uuid.uuid4().hex)
    backup.mkdir(parents=True)
    manifest={'format':'s-developer-upgrade-v1','root':str(root),'files':changes}
    for item in changes:
        if item['before'] is not None:
            target=backup/'files'/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(bounded(root,item['path']),target)
    # A local-only SQLite snapshot, never part of Git or the downloaded source.
    db=root/'data'/'continuum.sqlite3'
    if db.is_symlink() or db.parent.is_symlink():raise ValueError('Symlinked database; back up manually')
    if db.is_file():
        with closing(sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True)) as original, closing(sqlite3.connect(backup/'continuum.sqlite3')) as copy:
            original.backup(copy)
            if copy.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Backup integrity failure')
    (backup/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    changed=[]
    try:
        for item in changes:
            target=bounded(root,item['path'])
            before=digest(target) if target.is_file() else None
            if before!=item['before']:raise ValueError('App changed during upgrade')
            if digest(source/item['path'])!=item['after']:raise ValueError('Upgrade source changed')
            atomic_copy(source/item['path'],target)
            changed.append(item)
    except BaseException:
        for item in reversed(changed):
            target=bounded(root,item['path'])
            if item['before'] is None:target.unlink()
            else:atomic_copy(backup/'files'/item['path'],target)
        raise
    return {'changed':len(changes),'backup':str(backup)}


def rollback(root, backup):
    root,backup=Path(root).resolve(),Path(backup).resolve()
    parent=root/'upgrade-backups'
    if not backup.is_relative_to(parent.resolve()):raise ValueError('Backup outside app upgrade directory')
    manifest=json.loads((backup/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format')!='s-developer-upgrade-v1' or manifest.get('root')!=str(root):
        raise ValueError('Backup belongs to another app')
    for item in manifest['files']:
        target=bounded(root,item['path'])
        if not target.is_file() or digest(target)!=item['after']:
            raise ValueError('Post-upgrade edits detected; rollback stopped')
        if item['before'] is not None and digest(backup/'files'/item['path'])!=item['before']:
            raise ValueError('Backup checksum mismatch')
    for item in reversed(manifest['files']):
        target=bounded(root,item['path'])
        if item['before'] is None:target.unlink()
        else:atomic_copy(backup/'files'/item['path'],target)
    return {'restored_files':len(manifest['files']),'database_restored':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app-root',required=True,type=Path)
    parser.add_argument('--apply',action='store_true',help='Apply reviewed changes, otherwise only print a plan')
    parser.add_argument('--stopped',action='store_true',help='Assert that all S processes using this installation are stopped')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--rollback',type=Path)
    args=parser.parse_args()
    if not args.apply:
        if args.rollback:parser.error('Rollback requires --apply --stopped')
        print(json.dumps(plan(args.app_root),indent=2));return
    if not args.stopped:parser.error('Stop S first, then explicitly pass --stopped')
    try:
        with socket.create_connection(('127.0.0.1',args.port),timeout=1):
            parser.error('A server is still listening on the selected S port; stop it before upgrading')
    except OSError:pass
    print(json.dumps(rollback(args.app_root,args.rollback) if args.rollback else apply(args.app_root),indent=2))


if __name__=='__main__':main()
