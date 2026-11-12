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
    "b7cdc4d44e9f1faba14cfa24d864ce5c4d3275cf46245690290a3b207a3d8497",
    "9a96271fd10d5dd8458abc854a3d6a0caaddfaec38a42d3ee4c2571efacf6735",
    "a6c964332d25eb0ef017091b28bcc857c9a137d40ce404279ec3ee707c202fc8"
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
    "b31e725d2bf38a3b340de46851285e026b1c9ada4adeab9b43cf3b4f797ba758",
    "74e963c2e54b0c72e94091e0aef37c3898a2dff21ba035face831bed0dec5e00",
    "8cbf3da3bfa655f07b91e5a1fd95dafba1967277aee2680eb1a5bd70b1409b1f",
    "0edb1088dc0b8c5c8e3b8a3f3d441aa88a8e438eb29eff5fd73334ab71bbdb29"
  ],
  "web/developer.css": [
    "a49c7221306c5faf7e59a22a7e5bddbd4d49bd32a4088912fcd223770bdb7ea0",
    "0c150626a51954a4315792c8142ba46a76f53610539e84f469c67052c11ed4d9"
  ],
  "web/developer.js": [
    "472739c707aa1bdad16bcc9948226447596861a10b4d44b01e923f478c137e5b",
    "e3bccdd54c6dae9387395efac425b5fba6444f04f716be1149f0c7ce326a8302"
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
    "15f1d9bed5b710b5f969740e43d0ea7126c0ae897140fd28f23514f3354a6b2d",
    "c27032061859343010adab833eee19bc72159fb038c4068af3b04f1b35a43700",
    "7c89726a8f62c74b0487af58c75aac155b5e88dc87eea805e1c045de3f0b94ea",
    "dee501f2da72c7c43c8d5c015eddda9abfc2beac9542d79722638be028ef1bc8",
    "75cd99d9e5cfd83c88cd958dc9293a20c1d7b7682781d3ea0cdf374a91b9171b",
    "53c6e309dc7ea720906c2ac749f4871536d43892ec6323d7ee7f10025d47b15d",
    "aefc2f3ffde1a278a1418a5e96ce3804b5f70df429d0cfe79f059e1da04759e0"
  ],
  "s_developer/intent.py": [
    "2a832a82d132a606eefbbd4b8370296e56b17af323e057257cbc1bb25b57f004"
  ],
  "s_developer/registry.py": [
    "ba5811b55557e1f23926a77846462ea7ef0cc66a540fdbdd6fe04a1deafec2e5"
  ],
  "s_developer/specifications.py": [
    "2437b57e452ec18edf708f9c2822fa971659daa8b0df78762c84a6a905c39d73",
    "c0573bb341dc2b5c814652f37f104b002ebe54bd0d9fb4a20eb56d1748f15a52"
  ],
  "s_developer/bundles.py": [
    "9419dd4a48d35665c2e85fdc497b4d96aa1a82d1383f28d33e472af7e34da7e7"
  ],
  "s_developer/models.py": [
    "69457782c50f9eebf5e9577bc9ff4a1979955ee4a1e7b32d7c3b12b62f1ef67c",
    "343f6abea222f2f60b8277b7c14f48d5b74bd5b89224a3b5baf10d7c59a749f8",
    "1673c153b2963f76285df2944c450252275e3900afb8ad978361ea055b2330b1",
    "343f6abea222f2f60b8277b7c14f48d5b74bd5b89224a3b5baf10d7c59a749f8"
  ],
  "s_developer/engine.py": [
    "e4e167110213caeb4d387bc04b30c2cd02bfa5ea03ffaffa908153642ac26621",
    "375764f987a369b27e07c777e715e4b3f6074e7442d375a411e7f4b5531c9a63",
    "0681f3f6659ff99c04ca59df1c758073beba79b966aca36314d317c08df45e72"
  ],
  "s_developer/providers.py": [
    "fdc2a3c110dea1a82427ca82e15886deb871b5e18c4c9b40145ae95b7b18b96b",
    "16690bd478f6ee2c0c1f06d562688066c034b6d466e933e1972257045b1b6035",
    "a557611dcf3dac8ea723e5245edc98c84c046131779264008041927af753ca98",
    "ca8c2230fe12f96063157b55cdd3496cdfadc921bda4b445a13c2cc883c48c36",
    "ffa2f505f4a28862b344a8cde47894c13ea0bcec86a722f089856f9db2a31170"
  ],
  "s_developer/sandbox.py": [
    "81dc3f1cf57a703a5c0f1853c449eb5ab5e4f1a26e6931ccc98a2c652c2d44a2"
  ],
  "s_developer/resources/native-demo/manifest.json": [
    "f429931d11e206f26beabecab995e9fa73b084f04ab6800c96a43274f99ff1f3"
  ],
  "s_developer/resources/native-demo/training-report.json": [
    "0863eb9f75fa815bbef67010e1711de4e7ce99131c2f8776996100fe658c9330"
  ],
  "s_developer/resources/native-demo/model.safetensors": [
    "a6caeb1f219c01f42cec3ab54fb5dcdb61e20d6530dc91173a7805bdd7d89859"
  ],
  "web/coding.js": [
    "cd8185165c17bdaca0eb1ae2c97c6e767071693fed08a1a44cf58de7b0345f5f",
    "d0ec65bdfed903bee006e1d887dc1b1a626682d9adc7eaf424faf74dd9015ded",
    "dde899887d67b07d2a6b38ffcda2d87d8bee3b19ad85ad99789af872eeaed3c9"
  ],
  "s_developer/portable_native.py": [
    "b79d27da62e1e63d3fe50048edd050a4f2687dff61754f52524b6d2b7a2199be"
  ],
  "s_developer/capability.py": ["4c24e4e6965732099ace271578148d8fa4902698d2625c57a9f6d914c4a6fa07"],
  "s_developer/literal_repair.py": [],
  "s_developer/copy_models.py": [],
  "s_brain/byte_copy.py": [],
  "s_brain/span_copy.py": [],
  "s_brain/coding_decode.py": []
}


def digest(path):
    # Git/Windows may check out the same reviewed text with CRLF endings.
    return hashlib.sha256((Path(path).read_bytes() if Path(path).suffix=='.safetensors' else Path(path).read_bytes().replace(b'\r\n',b'\n'))).hexdigest()


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
