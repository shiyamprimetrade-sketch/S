"""Command line entrypoint. Usage: python -m continuum.cli --help"""
import argparse
import json
import os
import sqlite3
from datetime import datetime,timezone
from pathlib import Path
from .storage import Store
from .engine import ollama_models,DEFAULT_MODEL

ROOT=Path(__file__).resolve().parent.parent
DB=Path(os.environ.get('AXIOM_DB',str(ROOT/'data'/'continuum.sqlite3')))

def main(argv=None):
    p=argparse.ArgumentParser(prog='axiomflow',description='Offline, model-independent chat and research memory')
    p.add_argument('--db',default=str(DB),help='SQLite database path')
    sub=p.add_subparsers(dest='cmd',required=True)
    server=sub.add_parser('serve',help='Start local browser app')
    server.add_argument('--port',type=int,default=8765)
    sub.add_parser('doctor',help='Report model and database status')
    sub.add_parser('init',help='Initialize with teaching notes')
    backup=sub.add_parser('backup',help='Checkpoint database to a safe copy')
    backup.add_argument('--output')
    export=sub.add_parser('export',help='Export one project to portable JSON')
    export.add_argument('--project',required=True)
    export.add_argument('--output',required=True)
    handoff=sub.add_parser('handoff',help='Create portable Markdown prompt')
    handoff.add_argument('--project',required=True)
    handoff.add_argument('--output',required=True)
    cg=sub.add_parser('import-chatgpt',help='Import ChatGPT conversations.json or data-export ZIP into private SQLite')
    cg.add_argument('input');cg.add_argument('--title',default='',help='Only import conversations with this text in their title')
    cg.add_argument('--unredacted',action='store_true',help='Preserve sensitive tokens locally; never commit database')
    imp=sub.add_parser('import',help='Import a JSON project pack (keeps original projects)')
    imp.add_argument('input')
    rest=sub.add_parser('restore',help='Replace DB from verified snapshot (server must be stopped)')
    rest.add_argument('snapshot')
    rest.add_argument('--confirm',action='store_true')
    args=p.parse_args(argv)
    if args.cmd=='restore':
        if not args.confirm:p.error('Restore overwrites the database. Stop server and pass --confirm')
        snap=Path(args.snapshot)
        if not snap.is_file():p.error('Snapshot missing')
        with sqlite3.connect(f'file:{snap.resolve()}?mode=ro',uri=True) as check:
            if check.execute('PRAGMA integrity_check').fetchone()[0]!='ok':p.error('Snapshot failed SQLite integrity')
        if Path(args.db).exists():
            Store(args.db).backup(str(args.db)+'.pre-restore.sqlite3')
        import shutil
        shutil.copy2(snap,args.db)
        print('Restored; old version saved as .pre-restore.sqlite3')
        return
    s=Store(args.db)
    s.seed_if_empty()
    if args.cmd=='serve':
        from .server import serve
        serve(args.db,port=args.port)
    elif args.cmd=='init':print('Initialized',args.db)
    elif args.cmd=='doctor':
        print('Database:',s.path)
        print('Projects:',len(s.list_projects()))
        print('Ollama models:',', '.join(ollama_models()) or 'none detected')
        print('Requested default:',DEFAULT_MODEL)
        with s.connect() as db:print('Database integrity:',db.execute('PRAGMA integrity_check').fetchone()[0])
    elif args.cmd=='backup':
        path=args.output or str(ROOT/'backups'/('continuum-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'.sqlite3'))
        print('Verified backup:',s.backup(path))
    elif args.cmd=='export':
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(s.export_project(args.project),ensure_ascii=False,indent=2),encoding='utf8')
        print('Exported:',out)
    elif args.cmd=='handoff':
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(s.handoff(args.project),encoding='utf8')
        print('Handoff capsule:',out)
    elif args.cmd=='import-chatgpt':
        from .transcript import load_export
        document=load_export(args.input)
        print(json.dumps(s.import_chatgpt(document,args.title,not args.unredacted),indent=2))
    elif args.cmd=='import':
        data=json.loads(Path(args.input).read_text(encoding='utf8'))
        print('Imported:',s.import_project(data)['name'])

if __name__=='__main__':main()
