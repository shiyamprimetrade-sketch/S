"""S command launcher: python -m s_developer.cli develop application."""
import argparse
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlencode
import webbrowser
from continuum.cli import DB, ROOT
from continuum.server import AppServer
from .intent import parse_intent
from .registry import catalog, detect

DEFAULT_DB = DB if (ROOT/'S_PROJECT.md').is_file() else Path(os.environ.get(
    'AXIOM_DB', str(Path(os.environ.get('LOCALAPPDATA', str(Path.home()/'.local'/'share')))/'S'/'continuum.sqlite3')))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='S', description='S unified development foundation')
    parser.add_argument('--db', default=str(DEFAULT_DB))
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('command', nargs='+', help='develop application | build me a website | toolchains')
    parser.add_argument('--detect', metavar='LANGUAGE', help='Explicitly probe a trusted installed toolchain')
    args = parser.parse_args(argv)
    if args.command == ['toolchains']:
        print(json.dumps(detect(args.detect) if args.detect else catalog(), indent=2))
        return 0
    if args.detect:
        parser.error('--detect is only supported with toolchains')
    try:
        intent = parse_intent(' '.join(args.command))
    except ValueError as error:
        parser.error(str(error))
    # Binding before opening the browser prevents accidentally using an unrelated
    # service on an occupied port. Never silently reuses or kills an existing app.
    try:
        app = AppServer('127.0.0.1', args.port, args.db)
    except OSError as error:
        parser.exit(2, f'Cannot start S Developer: {error}. Choose another --port.\n')
    url = f'http://127.0.0.1:{app.server_port}/developer#' + urlencode({'request': intent['request']})
    print(f'S Developer: {url}', flush=True)
    print('Requirements only. No code, tools or downloads execute from this workspace.', flush=True)
    # Hash fragment keeps the original request out of HTTP URLs/access logs.
    with app:
        if not args.no_browser:
            threading.Timer(0.2, lambda: webbrowser.open(url)).start()
        try:
            app.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
