"""Install a reversible S source launcher into the current Termux account.

No downloads, pkg/pip installs, model execution or shell-profile edits. Run from
a reviewed S checkout after installing Python and git from Termux packages.
"""
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import uuid

MARKER = '# Managed by S Termux installer v1'
SOURCE = Path(__file__).resolve().parents[1]


def install(home, prefix, source, python):
    home, prefix, source = Path(home).resolve(), Path(prefix).resolve(), Path(source).resolve()
    python = Path(python).resolve()
    if not (source/'scripts'/'s_launcher.py').is_file():
        raise ValueError('S source checkout is incomplete')
    if not python.is_file():
        raise ValueError('Python executable is missing')
    bin_dir=home/'.local'/'bin'
    data_dir=home/'.local'/'share'/'S'
    for directory in (home/'.local',bin_dir,home/'.local'/'share',data_dir):
        if directory.is_symlink():
            raise ValueError('Refusing symlinked installation directories')
        directory.mkdir(exist_ok=True)
    launcher=bin_dir/'S'
    if launcher.is_symlink():
        raise ValueError('Refusing a symlinked S launcher')
    previous=None
    if launcher.exists():
        if MARKER not in launcher.read_text(encoding='utf-8'):
            raise FileExistsError('Existing S launcher is not managed by this installer; it was preserved')
        previous=bin_dir/('S.previous-'+uuid.uuid4().hex)
        previous.write_bytes(launcher.read_bytes())
        previous.chmod(0o700)
    # Source path and database are quoted as Python literals, never shell text.
    body=(f'#!{python}\n{MARKER}\nimport os, runpy\n'
          f'os.environ.setdefault("AXIOM_DB", {str(data_dir/"continuum.sqlite3")!r})\n'
          f'runpy.run_path({str(source/"scripts"/"s_launcher.py")!r}, run_name="__main__")\n')
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=bin_dir,delete=False) as handle:
            temporary=Path(handle.name)
            handle.write(body)
        temporary.chmod(0o700)
        os.replace(temporary,launcher)
    finally:
        if temporary and temporary.exists():temporary.unlink()
    return {'launcher':str(launcher),'source':str(source),'database':str(data_dir/'continuum.sqlite3'),
            'previous_launcher':str(previous) if previous else None,
            'model_installed':False,'phone_execution_verified':False}


def main():
    prefix=os.environ.get('PREFIX','')
    if not prefix or not (Path(prefix)/'bin'/'termux-info').is_file():
        raise SystemExit('Run this installer inside Termux on Android. No files changed.')
    result=install(Path.home(),prefix,SOURCE,sys.executable)
    print(json.dumps(result,indent=2))
    print('\nStart the local workspace:\n'+shlex.quote(result['launcher'])+' develop application')
    print('Keep Termux running; Ctrl+C stops S. Data stays private to this phone account.')
    print('This installs a Termux launcher, not an APK or a native neural model.')


if __name__=='__main__':main()
