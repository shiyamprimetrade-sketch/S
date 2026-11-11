"""Extensible official-source catalog, not bundled or reimplemented languages.

Release URLs are discovery pointers. No version is labelled latest without a
separate dated verification. Detection is explicit, uses fixed arguments, and
never installs packages. Locally installed programs are trusted operator tools.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import platform
import shutil
import subprocess
import sys


@dataclass(frozen=True)
class Toolchain:
    id: str
    name: str
    executable: str
    version_args: tuple
    suffix: str
    official_source: str
    systems: tuple = ('Windows', 'Linux', 'Darwin')
    requirement: str = 'Official compiler or interpreter; see source for platform prerequisites'
    stable_version: str | None = None
    release_status: str = 'not-verified'
    adapter_status: str = 'catalog-only'
    os_compatibility_status: str = 'platform families only; distribution prerequisites not validated'


# Executable arguments are code-owned; requirement text never becomes a command.
_ROWS = [
 ('python','Python','python',('--version',),'.py','https://www.python.org/downloads/'),
 ('c','C','cc',('--version',),'.c','https://gcc.gnu.org/releases.html'),
 ('cpp','C++','c++',('--version',),'.cpp','https://gcc.gnu.org/releases.html'),
 ('java','Java (JDK)','javac',('-version',),'.java','https://jdk.java.net/'),
 ('javascript','JavaScript','node',('--version',),'.js','https://nodejs.org/en/about/previous-releases'),
 ('typescript','TypeScript','tsc',('--version',),'.ts','https://www.typescriptlang.org/download/'),
 ('php','PHP','php',('--version',),'.php','https://www.php.net/downloads.php'),
 ('rust','Rust','rustc',('--version',),'.rs','https://www.rust-lang.org/tools/install'),
 ('ruby','Ruby','ruby',('--version',),'.rb','https://www.ruby-lang.org/en/downloads/'),
 ('sql','SQL (SQLite adapter)','sqlite3',('--version',),'.sql','https://sqlite.org/download.html'),
 ('go','Go','go',('version',),'.go','https://go.dev/dl/'),
 ('csharp','C# (.NET SDK)','dotnet',('--version',),'.cs','https://dotnet.microsoft.com/download'),
 ('kotlin','Kotlin','kotlinc',('-version',),'.kt','https://kotlinlang.org/docs/releases.html'),
 ('swift','Swift','swift',('--version',),'.swift','https://www.swift.org/install/'),
 ('dart','Dart','dart',('--version',),'.dart','https://dart.dev/get-dart'),
 ('r','R','R',('--version',),'.R','https://cran.r-project.org/'),
 ('julia','Julia','julia',('--version',),'.jl','https://julialang.org/downloads/'),
 ('lua','Lua','lua',('-v',),'.lua','https://www.lua.org/download.html'),
 ('perl','Perl','perl',('--version',),'.pl','https://www.perl.org/get.html'),
 ('bash','Bash','bash',('--version',),'.sh','https://www.gnu.org/software/bash/'),
 ('powershell','PowerShell','pwsh',('--version',),'.ps1','https://github.com/PowerShell/PowerShell/releases'),
 ('scala','Scala','scala',('-version',),'.scala','https://www.scala-lang.org/download/'),
 ('haskell','Haskell (GHC)','ghc',('--version',),'.hs','https://www.haskell.org/ghcup/'),
 ('elixir','Elixir','elixir',('--version',),'.ex','https://elixir-lang.org/install.html'),
 ('erlang','Erlang','erl',('-version',),'.erl','https://www.erlang.org/downloads'),
 ('zig','Zig','zig',('version',),'.zig','https://ziglang.org/download/'),
 ('fortran','Fortran','gfortran',('--version',),'.f90','https://gcc.gnu.org/fortran/'),
 ('cobol','COBOL (GnuCOBOL)','cobc',('--version',),'.cob','https://gnucobol.sourceforge.io/'),
 ('assembly','Assembly (NASM x86)','nasm',('--version',),'.asm','https://www.nasm.us/'),
]
CATALOG = {row[0]: Toolchain(*row) for row in _ROWS}
FRAMEWORKS = [
 {'id': 'pytorch', 'official_source': 'https://pytorch.org/get-started/locally/'},
 {'id': 'tensorflow', 'official_source': 'https://www.tensorflow.org/install'},
 {'id': 'jax', 'official_source': 'https://docs.jax.dev/en/latest/installation.html'},
]


def catalog():
    return {'languages': [asdict(t) for t in CATALOG.values()],
            'frameworks': [dict(f, adapter_status='catalog-only', stable_version=None,
                                release_status='not-verified') for f in FRAMEWORKS],
            'installation': 'Manual official-source review; automatic installation is not implemented',
            'integration': 'Future adapters exchange UTF-8 JSON over stdin/stdout or local HTTP; no binary ABI compatibility claimed'}


def detect(identifier):
    """Explicit CLI-only probe of a trusted installed tool; no project code runs."""
    if identifier not in CATALOG:
        raise ValueError('Unknown toolchain')
    tool = CATALOG[identifier]
    executable = sys.executable if identifier == 'python' else shutil.which(tool.executable)
    result = {'id': identifier, 'os': platform.system(), 'installed': bool(executable),
              'version_output': None, 'error': None}
    if executable:
        # Windows cmd/bat needs a shell and is deliberately not executed here.
        if Path(executable).suffix.lower() in ('.bat', '.cmd'):
            result['error'] = 'Shell wrapper detected; inspect version manually'
            return result
        try:
            run = subprocess.run([executable, *tool.version_args], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 timeout=5, check=False, shell=False)
            result['version_output'] = run.stdout[:2000].decode('utf-8', errors='replace')
            if run.returncode:
                result['error'] = f'Version probe exited {run.returncode}'
        except (OSError, subprocess.TimeoutExpired) as error:
            result['error'] = type(error).__name__
    return result
