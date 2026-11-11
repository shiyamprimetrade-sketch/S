"""Validated inert source bundles. No model-supplied paths are host paths."""
import hashlib
import json
import re
from pathlib import PurePosixPath

RECIPES = {
 'static-site': {'language':'html', 'required':('index.html',), 'image':'python',
                 'scope':'Static informational page; no backend or interactive business features'},
 'python-cli': {'language':'python','required':('app.py','test_app.py'),'image':'python',
                'scope':'Starter CLI exposing the approved project description'},
 'c-cli': {'language':'c','required':('main.c',),'image':'gcc',
           'scope':'Starter compiled CLI printing the approved project description'},
}
ALLOWED_SUFFIXES={'.py','.html','.css','.js','.json','.md','.c','.h','.txt'}
RESERVED={'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}


def validate_files(files, recipe):
    if recipe not in RECIPES:raise ValueError('Unknown supported recipe')
    if not isinstance(files,dict) or not 1<=len(files)<=20:raise ValueError('Source bundle needs 1–20 files')
    total=0; result={};case_names=set()
    for name,content in files.items():
        if not isinstance(name,str) or len(name)>160 or '\\' in name:raise ValueError('Invalid source filename')
        path=PurePosixPath(name)
        if path.is_absolute() or len(path.parts)>4 or str(path)!=name:raise ValueError('Invalid relative source path')
        if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,60}',p) or p.split('.')[0].upper() in RESERVED for p in path.parts):
            raise ValueError('Reserved or unsafe source path')
        if path.suffix.lower() not in ALLOWED_SUFFIXES:raise ValueError('Unsupported source file type')
        if name.lower() in case_names:raise ValueError('Case-colliding source filenames')
        if name.lower()=='s-provenance.json':raise ValueError('Reserved source-export provenance filename')
        case_names.add(name.lower())
        if not isinstance(content,str) or '\x00' in content:raise ValueError('Source must be UTF-8 text without null bytes')
        size=len(content.encode('utf-8'))
        if size>64000:raise ValueError('Source file exceeds 64 KB')
        total+=size
        if total>256000:raise ValueError('Source bundle exceeds 256 KB')
        result[name]=content
    # A regular file cannot also serve as a directory on extraction.
    for name in result:
        if any(str(p) in result for p in PurePosixPath(name).parents if str(p)!='.'):
            raise ValueError('File/directory source collision')
    if any(name not in result for name in RECIPES[recipe]['required']):raise ValueError('Recipe entry point or tests missing')
    return dict(sorted(result.items()))


def fingerprint(files):
    return hashlib.sha256(json.dumps(files,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def redact(value):
    value=str(value)
    value=re.sub(r'(?i)(api[_-]?key|password|token|secret)(\s*[:=]\s*)[^\s,;]+',r'\1\2[REDACTED]',value)
    value=re.sub(r'\b(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{12,}', '[REDACTED]',value)
    return value[:32000]
