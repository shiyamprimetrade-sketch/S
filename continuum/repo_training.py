"""Build a reproducible source-code corpus from a local Slickk checkout."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

ALLOWED={'.py','.js','.mjs','.ts','.json','.md'}
BLOCKED={'node_modules','.git','dist','build','coverage','.venv','venv','data','assets','models'}
SECRET=re.compile(r'(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)',re.I)
KEYS=re.compile(r'(?im)^\s*(?:api[_-]?key|password|secret|token)\s*[:=]\s*[^\n]+')

def prepare(source:Path,evaluation_fraction=0.2):
    source=Path(source).resolve()
    if not source.is_dir():raise ValueError('Source checkout missing')
    if not 0.05<=evaluation_fraction<=0.5:raise ValueError('Invalid holdout fraction')
    examples=[];excluded=[]
    for file in sorted(source.rglob('*')):
        if not file.is_file() or file.suffix.lower() not in ALLOWED:continue
        rel=file.relative_to(source)
        if any(p in BLOCKED or p.startswith('.') for p in rel.parts):continue
        if file.name in {'.env','package-lock.json','yarn.lock'} or file.stat().st_size>100_000:continue
        if file.is_symlink():continue
        content=file.read_text(encoding='utf-8',errors='replace')
        if SECRET.search(content) or KEYS.search(content):
            excluded.append(str(rel));continue
        if len(content.strip())<100:continue
        digest=hashlib.sha256(content.encode()).hexdigest()
        split='evaluation' if int(digest[:8],16)/0xffffffff<evaluation_fraction else 'training'
        examples.append({'path':rel.as_posix(),'sha256':digest,'split':split,'text':f'File: {rel.as_posix()}\n{content}\n'})
    if not examples:raise ValueError('No eligible source files')
    if not any(x['split']=='training' for x in examples) or not any(x['split']=='evaluation' for x in examples):
        raise ValueError('Need both training and held-out source files')
    manifest={'source':str(source),'files':len(examples),'train_files':sum(x['split']=='training' for x in examples),'eval_files':sum(x['split']=='evaluation' for x in examples),'excluded_sensitive':excluded,'entries':[{k:v for k,v in x.items() if k!='text'} for x in examples]}
    return examples,manifest

def run(source,output,steps=30):
    from .charllm import train
    examples,manifest=prepare(Path(source))
    training='\n'.join(x['text'] for x in examples if x['split']=='training')
    result=train(training,checkpoint=Path(output),steps=steps,context=64,dim=32,heads=4,layers=1,batch=4)
    manifest.update({'training':result,'evaluation_note':'Held-out source files reserved; checkpoint not scored on them. Internal heldout_loss_sample is from training text, not external files.','status':'weights-trained-evaluation-pending'})
    path=Path(output).with_suffix('.manifest.json')
    path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True,help='Local Slickk checkout on slickk-native-runtime')
    p.add_argument('--output',default='models/s-slickk-source.pt')
    p.add_argument('--steps',type=int,default=30)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    if a.prepare_only:
        _,manifest=prepare(Path(a.source));print(json.dumps(manifest,indent=2))
    else:
        m=run(a.source,a.output,a.steps);print(json.dumps({k:v for k,v in m.items() if k!='entries'},indent=2))
if __name__=='__main__':main()
