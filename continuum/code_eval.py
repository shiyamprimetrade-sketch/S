"""External held-out next-byte evaluation of S's tiny code language model."""
from __future__ import annotations
import argparse, hashlib, json, math
from collections import Counter
from pathlib import Path
from .repo_training import prepare

def unigram_loss(train_bytes: bytes, eval_bytes: bytes) -> float:
    if not train_bytes or not eval_bytes:
        raise ValueError('Empty corpus')
    counts=Counter(train_bytes)
    total=len(train_bytes)+256
    return -sum(math.log((counts.get(byte,0)+1)/total) for byte in eval_bytes)/len(eval_bytes)

def model_loss(checkpoint: Path, eval_bytes: bytes, max_tokens: int=4096) -> dict:
    from .charllm import torch_module, architecture
    torch=torch_module()
    if max_tokens<2:raise ValueError('max_tokens must be >=2')
    obj=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if obj.get('model_type')!='tiny-causal-byte-transformer-v1':
        raise ValueError('Unsupported checkpoint')
    config=obj['config']
    model=architecture()(**config)
    model.load_state_dict(obj['state_dict'])
    model.eval()
    context=int(config['context'])
    sample=eval_bytes[:max_tokens+1]
    if len(sample)<2:raise ValueError('Insufficient held-out bytes')
    total=0.0;tokens=0
    with torch.no_grad():
        for start in range(1,len(sample),context):
            end=min(start+context,len(sample))
            x=torch.tensor([list(sample[start-1:end-1])],dtype=torch.long)
            y=torch.tensor(list(sample[start:end]),dtype=torch.long)
            logits=model(x)[0]
            total+=float(torch.nn.functional.cross_entropy(logits,y,reduction='sum').item())
            tokens+=len(y)
    return {'nll':total/tokens,'bytes_scored':tokens}

def evaluate(source: Path, checkpoint: Path, max_tokens: int=4096) -> dict:
    rows,manifest=prepare(source)
    train=b'\n'.join(r['text'].encode() for r in rows if r['split']=='training')
    held=b'\n'.join(r['text'].encode() for r in rows if r['split']=='evaluation')
    train_hashes={r['sha256'] for r in rows if r['split']=='training'}
    held_hashes={r['sha256'] for r in rows if r['split']=='evaluation'}
    if train_hashes & held_hashes:raise ValueError('Duplicate content crosses train/eval split')
    scored=held[:max_tokens+1]
    if len(scored)<2:raise ValueError('Held-out corpus too small')
    baseline=unigram_loss(train,scored[1:])
    candidate=model_loss(Path(checkpoint),held,max_tokens)
    return {'schema':1,'source_file_count':manifest['files'],'training_files':manifest['train_files'],'evaluation_files':manifest['eval_files'],'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),'holdout_sha256':hashlib.sha256(held).hexdigest(),'tokens_scored':candidate['bytes_scored'],'model_nll':candidate['nll'],'unigram_nll':baseline,'model_better_than_unigram':candidate['nll']<baseline,'evaluation_type':'external-source-heldout-next-byte-prediction','limitation':'Not a code-generation or invention benchmark; near-duplicate files can still leak.'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--max-tokens',type=int,default=4096)
    p.add_argument('--output',default='reports/S-code-heldout-evaluation.json')
    args=p.parse_args()
    report=evaluate(Path(args.source),Path(args.checkpoint),args.max_tokens)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
