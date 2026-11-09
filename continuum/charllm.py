"""Experimental *from-scratch* UTF-8 byte-token causal Transformer.

This is a tiny language model, not GPT-6 and not a replacement for a
well-trained instruction-tuned local LLM. No cloud or weights downloads.
It can be trained on the user's *private* imported conversations locally.
"""
from __future__ import annotations
import argparse
import gzip
import json
import math
import os
import random
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
DEFAULT_CKPT=ROOT/'models'/'axiom_byte_lm.pt'

class ByteTokenizer:
    vocab_size=256
    def encode(self,text):return list(str(text).encode('utf-8'))
    def decode(self,ids):return bytes(int(i) for i in ids).decode('utf-8',errors='replace')

def torch_module():
    try:
        import torch
        return torch
    except ImportError as e:
        raise RuntimeError('This model needs PyTorch. Install: py -3 -m pip install torch') from e

def architecture():
    torch=torch_module()
    nn=torch.nn
    class ByteGPT(nn.Module):
        def __init__(self,context=96,dim=64,heads=4,layers=2):
            super().__init__()
            self.context=context;self.dim=dim;self.heads=heads;self.layers=layers
            self.tokens=nn.Embedding(256,dim)
            self.positions=nn.Embedding(context,dim)
            self.blocks=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim,nhead=heads,dim_feedforward=dim*4,dropout=0,batch_first=True,norm_first=True),num_layers=layers,enable_nested_tensor=False)
            self.norm=nn.LayerNorm(dim)
            self.output=nn.Linear(dim,256)
        def forward(self,x):
            n=x.shape[1]
            if n>self.context:raise ValueError('Too many tokens')
            pos=torch.arange(n,device=x.device)
            h=self.tokens(x)+self.positions(pos)
            mask=torch.triu(torch.ones(n,n,device=x.device,dtype=torch.bool),diagonal=1)
            return self.output(self.norm(self.blocks(h,mask=mask)))
    return ByteGPT

def load_text(kind='synthetic',db=None,project=None):
    if kind=='private':
        if not db or not project:raise ValueError('Private training needs --db and --project')
        from .storage import Store
        store=Store(db);store.get_project(project)
        chunks=[]
        for ses in store.list_sessions(project):
            for m in store.list_messages(ses['id'],100000):
                role='Teacher' if m['role']=='user' else 'Student'
                chunks.append(f'{role}: {m["content"]}\n')
        if len(chunks)<3:raise ValueError('Import or create conversations before training')
        return '\n'.join(chunks)
    path=ROOT/'data'/'simulations'/'teacher_student_4000.jsonl.gz'
    text=[]
    with gzip.open(path,'rt',encoding='utf8') as f:
        for line in f:
            row=json.loads(line)
            if row.get('split')=='train':text.append(f'Teacher: {row["teacher"]}\nStudent: {row["student"]}\n')
    return '\n'.join(text)

def train(text,checkpoint=DEFAULT_CKPT,steps=100,context=96,dim=64,heads=4,layers=2,batch=12,seed=41):
    torch=torch_module()
    torch.manual_seed(seed);random.seed(seed)
    if not (16<=context<=512 and 16<=dim<=256 and dim%heads==0 and 1<=layers<=6 and 1<=steps<=20000 and 1<=batch<=256):raise ValueError('Invalid training size')
    data=torch.tensor(ByteTokenizer().encode(text),dtype=torch.long)
    if len(data)<context+16:raise ValueError('Corpus too short for context')
    cut=max(context+4,int(0.9*len(data)))
    if len(data)-cut<=context:cut=len(data)-context-2
    train_data=data[:cut];valid_data=data[cut:]
    model=architecture()(context,dim,heads,layers)
    opt=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
    def batch_sample(src,b):
        offsets=torch.randint(0,len(src)-context-1,(b,))
        x=torch.stack([src[i:i+context] for i in offsets])
        y=torch.stack([src[i+1:i+context+1] for i in offsets])
        return x,y
    model.train();loss_value=0.0
    for step in range(steps):
        x,y=batch_sample(train_data,batch)
        logits=model(x)
        loss=torch.nn.functional.cross_entropy(logits.reshape(-1,256),y.reshape(-1))
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        opt.step()
        loss_value=float(loss.item())
    model.eval()
    with torch.no_grad():
        x,y=batch_sample(valid_data,1)
        validation=float(torch.nn.functional.cross_entropy(model(x).reshape(-1,256),y.reshape(-1)).item())
    out=Path(checkpoint);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'config':{'context':context,'dim':dim,'heads':heads,'layers':layers},
                'metrics':{'train_loss_last':loss_value,'heldout_loss_sample':validation,'steps':steps,'bytes':len(data),'seed':seed},
                'model_type':'tiny-causal-byte-transformer-v1'},out)
    return {'checkpoint':str(out),'parameters':sum(x.numel() for x in model.parameters()),'train_loss_last':round(loss_value,4),
            'heldout_loss_sample':round(validation,4),'steps':steps,'corpus_bytes':len(data),'seed':seed}

def complete(prompt,checkpoint=DEFAULT_CKPT,max_new_tokens=200,temperature=0.85,seed=None):
    torch=torch_module()
    if not Path(checkpoint).exists():raise ValueError('No trained character model. Run: py -3 -m continuum.charllm train --steps 200')
    if not 1<=max_new_tokens<=2048 or not 0.1<=temperature<=2:raise ValueError('Bad generation parameters')
    obj=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if obj.get('model_type')!='tiny-causal-byte-transformer-v1':raise ValueError('Unsupported checkpoint')
    model=architecture()(**obj['config'])
    model.load_state_dict(obj['state_dict']);model.eval()
    ids=ByteTokenizer().encode(prompt)
    if not ids:ids=[32]
    generated=[]
    if seed is not None:torch.manual_seed(seed)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            window=torch.tensor([ids[-model.context:]],dtype=torch.long)
            logits=model(window)[0,-1,:]/temperature
            probs=torch.softmax(logits,dim=-1)
            choice=int(torch.multinomial(probs,1).item())
            ids.append(choice);generated.append(choice)
            # Two line-breaks usually finish a student paragraph.
            if len(generated)>48 and bytes(generated[-2:])==b'\n\n':break
    return ByteTokenizer().decode(generated).strip()

def main(argv=None):
    p=argparse.ArgumentParser(description='Axiom byte-token causal language model laboratory')
    sub=p.add_subparsers(dest='cmd',required=True)
    t=sub.add_parser('train');t.add_argument('--steps',type=int,default=100);t.add_argument('--context',type=int,default=96)
    t.add_argument('--dim',type=int,default=64);t.add_argument('--batch',type=int,default=12)
    t.add_argument('--corpus',choices=['synthetic','private'],default='synthetic');t.add_argument('--db');t.add_argument('--project')
    t.add_argument('--output',default=str(DEFAULT_CKPT))
    g=sub.add_parser('generate');g.add_argument('prompt');g.add_argument('--checkpoint',default=str(DEFAULT_CKPT));g.add_argument('--tokens',type=int,default=200)
    args=p.parse_args(argv)
    if args.cmd=='train':
        text=load_text(args.corpus,args.db,args.project)
        print(json.dumps(train(text,steps=args.steps,context=args.context,dim=args.dim,batch=args.batch,checkpoint=args.output),indent=2))
    else:print(complete('Teacher: '+args.prompt+'\nStudent: ',checkpoint=args.checkpoint,max_new_tokens=args.tokens))

if __name__=='__main__':main()
