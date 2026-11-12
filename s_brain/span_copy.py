"""Learned single-span copy controller; no numeric parser or source repair.

The controller has a declared one-copy action budget and 32-byte span limit.
Start/end and the action gate are learned; invalid selections fail, never repair.
"""
import json
import torch
from torch import nn
from .byte_copy import capture_features
from .kv_cache import prefill,append


class SpanCopy(nn.Module):
    def __init__(self,dim,hidden=0):
        super().__init__();self.dim=dim
        def head():
            return nn.Sequential(nn.Linear(dim,hidden),nn.SiLU(),nn.Linear(hidden,1)) if hidden else nn.Linear(dim,1)
        self.start=head();self.end=head();self.gate=head()
        last=self.gate[-1] if hidden else self.gate
        nn.init.zeros_(last.weight);nn.init.constant_(last.bias,-4.0)
    def positions(self,source,mask):
        return (self.start(source).squeeze(-1).masked_fill(~mask,-float('inf')),
                self.end(source).squeeze(-1).masked_fill(~mask,-float('inf')))


@torch.inference_mode()
def generate_span(base,adapter,prompt,budget=200):
    if type(budget) is not int or not 1<=budget<=1024:raise ValueError('Generation budget must be 1–1024 bytes')
    raw=prompt.encode()
    if not raw or len(raw)>base.cfg.context:raise ValueError('Prompt exceeds context; no truncation')
    if base.cfg.dim!=adapter.dim:raise ValueError('Adapter/base mismatch')
    base.eval();adapter.eval();ids=torch.tensor([list(raw)],dtype=torch.long)
    output=bytearray();used=False
    with capture_features(base) as features:
        logits,state=prefill(base,ids)
        starts,ends=adapter.positions(features['hidden'],torch.ones_like(ids,dtype=torch.bool))
        start=int(starts[0].argmax());end=int(ends[0].argmax())
        while len(output)<budget:
            copy=not used and float(adapter.gate(features['hidden'][:,-1]).sigmoid()[0,0])>=0.5
            if copy:
                if not 0<=start<=end<len(raw) or end-start+1>32:raise ValueError('Learned source span outside copy budget')
                chunk=raw[start:end+1];used=True
            else:chunk=bytes([int(logits[0,-1].argmax())])
            for token in chunk:
                if len(output)==budget:break
                output.append(token)
                try:
                    value=json.loads(output.decode())
                    if isinstance(value,dict):return output.decode()
                except (ValueError,UnicodeDecodeError):pass
                logits,state=append(base,torch.tensor([[token]],dtype=torch.long),state)
    return output.decode(errors='replace')
