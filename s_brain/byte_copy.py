"""Learned prompt-byte attention/gate adapter on a frozen SBrain.

Inference sees prompt bytes and neural activations only: no number extraction,
expected answer, AST rewrite, syntax constraint, or repair is available here.
"""
from contextlib import contextmanager
import json
import math
import torch
from torch import nn
from .kv_cache import prefill, append


class ByteCopy(nn.Module):
    def __init__(self, dim, key_dim=32):
        super().__init__()
        self.query=nn.Linear(dim,key_dim,bias=False)
        self.key=nn.Linear(dim,key_dim,bias=False)
        self.gate=nn.Linear(dim,1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias,-4.0)
        self.dim=dim;self.key_dim=key_dim

    def distribution(self, hidden, logits, source_hidden, source_bytes, source_mask, gate_override=None):
        scores=self.query(hidden) @ self.key(source_hidden).transpose(-1,-2)/math.sqrt(self.key_dim)
        scores=scores.masked_fill(~source_mask[:,None,:],-float('inf'))
        attention=scores.softmax(-1)
        copied=torch.zeros((*hidden.shape[:2],256),device=hidden.device,dtype=hidden.dtype)
        indices=source_bytes[:,None,:].expand_as(attention)
        copied.scatter_add_(-1,indices,attention)
        gate=self.gate(hidden).sigmoid()
        if gate_override is not None:
            if gate_override not in (0,1):raise ValueError('Gate override is an explicit ablation only')
            gate=torch.full_like(gate,float(gate_override))
        probabilities=(1-gate)*logits.softmax(-1)+gate*copied
        return probabilities,scores,gate


@contextmanager
def capture_features(base):
    """Reuse unchanged full/cached SBrain inference and remove the temporary hook."""
    captured={}
    def receive(module,args,output):captured['hidden']=output
    handle=base.norm.register_forward_hook(receive)
    try:yield captured
    finally:handle.remove()


@torch.no_grad()
def frozen_features(base, ids):
    with capture_features(base) as captured:
        logits,_=base(ids)
        return logits.detach(),captured['hidden'].detach()


@torch.inference_mode()
def generate_copy(base, adapter, prompt, budget=200, *, gate_override=None):
    if type(budget) is not int or not 1<=budget<=1024:raise ValueError('Generation budget must be 1–1024 bytes')
    raw=prompt.encode()
    if not raw or len(raw)>base.cfg.context:raise ValueError('Prompt exceeds context; no truncation')
    if adapter.dim!=base.cfg.dim:raise ValueError('Adapter/base dimension mismatch')
    base.eval();adapter.eval()
    ids=torch.tensor([list(raw)],dtype=torch.long)
    mask=torch.ones_like(ids,dtype=torch.bool)
    output=bytearray()
    with capture_features(base) as features:
        logits,state=prefill(base,ids)
        source_hidden=features['hidden'].clone()
        for _ in range(budget):
            hidden=features['hidden'][:,-1:]
            p,_,_=adapter.distribution(hidden,logits[:,-1:],source_hidden,ids,mask,gate_override)
            token=int(p[0,-1].argmax());output.append(token)
            try:
                value=json.loads(output.decode())
                if isinstance(value,dict):return output.decode()
            except (ValueError,UnicodeDecodeError):pass
            logits,state=append(base,torch.tensor([[token]],dtype=torch.long),state)
    return output.decode(errors='replace')
