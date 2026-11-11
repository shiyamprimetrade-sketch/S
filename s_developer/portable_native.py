"""Bounded standard-library inference for small existing SBrain checkpoints.

No torch, numpy, pickle, downloads or model conversion. This is a restricted
F32 safetensors reader and the same Transformer equations, using Python floats.
It is intended for runtime diagnostics on phones, not capable general coding.
"""
import json
import hashlib
import math
from pathlib import Path
import struct
import time


def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate safetensors header key')
        result[key]=value
    return result


def _bad_constant(value):raise ValueError('Nonstandard JSON constant')


def read_f32(path,expected,expected_digest=None):
    path=Path(path)
    if path.stat().st_size>4000000:raise ValueError('Portable weights exceed 4 MB budget')
    raw=path.read_bytes()
    if expected_digest is not None and hashlib.sha256(raw).hexdigest()!=expected_digest:
        raise ValueError('Portable checkpoint digest mismatch')
    if len(raw)<10:raise ValueError('Truncated safetensors file')
    length=struct.unpack('<Q',raw[:8])[0]
    if not 2<=length<=64000 or 8+length>len(raw) or raw[8]!=123:raise ValueError('Invalid safetensors header length or prefix')
    header=json.loads(raw[8:8+length].decode('utf-8'),object_pairs_hook=unique_object,parse_constant=_bad_constant)
    if not isinstance(header,dict):raise ValueError('Invalid safetensors header')
    metadata=header.pop('__metadata__',{})
    if not isinstance(metadata,dict) or any(not isinstance(v,str) for v in metadata.values()):
        raise ValueError('Invalid safetensors metadata')
    if metadata.get('lm_head.weight','embed.weight')!='embed.weight':raise ValueError('Unsupported tied output weights')
    if set(header)!=set(expected):raise ValueError('Unexpected portable checkpoint tensor names')
    payload=raw[8+length:];ranges=[];weights={}
    for name,shape in expected.items():
        item=header[name]
        if not isinstance(item,dict) or set(item)!={'dtype','shape','data_offsets'}:
            raise ValueError('Invalid tensor descriptor')
        if item['dtype']!='F32' or item['shape']!=shape or any(type(v) is not int for v in item['shape']):
            raise ValueError('Portable inference requires exact expected F32 shapes')
        offsets=item['data_offsets']
        if not isinstance(offsets,list) or len(offsets)!=2 or any(type(v) is not int for v in offsets):
            raise ValueError('Invalid tensor offsets')
        start,end=offsets;count=math.prod(shape)
        if not 0<=start<=end<=len(payload) or end-start!=count*4:raise ValueError('Tensor byte range mismatch')
        ranges.append((start,end))
        values=struct.unpack('<'+str(count)+'f',payload[start:end])
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite portable weights')
        weights[name]=list(values) if len(shape)==1 else [values[i:i+shape[1]] for i in range(0,count,shape[1])]
    cursor=0
    for start,end in sorted(ranges):
        if start!=cursor:raise ValueError('Overlapping tensors or uncovered payload bytes')
        cursor=end
    if cursor!=len(payload):raise ValueError('Trailing unindexed tensor data')
    return weights


def mv(matrix,vector):return [sum(a*b for a,b in zip(row,vector)) for row in matrix]
def add(a,b):return [x+y for x,y in zip(a,b)]
def norm(vector,scale):
    inverse=1/math.sqrt(sum(x*x for x in vector)/len(vector)+1e-5)
    return [x*inverse*s for x,s in zip(vector,scale)]
def silu(value):
    if value>=0:return value/(1+math.exp(-value))
    exponent=math.exp(value);return value*exponent/(1+exponent)
def rotate(head,position):
    result=[];width=len(head)
    for i in range(0,width,2):
        angle=position/(10000**(i/width));c,s=math.cos(angle),math.sin(angle)
        a,b=head[i:i+2];result.extend((a*c-b*s,a*s+b*c))
    return result


class PortableBrain:
    def __init__(self,checkpoint):
        from .models import validate_native
        self.manifest=validate_native(checkpoint);self.cfg=self.manifest.model_config
        c=self.cfg
        if c['dim']>64 or c['layers']>2 or c['context']>256 or c['heads']>8:
            raise ValueError('Portable runtime supports only small models: dim<=64, layers<=2, context<=256, heads<=8')
        self.dim=c['dim'];self.heads=c['heads'];self.kv=c['kv_heads'];self.width=self.dim//self.heads
        hidden=((8*self.dim//3+63)//64)*64
        shapes={'embed.weight':[256,self.dim],'norm.scale':[self.dim]}
        for layer in range(c['layers']):
            p=f'blocks.{layer}.'
            shapes.update({p+'n1.scale':[self.dim],p+'n2.scale':[self.dim],p+'q.weight':[self.dim,self.dim],
                p+'kp.weight':[self.kv*self.width,self.dim],p+'v.weight':[self.kv*self.width,self.dim],
                p+'o.weight':[self.dim,self.dim],p+'g.weight':[hidden,self.dim],p+'u.weight':[hidden,self.dim],
                p+'down.weight':[self.dim,hidden]})
        self.weights=read_f32(Path(checkpoint)/'model.safetensors',shapes,self.manifest.weights_sha256)
    def new_cache(self):return {'position':0,'layers':[{'keys':[],'values':[]} for _ in range(self.cfg['layers'])]}
    def step(self,token,cache):
        if type(token) is not int or not 0<=token<256:raise ValueError('Token must be a byte')
        position=cache['position']
        if position>=self.cfg['context']:raise ValueError('Portable context budget exhausted; start a new request')
        x=list(self.weights['embed.weight'][token])
        for layer,state in enumerate(cache['layers']):
            p=f'blocks.{layer}.';w=self.weights;y=norm(x,w[p+'n1.scale'])
            query=mv(w[p+'q.weight'],y);keys=mv(w[p+'kp.weight'],y);values=mv(w[p+'v.weight'],y)
            query=[rotate(query[i:i+self.width],position) for i in range(0,self.dim,self.width)]
            keys=[rotate(keys[i:i+self.width],position) for i in range(0,len(keys),self.width)]
            values=[values[i:i+self.width] for i in range(0,len(values),self.width)]
            state['keys'].append(keys);state['values'].append(values);attention=[]
            for head in range(self.heads):
                group=head//(self.heads//self.kv)
                scores=[sum(a*b for a,b in zip(query[head],k[group]))/math.sqrt(self.width) for k in state['keys']]
                peak=max(scores);exp=[math.exp(s-peak) for s in scores];total=sum(exp)
                probabilities=[v/total for v in exp]
                attention.extend(sum(prob*value[group][j] for prob,value in zip(probabilities,state['values'])) for j in range(self.width))
            x=add(x,mv(w[p+'o.weight'],attention));y=norm(x,w[p+'n2.scale'])
            g,u=mv(w[p+'g.weight'],y),mv(w[p+'u.weight'],y)
            x=add(x,mv(w[p+'down.weight'],[silu(a)*b for a,b in zip(g,u)]))
        logits=mv(self.weights['embed.weight'],norm(x,self.weights['norm.scale']))
        if not all(math.isfinite(v) for v in logits):raise ValueError('Nonfinite neural computation')
        cache['position']+=1
        return logits
    def generate(self,prompt,tokens=16,timeout=5):
        if not isinstance(prompt,str) or not 1<=len(prompt.encode())<=64:raise ValueError('Portable prompt must be 1–64 UTF-8 bytes')
        if type(tokens) is not int or not 1<=tokens<=128:raise ValueError('Portable generation budget must be 1–128 tokens')
        data=list(prompt.encode())
        if len(data)+tokens>self.cfg['context']:raise ValueError('Prompt and output exceed model context')
        if not isinstance(timeout,(int,float)) or not 0<timeout<=10:raise ValueError('Invalid runtime deadline')
        start=time.monotonic();deadline=start+timeout;cache=self.new_cache();generated=[]
        for token in data:
            if time.monotonic()>deadline:raise TimeoutError('Portable neural inference exceeded its time budget')
            logits=self.step(token,cache)
        for i in range(tokens):
            if time.monotonic()>deadline:raise TimeoutError('Portable neural inference exceeded its time budget')
            token=max(range(256),key=logits.__getitem__);generated.append(token)
            if i+1<tokens:logits=self.step(token,cache)
        return {'output':bytes(generated).decode('utf-8',errors='replace'),'generated_byte_ids':generated,
                'generated_tokens':len(generated),'elapsed_seconds':round(time.monotonic()-start,4),
                'runtime':'python-stdlib-f32','decoding':'greedy','weights_sha256':self.manifest.weights_sha256,
                'inference_completed':True,'coding_capability':'unverified; runtime execution is not an intelligence or code-quality pass'}
