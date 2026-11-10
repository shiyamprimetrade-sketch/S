"""Independent causal Transformer. Import requires PyTorch; no Ollama dependency."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class Config:
    vocab_size: int = 256
    dim: int = 256
    layers: int = 4
    heads: int = 4
    kv_heads: int = 2
    context: int = 256

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5).to(x.dtype) * self.scale

def rope(x):
    # [batch, heads, time, dimension], dimension must be even
    t, d = x.size(-2), x.size(-1)
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device).float() / d))
    angles = torch.arange(t, device=x.device).float()[:, None] * inv[None, :]
    c, s = angles.cos()[None, None], angles.sin()[None, None]
    a, b = x.float()[..., ::2], x.float()[..., 1::2]
    return torch.stack((a*c-b*s, a*s+b*c), dim=-1).flatten(-2).to(x.dtype)

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.dim % cfg.heads == 0 and cfg.heads % cfg.kv_heads == 0
        self.h, self.k = cfg.heads, cfg.kv_heads
        self.d = cfg.dim // cfg.heads
        assert self.d % 2 == 0
        self.n1, self.n2 = RMSNorm(cfg.dim), RMSNorm(cfg.dim)
        self.q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.kp = nn.Linear(cfg.dim, self.k*self.d, bias=False)
        self.v = nn.Linear(cfg.dim, self.k*self.d, bias=False)
        self.o = nn.Linear(cfg.dim, cfg.dim, bias=False)
        hidden = ((8*cfg.dim//3 + 63)//64)*64
        self.g = nn.Linear(cfg.dim, hidden, bias=False)
        self.u = nn.Linear(cfg.dim, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.dim, bias=False)
    def forward(self, x):
        b,t,c = x.shape
        y = self.n1(x)
        q = rope(self.q(y).view(b,t,self.h,self.d).transpose(1,2))
        k = rope(self.kp(y).view(b,t,self.k,self.d).transpose(1,2))
        v = self.v(y).view(b,t,self.k,self.d).transpose(1,2)
        k = k.repeat_interleave(self.h//self.k, dim=1)
        v = v.repeat_interleave(self.h//self.k, dim=1)
        z = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        x = x + self.o(z.transpose(1,2).contiguous().view(b,t,c))
        y = self.n2(x)
        return x + self.down(F.silu(self.g(y))*self.u(y))

class SBrain(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
    def forward(self, ids, labels=None):
        if ids.ndim != 2 or not 1 <= ids.size(1) <= self.cfg.context:
            raise ValueError("Expected nonempty [batch, time] within context limit")
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None if labels is None else F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1))
        return logits, loss
    @torch.inference_mode()
    def generate(self, ids, steps=64, temperature=0.8, top_k=40):
        if temperature <= 0 or not 1 <= steps <= 2048 or ids.ndim != 2 or ids.size(0) != 1 or ids.size(1) < 1:
            raise ValueError("Invalid generation arguments")
        self.eval()
        for _ in range(steps):
            scores = self(ids[:, -self.cfg.context:])[0][:, -1, :].float() / temperature
            if top_k > 0:
                threshold = torch.topk(scores, min(top_k, scores.size(-1))).values[:, -1, None]
                scores = scores.masked_fill(scores < threshold, -float("inf"))
            next_id = torch.multinomial(F.softmax(scores, dim=-1), 1)
            ids = torch.cat((ids, next_id), dim=1)
        return ids
