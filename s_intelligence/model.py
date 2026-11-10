"""Small character-level causal language model with independently initialized weights."""
import torch
from torch import nn
from torch.nn import functional as F

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
    def encode(self, text):
        return [self.stoi[c] for c in text]
    def decode(self, ids):
        return ''.join(self.chars[i] for i in ids)

class SModel(nn.Module):
    def __init__(self, vocab_size, width=64, context=32):
        super().__init__()
        self.context = context
        self.embedding = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(context, width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)
    def forward(self, tokens, targets=None):
        batch, length = tokens.shape
        if length > self.context:
            raise ValueError('Input exceeds context window')
        positions = torch.arange(length, device=tokens.device)
        x = self.embedding(tokens) + self.position(positions)
        mask = torch.triu(torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1)
        a, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        logits = self.head(self.norm(x + a))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, tokens, count=80):
        self.eval()
        for _ in range(count):
            logits, _ = self(tokens[:, -self.context:])
            next_id = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), 1)
            tokens = torch.cat((tokens, next_id), dim=1)
        return tokens

# s_brain/model.py

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BrainConfig:
    vocab_size: int = 16000
    dim: int = 512
    layers: int = 8
    heads: int = 8
    kv_heads: int = 2
    max_context: int = 512
    dropout: float = 0.0
    norm_eps: float = 1e-5


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        )
        return x * scale * self.weight


def rotary_embeddings(x, positions):
    # x: batch, heads, sequence, head_dim
    head_dim = x.shape[-1]
    freq = 1.0 / (
        10000 ** (
            torch.arange(
                0, head_dim, 2,
                device=x.device,
                dtype=torch.float32
            ) / head_dim
        )
    )

    angles = torch.outer(
        positions.float(), freq
    )

    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    even = x[..., 0::2].float()
    odd = x[..., 1::2].float()

    rotated = torch.stack(
        (
            even * cos - odd * sin,
            even * sin + odd * cos
        ),
        dim=-1
    ).flatten(-2)

    return rotated.to(x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.dim % config.heads == 0
        assert config.heads % config.kv_heads == 0

        self.heads = config.heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.dim // config.heads

        assert self.head_dim % 2 == 0

        self.q_proj = nn.Linear(
            config.dim, config.heads * self.head_dim,
            bias=False
        )
        self.k_proj = nn.Linear(
            config.dim, config.kv_heads * self.head_dim,
            bias=False
        )
        self.v_proj = nn.Linear(
            config.dim, config.kv_heads * self.head_dim,
            bias=False
        )
        self.o_proj = nn.Linear(
            config.dim, config.dim, bias=False
        )

        self.dropout = config.dropout

    def forward(self, x):
        batch, seq, dim = x.shape

        q = self.q_proj(x).view(
            batch, seq, self.heads, self.head_dim
        ).transpose(1, 2)

        k = self.k_proj(x).view(
            batch, seq, self.kv_heads, self.head_dim
        ).transpose(1, 2)

        v = self.v_proj(x).view(
            batch, seq, self.kv_heads, self.head_dim
        ).transpose(1, 2)

        positions = torch.arange(
            seq, device=x.device
        )

        q = rotary_embeddings(q, positions)
        k = rotary_embeddings(k, positions)

        repeat = self.heads // self.kv_heads

        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        attention = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )

        attention = attention.transpose(
            1, 2
        ).contiguous().view(batch, seq, dim)

        return self.o_proj(attention)


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()

        hidden = int(8 * dim / 3)
        hidden = 64 * math.ceil(hidden / 64)

        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(
            F.silu(self.gate(x)) * self.up(x)
        )


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn_norm = RMSNorm(
            config.dim, config.norm_eps
        )
        self.attn = GroupedQueryAttention(config)

        self.ffn_norm = RMSNorm(
            config.dim, config.norm_eps
        )
        self.ffn = SwiGLU(config.dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SBrain(nn.Module):
    def __init__(self, config: BrainConfig):
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size, config.dim
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.layers)
        ])

        self.norm = RMSNorm(
            config.dim, config.norm_eps
        )

        self.output = nn.Linear(
            config.dim,
            config.vocab_size,
            bias=False
        )

        # Shared token embedding and output weights
        self.output.weight = self.embedding.weight

        self.apply(self._initialize)

    def _initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, tokens, targets=None):
        if tokens.ndim != 2:
            raise ValueError("Expected [batch, sequence] tokens")

        if tokens.shape[1] > self.config.max_context:
            raise ValueError("Context limit exceeded")

        x = self.embedding(tokens)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.output(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1)
            )

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
        eos_id=None
    ):
        self.eval()

        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError("Generation expects batch size 1")

        if tokens.shape[1] == 0:
            raise ValueError("At least one input token required")

        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        for _ in range(max_new_tokens):
            context = tokens[:, -self.config.max_context:]

            logits, _ = self(context)
            logits = logits[:, -1, :].float()
            logits /= temperature

            if top_k is not None and top_k > 0:
                values, _ = torch.topk(
                    logits,
                    min(top_k, logits.shape[-1])
                )
                cutoff = values[:, -1, None]
                logits = logits.masked_fill(
                    logits < cutoff,
                    float("-inf")
                )

            probabilities = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probabilities, num_samples=1
            )

            tokens = torch.cat(
                [tokens, next_token], dim=1
            )

            if (
                eos_id is not None
                and next_token.item() == eos_id
            ):
                break

        return tokens

