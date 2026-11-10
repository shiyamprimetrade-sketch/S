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
