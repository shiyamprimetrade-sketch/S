"""Reproducible CPU experiment. Training corpus is demonstrative, not evidence of general intelligence."""
import argparse
import json
from pathlib import Path
import torch
from .model import CharTokenizer, SModel

CORPUS = ('S observes a pattern. S tests the pattern. S records the evidence.\\n' * 120)

def run(steps=80, output='artifacts/s-native.pt'):
    torch.manual_seed(20261110)
    tok = CharTokenizer(CORPUS)
    ids = torch.tensor(tok.encode(CORPUS), dtype=torch.long)
    cut = int(len(ids) * .9)
    train, validation = ids[:cut], ids[cut:]
    model = SModel(len(tok.chars))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    def sample(data, batch=16, context=32):
        starts = torch.randint(0, len(data)-context-1, (batch,))
        return torch.stack([data[i:i+context] for i in starts]), torch.stack([data[i+1:i+context+1] for i in starts])
    with torch.no_grad():
        x, y = sample(validation)
        _, initial = model(x, y)
    model.train()
    for _ in range(steps):
        x, y = sample(train)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        x, y = sample(validation)
        _, final = model(x, y)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'weights': model.state_dict(), 'vocabulary': tok.chars, 'context': model.context, 'width':64, 'seed':20261110}, path)
    result = {'steps':steps,'initial_validation_loss':float(initial),'final_validation_loss':float(final),'parameters':sum(p.numel() for p in model.parameters()),'checkpoint':str(path),'limitation':'Repeated synthetic text; validation shares patterns with training'}
    print(json.dumps(result, indent=2))
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=80)
    p.add_argument('--output', default='artifacts/s-native.pt')
    args = p.parse_args()
    run(args.steps,args.output)
