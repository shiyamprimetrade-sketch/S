"""Generate from an independently trained S checkpoint without Ollama."""
import argparse
import torch
from .model import SModel

def run(checkpoint, prompt='S ', count=80):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    chars = state['vocabulary']
    indices = {c:i for i,c in enumerate(chars)}
    if not prompt or any(c not in indices for c in prompt):
        raise ValueError('Prompt must be nonempty and contain only trained vocabulary')
    model = SModel(len(chars), state['width'], state['context'])
    model.load_state_dict(state['weights'])
    torch.manual_seed(20261110)
    result = model.generate(torch.tensor([[indices[c] for c in prompt]]), count=count)
    print(''.join(chars[i] for i in result[0].tolist()))

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='artifacts/s-native.pt')
    p.add_argument('--prompt',default='S ')
    p.add_argument('--count',type=int,default=80)
    a=p.parse_args()
    run(a.checkpoint,a.prompt,a.count)
