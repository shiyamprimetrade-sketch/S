"""Generate from an independently trained S checkpoint without Ollama."""
import argparse
import torch
from .model import SModel


def run(checkpoint, prompt='S ', count=80, seed=20261110):
    if not isinstance(count, int) or count < 0:
        raise ValueError('Generation count must be a nonnegative integer')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    chars = state['vocabulary']
    indices = {c: i for i, c in enumerate(chars)}
    if not prompt or any(c not in indices for c in prompt):
        raise ValueError('Prompt must be nonempty and contain only trained vocabulary')
    model = SModel(len(chars), state['width'], state['context'])
    model.load_state_dict(state['weights'])
    torch.manual_seed(seed)
    result = model.generate(torch.tensor([[indices[c] for c in prompt]]), count=count)
    text = ''.join(chars[i] for i in result[0].tolist())
    print(text)
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='artifacts/s-native.pt')
    parser.add_argument('--prompt', default='S ')
    parser.add_argument('--count', type=int, default=80)
    parser.add_argument('--seed', type=int, default=20261110)
    args = parser.parse_args()
    run(args.checkpoint, args.prompt, args.count, args.seed)
